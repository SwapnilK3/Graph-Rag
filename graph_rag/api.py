"""
api.py
------
Graph-RAG API — V3.2 Multi-KG Platform

Endpoints:
  GET  /health              Health check
  POST /graphs/connect      Register a new knowledge graph
  GET  /graphs              List all registered KGs
  GET  /graphs/{kg_id}      Get KG details + schema
  DELETE /graphs/{kg_id}    Remove a KG
  POST /query               Fast single-pass query (V2.5)
  POST /agent/query         Agentic multi-step query (V3)
  POST /memory/recall       Search episodic memory
  GET  /memory/stats        Memory statistics
"""

import asyncio
import logging
from functools import partial
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .graph_registry import GraphRegistry
from .memory import AgentMemory
from .config import NEO4J_DATABASE

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Graph-RAG API", version="3.2.0")

# Enable CORS for frontend microservice
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Registry (replaces old domain-based caching) ─────────────────────
registry = GraphRegistry()


@app.on_event("startup")
async def startup_event():
    """Register the default KG from .env on startup."""
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, registry.ensure_default)
        await loop.run_in_executor(None, partial(registry.get_pipeline, "default"))
        try:
            await loop.run_in_executor(None, partial(registry.get_agent, "default"))
            logger.info("Default KG registration + pipeline + memory layer ready")
        except Exception as agent_init_error:
            logger.warning("Default agent prewarm skipped: %s", agent_init_error)
            logger.info("Default KG registration + pipeline ready")
    except Exception as e:
        logger.warning("Default KG startup failed: %s", e)


# ── Request/Response Models ───────────────────────────────────────────

class ConnectRequest(BaseModel):
    uri: str
    username: str
    password: str
    database: str = NEO4J_DATABASE or "neo4j"
    name: str = ""


class QueryRequest(BaseModel):
    query: str
    kg_id: str = "default"


class QueryResponse(BaseModel):
    query: str
    answer: Optional[str]
    context: str
    entities: list[str]
    intent: str
    strategy: str
    hop_depth: int = 0
    subgraph: Optional[dict] = None
    thought_process: list[dict] = []
    timing: dict


class AgentQueryResponse(BaseModel):
    query: str
    answer: str
    sub_results: list[dict] = []
    plan_history: list[dict] = []
    memory_hit: bool = False
    iterations: int = 0
    timing: dict = {}


class MemoryRecallRequest(BaseModel):
    query: str
    kg_id: str = "default"
    threshold: float = 0.85


class MemoryRecallResponse(BaseModel):
    results: list[dict] = []


# ── Graph Management Endpoints ────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "3.2.0"}


@app.post("/graphs/connect")
async def connect_graph(request: ConnectRequest):
    """
    Register a new knowledge graph.

    Connects, discovers schema, generates config, persists credentials.
    Returns the kg_id and schema summary.
    """
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            partial(
                registry.register,
                uri=request.uri,
                username=request.username,
                password=request.password,
                database=request.database,
                name=request.name,
            ),
        )

        # Prewarm pipeline immediately so generated config is validated and active.
        kg_id = result["kg_id"]
        await loop.run_in_executor(None, partial(registry.get_pipeline, kg_id))
        result["pipeline_initialized"] = True

        # Prewarm memory-layer agent when LLM is available.
        try:
            await loop.run_in_executor(None, partial(registry.get_agent, kg_id))
            result["memory_layer_initialized"] = True
        except Exception as agent_init_error:
            logger.warning("Agent prewarm skipped for %s: %s", kg_id, agent_init_error)
            result["memory_layer_initialized"] = False
            result["memory_layer_status"] = str(agent_init_error)

        return result
    except ConnectionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Graph connection failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graphs")
async def list_graphs():
    """List all registered knowledge graphs."""
    try:
        return {"graphs": registry.list_all()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graphs/{kg_id}")
async def get_graph(kg_id: str):
    """Get details for a specific KG."""
    kg = registry.get(kg_id)
    if kg is None:
        raise HTTPException(status_code=404, detail=f"KG '{kg_id}' not found")

    # Don't expose password in response
    import json
    schema = json.loads(kg.get("schema_json", "{}") or "{}")
    return {
        "kg_id": kg["kg_id"],
        "name": kg["name"],
        "uri": kg["uri"],
        "database": kg["database"],
        "status": kg["status"],
        "created_at": kg["created_at"],
        "last_used": kg["last_used"],
        "schema": {
            "node_labels": schema.get("node_labels", []),
            "total_nodes": schema.get("total_nodes", 0),
            "total_relationships": schema.get("total_relationships", 0),
            "relationships": schema.get("relationships", []),
        },
    }


@app.delete("/graphs/{kg_id}")
async def delete_graph(kg_id: str):
    """Remove a knowledge graph from the registry."""
    if kg_id == "default":
        raise HTTPException(status_code=400, detail="Cannot delete the default KG")
    deleted = registry.delete(kg_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"KG '{kg_id}' not found")
    return {"status": "deleted", "kg_id": kg_id}


# ── Query Endpoints ───────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
async def query_graph(request: QueryRequest):
    """
    V2.5 Pipeline query. Fast, single-pass retrieval.
    Routes to the correct KG via kg_id.
    """
    try:
        pipeline = registry.get_pipeline(request.kg_id)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, pipeline.query, request.query)

        return QueryResponse(
            query=result["query"],
            answer=result["answer"],
            context=result["context"],
            entities=result["entry_nodes"],
            intent=result["intent"],
            strategy=result["strategy"],
            hop_depth=result.get("hop_depth", 0),
            subgraph=result.get("subgraph"),
            thought_process=result.get("thought_process", []),
            timing=result["timing"],
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Error processing query")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


@app.post("/agent/query", response_model=AgentQueryResponse)
async def agent_query(request: QueryRequest):
    """
    V3 Agentic query. Multi-step reasoning with memory and planning.
    Routes to the correct KG via kg_id. Memory is scoped per kg_id.
    """
    try:
        agent = registry.get_agent(request.kg_id)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, agent.execute, request.query, request.kg_id
        )

        return AgentQueryResponse(
            query=result.query,
            answer=result.answer,
            sub_results=result.sub_results,
            plan_history=result.plan_history,
            memory_hit=result.memory_hit,
            iterations=result.iterations,
            timing=result.timing,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Agent query error")
        raise HTTPException(status_code=500, detail=str(e))


# ── Memory Endpoints ─────────────────────────────────────────────────

@app.post("/memory/recall", response_model=MemoryRecallResponse)
async def recall_memory(request: MemoryRecallRequest):
    """Search agent memory for similar past queries (scoped to kg_id)."""
    try:
        agent = registry.get_agent(request.kg_id)
        results = agent.memory.recall(
            query=request.query,
            domain=request.kg_id,
            threshold=request.threshold,
        )
        return MemoryRecallResponse(
            results=[{
                "query": r.query,
                "answer": r.answer,
                "similarity": round(r.similarity, 3),
                "age_seconds": round(r.age_seconds, 1),
                "intent": r.intent,
            } for r in results]
        )
    except Exception as e:
        logger.warning("Memory recall error: %s", e)
        return MemoryRecallResponse(results=[])


@app.get("/memory/stats")
async def memory_stats(kg_id: str = "default"):
    """Return memory usage statistics for a specific KG."""
    try:
        agent = registry.get_agent(kg_id)
        return agent.memory.stats(kg_id)
    except Exception:
        return {"total_memories": 0, "memories_last_hour": 0, "domain": kg_id}


# ── Legacy Compatibility ─────────────────────────────────────────────

@app.get("/schema")
async def get_schema(kg_id: str = "default"):
    """Return the discovered schema for a KG."""
    try:
        pipeline = registry.get_pipeline(kg_id)
        return {
            "node_labels": pipeline.schema.get("node_labels", []),
            "relationships": pipeline.schema.get("relationships", []),
            "node_counts": pipeline.schema.get("node_counts", {}),
            "total_nodes": pipeline.schema.get("total_nodes", 0),
            "total_relationships": pipeline.schema.get("total_relationships", 0),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
