import asyncio
import os
import logging
from typing import Optional, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .pipeline import GraphRAGPipeline
from .connector import GraphDBConnector
from .agent import GraphRAGAgent
from .memory import AgentMemory
from .evaluator import RAGEvaluator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Graph-RAG API", version="3.0.0")

# Enable CORS for frontend microservice
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Cache ─────────────────────────────────────────────────────────────
pipelines: dict[str, GraphRAGPipeline] = {}
agents: dict[str, GraphRAGAgent] = {}
memory_store: Optional[AgentMemory] = None


def _get_pipeline(domain: str) -> GraphRAGPipeline:
    """Get or create a cached pipeline for a domain."""
    if domain not in pipelines:
        logger.info("Initializing pipeline for domain: %s", domain)
        pipelines[domain] = GraphRAGPipeline(domain=domain)
    return pipelines[domain]


def _get_agent(domain: str) -> GraphRAGAgent:
    """Get or create a cached agent for a domain."""
    global memory_store
    if domain not in agents:
        pipeline = _get_pipeline(domain)
        if not pipeline.llm:
            raise HTTPException(status_code=503, detail="LLM not configured — agent requires LLM")
        if memory_store is None:
            memory_store = AgentMemory(llm=pipeline.llm)
        agents[domain] = GraphRAGAgent(
            pipeline=pipeline,
            memory=memory_store,
            llm=pipeline.llm,
        )
    return agents[domain]


# ── Models ────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    domain: str = "medical"


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
    domain: str = "default"
    threshold: float = 0.85


class MemoryRecallResponse(BaseModel):
    results: list[dict] = []


# ── Endpoints ─────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "3.0.0"}


@app.post("/query", response_model=QueryResponse)
async def query_graph(request: QueryRequest):
    """
    V2.5 Pipeline query. Fast, single-pass retrieval.
    Use /agent/query for multi-step agentic reasoning.
    """
    try:
        pipeline = _get_pipeline(request.domain)
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
            timing=result["timing"]
        )
    except FileNotFoundError as e:
        logger.error("Domain config not found: %s", e)
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Error processing query")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


@app.post("/agent/query", response_model=AgentQueryResponse)
async def agent_query(request: QueryRequest):
    """
    V3 Agentic query. Multi-step reasoning with memory and planning.
    Slower but more thorough than /query.
    """
    try:
        agent = _get_agent(request.domain)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, agent.execute, request.query, request.domain
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
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Agent query error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/recall", response_model=MemoryRecallResponse)
async def recall_memory(request: MemoryRecallRequest):
    """Search agent memory for similar past queries."""
    global memory_store
    if memory_store is None:
        return MemoryRecallResponse(results=[])
    
    try:
        results = memory_store.recall(
            query=request.query,
            domain=request.domain,
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


@app.get("/schema")
async def get_schema():
    """Return the discovered schema for the default pipeline."""
    try:
        pipeline = _get_pipeline("medical")
        return {
            "node_labels": pipeline.schema.get("node_labels", []),
            "relationships": pipeline.schema.get("relationships", []),
            "node_counts": pipeline.schema.get("node_counts", {}),
            "total_nodes": pipeline.schema.get("total_nodes", 0),
            "total_relationships": pipeline.schema.get("total_relationships", 0),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory/stats")
async def memory_stats(domain: str = "default"):
    """Return memory usage statistics."""
    global memory_store
    if memory_store is None:
        return {"total_memories": 0, "memories_last_hour": 0, "domain": domain}
    return memory_store.stats(domain)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
