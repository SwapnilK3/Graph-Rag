import asyncio
import os
import logging
from typing import Optional, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .pipeline import GraphRAGPipeline
from .connector import GraphDBConnector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Graph-RAG API", version="2.0.0")

# Enable CORS for frontend microservice
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache for pipelines to avoid re-discovering schema every time
# In a production environment, you might use a more robust caching/pooling mechanism
pipelines = {}

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

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "2.0.0"}

@app.post("/query", response_model=QueryResponse)
async def query_graph(request: QueryRequest):
    """
    Process a natural language query using the Graph-RAG pipeline.
    The domain parameter determines which configuration is used.
    """
    try:
        domain = request.domain
        
        # Initialize or retrieve cached pipeline for the domain
        if domain not in pipelines:
            logger.info(f"Initializing new pipeline for domain: {domain}")
            pipelines[domain] = GraphRAGPipeline(domain=domain)
        
        pipeline = pipelines[domain]
        # Fix 13: Run sync pipeline in executor to avoid blocking event loop
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
        logger.error(f"Domain config not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Error processing query")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
