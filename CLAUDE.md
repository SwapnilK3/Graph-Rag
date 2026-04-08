# Graph-RAG Project Memory (CLAUDE.md)

## Current Project State (Version 3.2 - Stable Platform)
Graph-RAG has evolved from a single-graph script into a **Multi-Tenant Knowledge Graph Platform** with an agentic reasoning loop.

### Core Innovations (V3.2)
- [x] **Agentic Reasoning Loop**: Plan → Retrieve → Evaluate → Refine. Iterative multi-step reasoning with adaptive re-planning.
- [x] **Multi-KG Registry**: SQLite-backed registry for managing multiple Neo4j connections, credentials, and schemas.
- [x] **Zero-Config Discovery**: Fully automated graph bootstrapping. Discovers labels, relationships, and identifies "searchable properties" based on frequency.
- [x] **Isolated Memory Layer**: SQL-backed episodic and semantic memory scoped to specific `kg_id`, preventing cross-graph data leakage.
- [x] **V2.5 Adaptive Pipeline**: Domain Guard (early rejection), motif summarization for abstract queries, and path scoring/ranking.

### Architectural Stack
- **Backend (8080)**: FastAPI. Routes queries based on `kg_id`.
- **Frontend (3000)**: D3.js powered dashboard with multi-graph management and citation cards.
- **Persistence**: Neo4j (Graph), SQLite (Registry & Memory).
- **Intelligence**: Gemini 1.5/2.0 (Pro/Flash).

---

## Technical Components Reference

### 1. Platform Infrastructure
- **`graph_registry.py`**: The central "brain." Handles KG registration, driver pooling, and cached schema/config lookups.
- **`api.py`**: Multi-graph aware endpoints (`/graphs`, `/query`, `/agent/query`). Auto-registers the default graph from `.env` on startup.
- **`connector.py`**: Neo4j driver wrapper supported by dynamic credentials.

### 2. Reasoning Core
- **`agent.py`**: Orchestrates the multi-step loop. Uses the LLM to evaluate context quality and re-plan searches.
- **`pipeline.py`**: The V2.5 high-precision retrieval engine. Features Domain Guard and Adaptive Retry.
- **`entity_extractor.py`**: Hybrid extraction (Lexical n-grams → LLM extraction → Semantic search).
- **`traversal_engine.py`**: 5 strategies: Targeted, Chained, Shortest Path, Shared Neighbor, Variable-hop.
- **`path_scorer.py`**: Scores and ranks subgraphs by relevance to the query tokens.

### 3. Knowledge Layer
- **`schema_discovery.py`**: Deep graph inspection to identify the structural blueprint of unknown databases.
- **`auto_config.py`**: Heuristic-based generation of the reasoning configuration.
- **`memory.py`**: SQLite + Embeddings memory layer with `domain` (kg_id) scoping.

---

## Development Standards
- **Language**: Python 3.10+
- **Database**: Neo4j (free tier requires `neo4j+ssc://` for Aura).
- **Ports**: Backend (8080), Frontend (3000).
- **Style**: Relative imports within `graph_rag/`. Strict grounding (no external knowledge).

## Critical Commands
- **Run Platform (Docker)**: `docker compose up --build`
- **Register Metadata**: `python -c "from graph_rag.graph_registry import GraphRegistry; GraphRegistry().ensure_default()"`
- **Test Suite**: `pytest tests/test_hopping.py tests/test_complex_pipeline.py`

## Documentation Index
- `RESEARCH_TECHNICAL_REPORT.md`: Methodology and novelty for research paper drafting.
- `Architecture.md`: Detailed system diagrams and agentic loop flows.
- `README.md`: High-level platform overview.
