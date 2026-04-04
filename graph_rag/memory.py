"""
memory.py
---------
V3 Component: Agent Memory Layer

Three-tier memory system for cross-session learning:
1. Short-term: Current conversation context (in-memory dict)
2. Episodic:   Past query→answer pairs with subgraph fingerprints (SQLite)
3. Semantic:   Query embeddings for similarity recall (stored in SQLite as blobs)

Storage
-------
SQLite database with table:
    memories(
        id INTEGER PRIMARY KEY,
        query_text TEXT,
        query_embedding BLOB,
        answer TEXT,
        context TEXT,
        intent TEXT,
        strategy TEXT,
        quality_score REAL,
        subgraph_json TEXT,
        domain TEXT,
        timestamp REAL
    )
"""

from __future__ import annotations

import json
import time
import struct
import sqlite3
import logging
import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MemoryResult:
    """A recalled memory from the episodic store."""
    query: str
    answer: str
    context: str
    intent: str
    similarity: float
    age_seconds: float


def _embed_to_bytes(embedding: list[float]) -> bytes:
    """Pack a float list into a compact bytes blob."""
    return struct.pack(f"{len(embedding)}f", *embedding)


def _bytes_to_embed(blob: bytes) -> list[float]:
    """Unpack a bytes blob back into a float list."""
    count = len(blob) // 4  # 4 bytes per float32
    return list(struct.unpack(f"{count}f", blob))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x * x for x in a) ** 0.5
    mag_b = sum(x * x for x in b) ** 0.5
    if not mag_a or not mag_b:
        return 0.0
    return dot / (mag_a * mag_b)


class AgentMemory:
    """
    Multi-layer memory for the Graph-RAG agent.

    Parameters
    ----------
    llm : LLMInterface
        For generating query embeddings.
    db_path : str or Path
        Path to the SQLite database file.
    """

    def __init__(self, llm, db_path: str | Path | None = None):
        self.llm = llm
        if db_path is None:
            db_path = os.getenv("GRAPH_RAG_MEMORY_DB_PATH", "graph_rag_memory.db")
        self.db_path = str(db_path)
        self.short_term: dict[str, dict] = {}  # session-scoped
        self._init_db()

    def _init_db(self):
        """Create the memories table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_text TEXT NOT NULL,
                    query_embedding BLOB,
                    answer TEXT,
                    context TEXT,
                    intent TEXT,
                    strategy TEXT,
                    quality_score REAL DEFAULT 0,
                    subgraph_json TEXT,
                    domain TEXT DEFAULT 'default',
                    timestamp REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_domain 
                ON memories(domain)
            """)
            conn.commit()
        logger.info("Memory database initialized at %s", self.db_path)

    def store(
        self,
        query: str,
        answer: str,
        context: str,
        intent: str = "",
        strategy: str = "",
        quality_score: float = 0.0,
        subgraph: dict | None = None,
        domain: str = "default",
    ):
        """
        Store a query result in episodic memory.

        Also stores the query embedding for future similarity recall.
        """
        # Generate embedding
        try:
            embedding = self.llm.embed_text(query)
            embedding_blob = _embed_to_bytes(embedding)
        except Exception as e:
            logger.warning("Failed to embed query for memory: %s", e)
            embedding_blob = None

        # Serialize subgraph
        subgraph_json = json.dumps(subgraph) if subgraph else None

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO memories 
                   (query_text, query_embedding, answer, context, intent, 
                    strategy, quality_score, subgraph_json, domain, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (query, embedding_blob, answer, context, intent,
                 strategy, quality_score, subgraph_json, domain, time.time()),
            )
            conn.commit()

        # Also store in short-term
        self.short_term[query] = {
            "answer": answer, "context": context, "intent": intent,
        }

        logger.debug("Stored memory for query: '%s'", query[:50])

    def recall(
        self,
        query: str,
        domain: str = "default",
        threshold: float = 0.85,
        max_results: int = 3,
    ) -> list[MemoryResult]:
        """
        Find similar past queries in episodic memory.

        Parameters
        ----------
        query : str
            The query to search for.
        domain : str
            Restrict recall to this domain.
        threshold : float
            Minimum cosine similarity to return.
        max_results : int
            Maximum number of results.

        Returns
        -------
        list[MemoryResult]
            Similar past queries sorted by similarity (descending).
        """
        # Check short-term first
        if query in self.short_term:
            st = self.short_term[query]
            return [MemoryResult(
                query=query, answer=st["answer"], context=st["context"],
                intent=st["intent"], similarity=1.0, age_seconds=0.0,
            )]

        # Generate query embedding
        try:
            query_vec = self.llm.embed_text(query)
        except Exception as e:
            logger.warning("Failed to embed query for recall: %s", e)
            return []

        # Load all embeddings from the domain
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """SELECT query_text, query_embedding, answer, context, 
                          intent, timestamp
                   FROM memories 
                   WHERE domain = ? AND query_embedding IS NOT NULL
                   ORDER BY timestamp DESC
                   LIMIT 500""",
                (domain,),
            ).fetchall()

        # Score by cosine similarity
        scored = []
        for row in rows:
            q_text, emb_blob, answer, context, intent, ts = row
            stored_vec = _bytes_to_embed(emb_blob)
            sim = _cosine_similarity(query_vec, stored_vec)
            if sim >= threshold:
                scored.append(MemoryResult(
                    query=q_text,
                    answer=answer or "",
                    context=context or "",
                    intent=intent or "",
                    similarity=sim,
                    age_seconds=now - ts,
                ))

        # Sort by similarity descending
        scored.sort(key=lambda x: x.similarity, reverse=True)
        return scored[:max_results]

    def get_session_history(self) -> list[dict]:
        """Return short-term conversation history."""
        return [
            {"query": q, **data}
            for q, data in self.short_term.items()
        ]

    def clear_session(self):
        """Clear short-term memory (start new conversation)."""
        self.short_term.clear()

    def stats(self, domain: str = "default") -> dict:
        """Return memory statistics."""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE domain = ?", (domain,)
            ).fetchone()[0]
            recent = conn.execute(
                """SELECT COUNT(*) FROM memories 
                   WHERE domain = ? AND timestamp > ?""",
                (domain, time.time() - 3600),
            ).fetchone()[0]
        return {"total_memories": total, "memories_last_hour": recent, "domain": domain}
