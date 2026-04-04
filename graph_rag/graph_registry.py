"""
graph_registry.py
-----------------
Graph Registry: Manages multiple knowledge graph connections.

This is the central manager for the plug-and-play KG platform.
It stores credentials, cached schemas, and auto-generated configs
in a SQLite database so graphs persist across restarts.

Each KG is identified by a unique `kg_id` and gets its own:
  - Neo4j connector (driver)
  - Discovered schema
  - Auto-generated config
  - Pipeline instance
  - Memory namespace

Usage
-----
    registry = GraphRegistry()
    kg_id = registry.register("bolt://localhost:7687", "neo4j", "password")
    pipeline = registry.get_pipeline(kg_id)
    result = pipeline.query("What treats headaches?")
"""

from __future__ import annotations

import json
import time
import uuid
import sqlite3
import logging
import os
from typing import Optional
from pathlib import Path

from .connector import GraphDBConnector
from .schema_discovery import SchemaDiscovery
from .auto_config import AutoConfigGenerator

logger = logging.getLogger(__name__)

# Database file location
_DEFAULT_DB_PATH = Path(
    os.getenv(
        "GRAPH_REGISTRY_DB_PATH",
        str(Path(__file__).parent.parent / "graph_registry.db"),
    )
)


class GraphRegistry:
    """
    SQLite-backed registry for managing multiple knowledge graphs.

    Parameters
    ----------
    db_path : str or Path
        Path to the SQLite database file.
    """

    def __init__(self, db_path: str | Path = _DEFAULT_DB_PATH):
        self.db_path = str(db_path)
        self._connectors: dict[str, GraphDBConnector] = {}
        self._pipelines: dict[str, object] = {}  # lazy import to avoid circular
        self._agents: dict[str, object] = {}
        self._init_db()

    # ------------------------------------------------------------------
    # Database Setup
    # ------------------------------------------------------------------

    def _init_db(self):
        """Create the knowledge_graphs table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_graphs (
                    kg_id       TEXT PRIMARY KEY,
                    name        TEXT NOT NULL,
                    uri         TEXT NOT NULL,
                    username    TEXT NOT NULL,
                    password    TEXT NOT NULL,
                    database    TEXT DEFAULT 'neo4j',
                    schema_json TEXT,
                    config_json TEXT,
                    status      TEXT DEFAULT 'disconnected',
                    created_at  REAL NOT NULL,
                    last_used   REAL NOT NULL
                )
            """)
            conn.commit()
        logger.info("Graph registry initialized at %s", self.db_path)

    # ------------------------------------------------------------------
    # Public API: CRUD
    # ------------------------------------------------------------------

    def register(
        self,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
        name: str = "",
    ) -> dict:
        """
        Register a new knowledge graph.

        1. Connects to the Neo4j instance
        2. Verifies connectivity
        3. Discovers schema automatically
        4. Generates config automatically
        5. Stores everything in the registry DB

        Returns
        -------
        dict
            {"kg_id": str, "name": str, "status": str, "schema_summary": dict}

        Raises
        ------
        ConnectionError
            If the Neo4j instance is unreachable.
        """
        # Generate unique ID
        kg_id = f"kg_{uuid.uuid4().hex[:12]}"

        # Auto-generate name if not provided
        if not name:
            # Extract host from URI for display
            host = uri.split("//")[-1].split(":")[0].split("/")[0]
            name = f"{database}@{host}"

        # Step 1: Connect
        connector = GraphDBConnector(
            uri=uri, user=username, password=password, database=database
        )
        ok, msg = connector.check_connection()
        if not ok:
            connector.close()
            lowered = (msg or "").lower()
            if (
                "neo.clienterror.security.unauthorized" in lowered
                or "authentication failure" in lowered
                or "permission/access denied" in lowered
            ):
                raise ConnectionError(
                    "Authentication failed for this connection request. "
                    "Double-check Neo4j username/password, avoid hidden spaces, "
                    "and if you recently changed the Aura password, restart the app "
                    "so any cached connectors are refreshed."
                )
            raise ConnectionError(f"Cannot connect to {uri}: {msg}")

        # Step 2: Discover schema
        try:
            logger.info("Discovering schema for %s ...", name)
            discovery = SchemaDiscovery(connector)
            schema = discovery.discover()

            # Step 3: Generate config
            generator = AutoConfigGenerator(schema)
            config = generator.generate()
        except Exception as e:
            connector.close()
            error_text = str(e)
            if (
                "DatabaseNotFound" in error_text
                or "graph reference with the name" in error_text
            ):
                raise ConnectionError(
                    f"Database '{database}' was not found. "
                    "Use the exact Neo4j database name "
                    "(for Aura this is usually the DB id, not the instance name)."
                ) from e
            raise ConnectionError(
                f"Schema discovery failed for database '{database}': {error_text}"
            ) from e

        # Step 4: Persist to DB
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO knowledge_graphs
                   (kg_id, name, uri, username, password, database,
                    schema_json, config_json, status, created_at, last_used)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    kg_id, name, uri, username, password, database,
                    json.dumps(schema), json.dumps(config),
                    "connected", now, now,
                ),
            )
            conn.commit()

        # Cache connector
        self._connectors[kg_id] = connector

        logger.info("Registered KG '%s' as %s (%d nodes, %d rels)",
                     name, kg_id,
                     schema.get("total_nodes", 0),
                     schema.get("total_relationships", 0))

        intent_patterns = config.get("intent_patterns", {}) if isinstance(config, dict) else {}
        strategy_counts: dict[str, int] = {}
        for pattern in intent_patterns.values():
            strategy = pattern.get("strategy", "unknown") if isinstance(pattern, dict) else "unknown"
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        return {
            "kg_id": kg_id,
            "name": name,
            "status": "connected",
            "schema_summary": {
                "node_labels": schema.get("node_labels", []),
                "total_nodes": schema.get("total_nodes", 0),
                "total_relationships": schema.get("total_relationships", 0),
                "relationships": [
                    {"type": r["type"],
                     "source": r["source_label"],
                     "target": r["target_label"]}
                    for r in schema.get("relationships", [])
                ],
            },
            "config_summary": {
                "domain": config.get("domain", "auto-generated") if isinstance(config, dict) else "auto-generated",
                "intent_count": len(intent_patterns),
                "strategy_counts": strategy_counts,
            },
        }

    def get(self, kg_id: str) -> Optional[dict]:
        """Get a single KG's metadata by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM knowledge_graphs WHERE kg_id = ?", (kg_id,)
            ).fetchone()
        if row is None:
            return None
        return dict(row)

    def list_all(self) -> list[dict]:
        """List all registered KGs with summary info (no passwords)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT kg_id, name, uri, database, status,
                          created_at, last_used, schema_json
                   FROM knowledge_graphs
                   ORDER BY last_used DESC"""
            ).fetchall()

        result = []
        for row in rows:
            r = dict(row)
            # Parse schema for summary
            schema = json.loads(r.pop("schema_json", "{}") or "{}")
            r["total_nodes"] = schema.get("total_nodes", 0)
            r["total_relationships"] = schema.get("total_relationships", 0)
            r["node_labels"] = schema.get("node_labels", [])
            result.append(r)
        return result

    def delete(self, kg_id: str) -> bool:
        """
        Remove a KG from the registry and clean up resources.

        Returns True if found and deleted, False if not found.
        """
        # Close connector if cached
        if kg_id in self._connectors:
            try:
                self._connectors[kg_id].close()
            except Exception:
                pass
            del self._connectors[kg_id]

        # Remove cached pipeline and agent
        self._pipelines.pop(kg_id, None)
        self._agents.pop(kg_id, None)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM knowledge_graphs WHERE kg_id = ?", (kg_id,)
            )
            conn.commit()
            deleted = cursor.rowcount > 0

        if deleted:
            logger.info("Deleted KG %s", kg_id)
        return deleted

    # ------------------------------------------------------------------
    # Public API: Runtime Access
    # ------------------------------------------------------------------

    def get_connector(self, kg_id: str) -> GraphDBConnector:
        """
        Get a cached connector for a KG, creating one if needed.
        """
        if kg_id not in self._connectors:
            kg = self.get(kg_id)
            if kg is None:
                raise KeyError(f"Unknown KG: {kg_id}")
            self._connectors[kg_id] = GraphDBConnector(
                uri=kg["uri"],
                user=kg["username"],
                password=kg["password"],
                database=kg["database"],
            )
        # Update last_used
        self._touch(kg_id)
        return self._connectors[kg_id]

    def get_pipeline(self, kg_id: str):
        """
        Get a cached pipeline for a KG, creating one if needed.

        Uses the stored schema and config to avoid re-discovery.
        """
        if kg_id not in self._pipelines:
            kg = self.get(kg_id)
            if kg is None:
                raise KeyError(f"Unknown KG: {kg_id}")

            connector = self.get_connector(kg_id)
            schema = json.loads(kg.get("schema_json", "{}") or "{}")
            config = json.loads(kg.get("config_json", "{}") or "{}")

            # Lazy import to avoid circular dependency
            from .pipeline import GraphRAGPipeline
            pipeline = GraphRAGPipeline(
                connector=connector,
                config=config if config else None,
                verbose=False,
            )
            # Override schema from cache (avoids re-discovery)
            if schema:
                pipeline.schema = schema

            self._pipelines[kg_id] = pipeline
            logger.info("Pipeline created for KG %s", kg_id)

        self._touch(kg_id)
        return self._pipelines[kg_id]

    def get_agent(self, kg_id: str):
        """
        Get a cached agent for a KG, creating one if needed.

        Memory is scoped to kg_id via the domain parameter.
        """
        if kg_id not in self._agents:
            pipeline = self.get_pipeline(kg_id)
            if not pipeline.llm:
                raise RuntimeError("LLM not configured — agent requires LLM")

            # Lazy imports
            from .agent import GraphRAGAgent
            from .memory import AgentMemory

            memory = AgentMemory(llm=pipeline.llm)
            self._agents[kg_id] = GraphRAGAgent(
                pipeline=pipeline,
                memory=memory,
                llm=pipeline.llm,
            )
            logger.info("Agent created for KG %s", kg_id)

        self._touch(kg_id)
        return self._agents[kg_id]

    # ------------------------------------------------------------------
    # Default KG (from .env)
    # ------------------------------------------------------------------

    def ensure_default(self) -> str:
        """
        Ensure the default KG from .env is registered.

        If a KG with kg_id='default' doesn't exist, registers
        one using the NEO4J_* environment variables.

        Returns the kg_id ('default').
        """
        from .config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE

        if not NEO4J_URI:
            logger.warning("No NEO4J_URI configured — skipping default KG")
            return "default"

        try:
            connector = GraphDBConnector()
            ok, msg = connector.check_connection()
            if not ok:
                logger.warning("Default Neo4j not reachable: %s", msg)
                return "default"

            discovery = SchemaDiscovery(connector)
            schema = discovery.discover()
            generator = AutoConfigGenerator(schema)
            config = generator.generate()

            now = time.time()
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO knowledge_graphs
                       (kg_id, name, uri, username, password, database,
                        schema_json, config_json, status, created_at, last_used)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        "default", "Medical Graph (Default)",
                        NEO4J_URI or "", NEO4J_USER or "", NEO4J_PASSWORD or "",
                        NEO4J_DATABASE or "neo4j",
                        json.dumps(schema), json.dumps(config),
                        "connected", now, now,
                    ),
                )
                conn.commit()

            # Invalidate stale runtime caches and swap to the freshly verified connector.
            old_connector = self._connectors.pop("default", None)
            if old_connector and old_connector is not connector:
                try:
                    old_connector.close()
                except Exception:
                    pass
            self._pipelines.pop("default", None)
            self._agents.pop("default", None)
            self._connectors["default"] = connector
            logger.info("Default KG registered from .env (%d nodes)",
                         schema.get("total_nodes", 0))

        except Exception as e:
            logger.warning("Failed to register default KG: %s", e)

        return "default"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _touch(self, kg_id: str):
        """Update last_used timestamp."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "UPDATE knowledge_graphs SET last_used = ? WHERE kg_id = ?",
                    (time.time(), kg_id),
                )
                conn.commit()
        except Exception:
            pass
