"""
src/utils/connector.py
----------------------
Shared Neo4j connector for the methodology pipeline.
Wraps neo4j driver with helper methods for Cypher execution.

Based on the existing GraphDBConnector but domain-independent.
"""

import os
from neo4j import GraphDatabase
from neo4j.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()


class Neo4jConnector:
    """
    Manages a Neo4j driver and provides query execution helpers.

    Can be initialised explicitly or from environment variables
    (NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE).
    """

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
    ):
        self.uri = uri or os.getenv("NEO4J_URI")
        self.user = user or os.getenv("NEO4J_USERNAME")
        self.password = password or os.getenv("NEO4J_PASSWORD")
        self.database = database or os.getenv("NEO4J_DATABASE")

        self.driver = GraphDatabase.driver(
            self.uri, auth=(self.user, self.password)
        )

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------
    def verify(self) -> bool:
        """Return True if the database is reachable."""
        try:
            self.driver.verify_connectivity()
            return True
        except Exception:
            return False

    def close(self):
        self.driver.close()

    # ------------------------------------------------------------------
    # Query execution
    # ------------------------------------------------------------------
    def run(self, cypher: str, params: dict | None = None) -> list[dict]:
        """Execute a Cypher query and return a list of record dicts."""
        with self.driver.session(database=self.database) as session:
            result = session.run(cypher, params or {})
            return result.data()

    def run_single(self, cypher: str, params: dict | None = None) -> dict | None:
        """Execute a Cypher query and return the first record or None."""
        rows = self.run(cypher, params)
        return rows[0] if rows else None

    def write(self, cypher: str, params: dict | None = None) -> list[dict]:
        """Execute a write transaction."""
        with self.driver.session(database=self.database) as session:
            result = session.run(cypher, params or {})
            return result.data()

    # ------------------------------------------------------------------
    # Bulk helpers
    # ------------------------------------------------------------------
    def run_batch(self, cypher: str, batch: list[dict]) -> None:
        """Run the same Cypher statement for each param-dict in *batch*."""
        with self.driver.session(database=self.database) as session:
            for params in batch:
                session.run(cypher, params)

    def clear_database(self) -> None:
        """Delete every node and relationship in the database."""
        self.run("MATCH (n) DETACH DELETE n")

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
