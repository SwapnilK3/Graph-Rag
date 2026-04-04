"""
connector.py
------------
Neo4j Graph Database Connector.

Manages the Neo4j driver connection and executes Cypher queries.

Supports two modes:
  1. Default: reads credentials from .env (NEO4J_URI, etc.)
  2. Dynamic: accepts uri/user/password/database at construction time
     for multi-KG scenarios where each graph has its own credentials.
"""
from neo4j import GraphDatabase
from neo4j.exceptions import ClientError

from .config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE


class GraphDBConnector:
    """
    Neo4j driver wrapper with optional dynamic credentials.

    Parameters
    ----------
    uri : str, optional
        Bolt URI. Falls back to NEO4J_URI from .env.
    user : str, optional
        Username. Falls back to NEO4J_USERNAME from .env.
    password : str, optional
        Password. Falls back to NEO4J_PASSWORD from .env.
    database : str, optional
        Database name. Falls back to NEO4J_DATABASE from .env.
    """

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
    ):
        self._uri = uri or NEO4J_URI
        self._user = user or NEO4J_USER
        self._password = password or NEO4J_PASSWORD
        self._database = database or NEO4J_DATABASE

        self.driver = GraphDatabase.driver(
            self._uri,
            auth=(self._user, self._password),
        )

    def check_connection(self) -> tuple[bool, str]:
        """
        Verify connectivity to the Neo4j instance.

        Returns
        -------
        (bool, str)
            (True, "Connection successful") on success,
            (False, error_message) on failure.
        """
        try:
            self.driver.verify_connectivity()
            return True, "Connection successful"
        except Exception as error:
            return False, f"Connection failed: {error}"

    def close(self):
        self.driver.close()

    def execute_query(self, query, params=None):
        """Execute a Cypher query using a short-lived session (best practice)."""
        with self.driver.session(database=self._database) as session:
            result = session.run(query, params)
            return result.data()
