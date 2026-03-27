"""
connector.py
------------
Neo4j Graph Database Connector.

Manages the Neo4j driver connection and executes Cypher queries.
Fixed: Removed dangling session (sessions are now short-lived per-query as intended).
"""
from neo4j import GraphDatabase
from neo4j.exceptions import ClientError

from .config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE


class GraphDBConnector:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )

    def check_connection(self):
        try:
            self.driver.verify_connectivity()
            print("Connection successful")
        except ClientError as error:
            print(f"Connection Failed: {error}")

    def close(self):
        self.driver.close()

    def execute_query(self, query, params=None):
        """Execute a Cypher query using a short-lived session (best practice)."""
        with self.driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(query, params)
            return result.data()
