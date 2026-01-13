from neo4j import GraphDatabase
from neo4j.exceptions import ClientError

from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE


class GraphDBConnector:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI, 
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        self.session= self.driver.session(database=NEO4J_DATABASE)

    def check_connection(self):
        try:
            self.driver.verify_connectivity()
        except ClientError as error:
            print(f"Connection Failed")
            
    def close(self):
        if self.session:
            self.session.close()
        self.driver.close()
            
    def execute_query(self, query, params=None):
        with self.driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(query, params)
            return result.data()
