from connector import GraphDBConnector

if __name__ == "__main__":
    connector = GraphDBConnector()
    connector.check_connection()
    result = connector.execute_query("MATCH (n:Movie) RETURN n.title LIMIT 25")
    print(result)
    connector.close() 