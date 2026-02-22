"""
test_live_extractor.py
----------------------
Live integration test for the Graph-RAG entity extractor.

Steps:
  1. Connect to the Aura instance from .env
  2. Seed a small knowledge graph (clears then re-creates sample nodes)
  3. Run QueryTimeEntityExtractor against a plain-English prompt
  4. Print found entry-nodes — these are the Graph-RAG traversal starting points

Run:
    python test_live_extractor.py
"""

from connector import GraphDBConnector
from entity_extractor import QueryTimeEntityExtractor

# ---------------------------------------------------------------------------
# Sample knowledge graph data
# ---------------------------------------------------------------------------
SEED_QUERY = """
// Clear existing sample data first
MATCH (n) DETACH DELETE n
"""

SAMPLE_NODES = [
    # Technology concepts
    ("Python",          "Technology", {"description": "A high-level programming language"}),
    ("Machine Learning","Technology", {"description": "Algorithms that learn from data"}),
    ("Neural Network",  "Technology", {"description": "Computing system inspired by biological brains"}),
    ("Graph Database",  "Technology", {"description": "Database that uses graph structures"}),
    ("Neo4j",           "Technology", {"description": "Popular graph database platform"}),
    ("Knowledge Graph", "Technology", {"description": "Structured knowledge representation as a graph"}),
    ("Natural Language Processing", "Technology", {"description": "AI field for understanding human language"}),
    ("Vector Embedding","Technology", {"description": "Numerical representation of text or data"}),
    ("Transformer",     "Technology", {"description": "Deep learning model architecture for NLP"}),
    # Relationships / concepts
    ("Retrieval Augmented Generation", "Concept", {"description": "Combines retrieval systems with generative AI"}),
    ("Graph RAG",       "Concept",    {"description": "RAG that uses a knowledge graph as retrieval source"}),
    ("Entity Extraction","Concept",   {"description": "Identifying named entities in text"}),
    ("Semantic Search", "Concept",    {"description": "Search based on meaning rather than keywords"}),
    # Persons
    ("Alan Turing",     "Person",     {"description": "Pioneer of computer science and AI"}),
    ("Geoffrey Hinton", "Person",     {"description": "Deep learning pioneer, Nobel laureate 2024"}),
]

SAMPLE_RELS = [
    ("Graph RAG",        "USES",       "Knowledge Graph"),
    ("Graph RAG",        "USES",       "Retrieval Augmented Generation"),
    ("Graph RAG",        "USES",       "Entity Extraction"),
    ("Knowledge Graph",  "STORED_IN",  "Graph Database"),
    ("Neo4j",            "IS_A",       "Graph Database"),
    ("Neural Network",   "IS_A",       "Machine Learning"),
    ("Transformer",      "IS_A",       "Neural Network"),
    ("Natural Language Processing", "USES", "Transformer"),
    ("Graph RAG",        "USES",       "Natural Language Processing"),
    ("Entity Extraction","USES",       "Natural Language Processing"),
    ("Semantic Search",  "USES",       "Vector Embedding"),
    ("Geoffrey Hinton",  "PIONEERED",  "Neural Network"),
    ("Alan Turing",      "FOUNDED",    "Machine Learning"),
]


def seed_graph(connector: GraphDBConnector) -> None:
    print("\n── Seeding knowledge graph ──────────────────────────────────────────")
    connector.execute_query("MATCH (n) DETACH DELETE n")
    print(f"  Cleared existing data")

    for name, label, props in SAMPLE_NODES:
        connector.execute_query(
            f"MERGE (n:{label} {{name: $name}}) SET n += $props",
            {"name": name, "props": props},
        )
    print(f"  Created {len(SAMPLE_NODES)} nodes")

    for src, rel, tgt in SAMPLE_RELS:
        connector.execute_query(
            f"""
            MATCH (a {{name: $src}}), (b {{name: $tgt}})
            MERGE (a)-[:{rel}]->(b)
            """,
            {"src": src, "tgt": tgt},
        )
    print(f"  Created {len(SAMPLE_RELS)} relationships")


def run_extraction(connector: GraphDBConnector, prompt: str) -> None:
    print(f"\n── Entity extraction ────────────────────────────────────────────────")
    print(f"  Prompt : \"{prompt}\"")

    extractor = QueryTimeEntityExtractor(
        connector,
        search_properties=["name"],
    )

    entry_nodes = extractor.extract_entry_nodes(prompt)

    if not entry_nodes:
        print("  No entry nodes found — try a different prompt or check graph data.")
        return

    print(f"\n  ✓ Found {len(entry_nodes)} entry node(s):\n")
    print(f"  {'ID':<40} {'Label':<30} {'Name'}")
    print(f"  {'-'*40} {'-'*30} {'-'*30}")
    for node in entry_nodes:
        print(f"  {node['id']:<40} {node['label']:<30} {node['name']}")

    print("\n  ── Suggested first traversal step (neighbours) ───────────────────")
    for node in entry_nodes[:3]:          # show neighbours for top 3 only
        rows = connector.execute_query(
            """
            MATCH (n)-[r]-(m)
            WHERE elementId(n) = $id
            RETURN type(r) AS rel, m.name AS neighbour, labels(m)[0] AS nlabel
            LIMIT 5
            """,
            {"id": node["id"]},
        )
        if rows:
            print(f"\n  [{node['label']}] {node['name']}")
            for r in rows:
                print(f"    ─{r['rel']}─▶ [{r['nlabel']}] {r['neighbour']}")


if __name__ == "__main__":
    print("Connecting to Neo4j Aura …")
    connector = GraphDBConnector()
    connector.check_connection()

    # 1. Seed sample data
    seed_graph(connector)

    # 2. Test prompts — change these to explore different entry points
    prompts = [
        "How does Graph RAG use knowledge graphs and entity extraction?",
        "Explain machine learning and neural networks",
        "What did Geoffrey Hinton contribute to deep learning?",
    ]

    for prompt in prompts:
        run_extraction(connector, prompt)
        print()

    connector.close()
    print("Done.")
