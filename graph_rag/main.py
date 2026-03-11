"""
main.py
-------
Entry point for the fully automatic Graph-RAG system.

Demonstrates the complete pipeline:
  1. Connect to Neo4j (any graph)
  2. Auto-discover schema
  3. Auto-generate config
  4. Answer queries using graph traversal + Gemini LLM

Usage:
    python main.py                         # interactive mode
    python main.py "your question here"    # single query mode
    python main.py --seed                  # seed sample data then query
"""

import sys
from pipeline import GraphRAGPipeline
from connector import GraphDBConnector


# ---------------------------------------------------------------------------
# Sample medical graph seeding (for demo / first run)
# ---------------------------------------------------------------------------

SAMPLE_NODES = [
    ("Aspirin",            "Drug",       {"type": "NSAID"}),
    ("Ibuprofen",          "Drug",       {"type": "NSAID"}),
    ("Warfarin",           "Drug",       {"type": "Anticoagulant"}),
    ("Metformin",          "Drug",       {"type": "Biguanide"}),
    ("Lisinopril",         "Drug",       {"type": "ACE Inhibitor"}),
    ("Omeprazole",         "Drug",       {"type": "PPI"}),
    ("Atorvastatin",       "Drug",       {"type": "Statin"}),
    ("Amoxicillin",        "Drug",       {"type": "Antibiotic"}),

    ("Headache",           "Disease",    {"severity": "Mild"}),
    ("Arthritis",          "Disease",    {"severity": "Moderate"}),
    ("Diabetes",           "Disease",    {"severity": "Chronic"}),
    ("Hypertension",       "Disease",    {"severity": "Chronic"}),
    ("Heart Disease",      "Disease",    {"severity": "Severe"}),
    ("Peptic Ulcer",       "Disease",    {"severity": "Moderate"}),
    ("High Cholesterol",   "Disease",    {"severity": "Chronic"}),
    ("Bacterial Infection","Disease",    {"severity": "Moderate"}),

    ("Nausea",             "SideEffect", {"severity": "Mild"}),
    ("Stomach Bleeding",   "SideEffect", {"severity": "Severe"}),
    ("Kidney Damage",      "SideEffect", {"severity": "Severe"}),
    ("Dry Cough",          "SideEffect", {"severity": "Mild"}),
    ("Dizziness",          "SideEffect", {"severity": "Mild"}),
    ("Fatigue",            "SideEffect", {"severity": "Mild"}),
    ("Muscle Pain",        "SideEffect", {"severity": "Moderate"}),
    ("Diarrhea",           "SideEffect", {"severity": "Mild"}),
    ("Liver Damage",       "SideEffect", {"severity": "Severe"}),

    ("Severe Pain",        "Symptom",    {}),
    ("Joint Pain",         "Symptom",    {}),
    ("Frequent Urination", "Symptom",    {}),
    ("High Blood Pressure","Symptom",    {}),
    ("Chest Pain",         "Symptom",    {}),
    ("Burning Sensation",  "Symptom",    {}),
    ("Fever",              "Symptom",    {}),
]

SAMPLE_RELATIONSHIPS = [
    ("Aspirin",      "TREATS",           "Headache"),
    ("Aspirin",      "TREATS",           "Heart Disease"),
    ("Aspirin",      "CAUSES",           "Nausea"),
    ("Aspirin",      "CAUSES",           "Stomach Bleeding"),
    ("Aspirin",      "INTERACTS_WITH",   "Warfarin"),
    ("Ibuprofen",    "TREATS",           "Headache"),
    ("Ibuprofen",    "TREATS",           "Arthritis"),
    ("Ibuprofen",    "CAUSES",           "Nausea"),
    ("Ibuprofen",    "CAUSES",           "Kidney Damage"),
    ("Ibuprofen",    "INTERACTS_WITH",   "Warfarin"),
    ("Warfarin",     "TREATS",           "Heart Disease"),
    ("Warfarin",     "CAUSES",           "Stomach Bleeding"),
    ("Metformin",    "TREATS",           "Diabetes"),
    ("Metformin",    "CAUSES",           "Nausea"),
    ("Metformin",    "CAUSES",           "Fatigue"),
    ("Lisinopril",   "TREATS",           "Hypertension"),
    ("Lisinopril",   "CAUSES",           "Dry Cough"),
    ("Lisinopril",   "CAUSES",           "Dizziness"),
    ("Omeprazole",   "TREATS",           "Peptic Ulcer"),
    ("Omeprazole",   "CAUSES",           "Diarrhea"),
    ("Atorvastatin", "TREATS",           "High Cholesterol"),
    ("Atorvastatin", "CAUSES",           "Muscle Pain"),
    ("Atorvastatin", "CAUSES",           "Liver Damage"),
    ("Amoxicillin",  "TREATS",           "Bacterial Infection"),
    ("Amoxicillin",  "CAUSES",           "Diarrhea"),
    ("Amoxicillin",  "INTERACTS_WITH",   "Warfarin"),
    ("Headache",          "HAS_SYMPTOM", "Severe Pain"),
    ("Arthritis",         "HAS_SYMPTOM", "Joint Pain"),
    ("Diabetes",          "HAS_SYMPTOM", "Frequent Urination"),
    ("Hypertension",      "HAS_SYMPTOM", "High Blood Pressure"),
    ("Heart Disease",     "HAS_SYMPTOM", "Chest Pain"),
    ("Peptic Ulcer",      "HAS_SYMPTOM", "Burning Sensation"),
    ("Peptic Ulcer",      "HAS_SYMPTOM", "Stomach Bleeding"),
    ("Bacterial Infection","HAS_SYMPTOM","Fever"),
    ("Stomach Bleeding",  "INCREASES_RISK_OF", "Peptic Ulcer"),
    ("Kidney Damage",     "INCREASES_RISK_OF", "Hypertension"),
    ("Muscle Pain",       "INCREASES_RISK_OF", "Heart Disease"),
]


def seed_sample_graph(connector: GraphDBConnector) -> None:
    """Seed the sample medical knowledge graph."""
    print("\n  Clearing existing data …")
    connector.execute_query("MATCH (n) DETACH DELETE n")

    print(f"  Creating {len(SAMPLE_NODES)} nodes …")
    for name, label, props in SAMPLE_NODES:
        connector.execute_query(
            f"MERGE (n:{label} {{name: $name}}) SET n += $props",
            {"name": name, "props": props},
        )

    print(f"  Creating {len(SAMPLE_RELATIONSHIPS)} relationships …")
    for src, rel, tgt in SAMPLE_RELATIONSHIPS:
        connector.execute_query(
            f"MATCH (a {{name: $src}}), (b {{name: $tgt}}) MERGE (a)-[:{rel}]->(b)",
            {"src": src, "tgt": tgt},
        )
    print("  Done.\n")


# ---------------------------------------------------------------------------
# Demo queries
# ---------------------------------------------------------------------------

DEMO_QUERIES = [
    "What are the side effects of aspirin?",
    "What treats headaches?",
    "Can I take aspirin with warfarin?",
    "What are the symptoms of diabetes?",
    "Give me an overview of metformin",
    "How is aspirin connected to peptic ulcer?",
    "What do aspirin and ibuprofen have in common?",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    seed = "--seed" in sys.argv
    single_query = None

    # Check for a query argument (skip flags)
    for arg in sys.argv[1:]:
        if not arg.startswith("--"):
            single_query = arg
            break

    print("=" * 60)
    print("  GRAPH-RAG — FULLY AUTOMATIC PIPELINE")
    print("=" * 60)

    connector = GraphDBConnector()

    if seed:
        seed_sample_graph(connector)

    # Build the pipeline — everything is automatic from here
    pipeline = GraphRAGPipeline(connector=connector, verbose=True)

    if single_query:
        # Single query mode
        pipeline.query_interactive(single_query)
    else:
        # Interactive / demo mode
        print("\n" + "=" * 60)
        print("  RUNNING DEMO QUERIES")
        print("=" * 60)

        for q in DEMO_QUERIES:
            pipeline.query_interactive(q)

        # Interactive loop
        print("\n  Type a question (or 'quit' to exit):\n")
        while True:
            try:
                user_input = input("  > ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input or user_input.lower() in ("quit", "exit", "q"):
                break

            pipeline.query_interactive(user_input)

    pipeline.close()
    print("\n  Goodbye.\n")


if __name__ == "__main__":
    main()
