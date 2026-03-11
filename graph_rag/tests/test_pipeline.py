"""
test_pipeline.py
----------------
Live end-to-end integration test of the full Graph-RAG pipeline.

Steps
-----
1. Connect to Neo4j Aura (credentials from .env)
2. Seed a medical knowledge graph
3. Run each pipeline step for several test queries:
     Entity Extraction → Intent Classification → Traversal → Context Generation
4. If GEMINI_API_KEY is present in .env, run the LLM step too.

Run:
    python test_pipeline.py
"""

import os
import sys

from connector import GraphDBConnector
from entity_extractor import QueryTimeEntityExtractor
from intent_classifier import IntentClassifier
from traversal_engine import SmartTraversalEngine
from context_generator import ContextGenerator

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config", "medical_graph.json")

# ---------------------------------------------------------------------------
# Medical knowledge graph seed data
# ---------------------------------------------------------------------------

NODES = [
    # (name, label, extra_properties)
    ("Aspirin",            "Drug",       {"type": "NSAID"}),
    ("Ibuprofen",          "Drug",       {"type": "NSAID"}),
    ("Warfarin",           "Drug",       {"type": "Anticoagulant"}),
    ("Metformin",          "Drug",       {"type": "Biguanide"}),
    ("Lisinopril",         "Drug",       {"type": "ACE Inhibitor"}),
    ("Headache",           "Disease",    {"severity": "Mild"}),
    ("Arthritis",          "Disease",    {"severity": "Moderate"}),
    ("Diabetes",           "Disease",    {"severity": "Chronic"}),
    ("Hypertension",       "Disease",    {"severity": "Chronic"}),
    ("Heart Disease",      "Disease",    {"severity": "Severe"}),
    ("Nausea",             "SideEffect", {"severity": "Mild"}),
    ("Stomach Bleeding",   "SideEffect", {"severity": "Severe"}),
    ("Kidney Damage",      "SideEffect", {"severity": "Severe"}),
    ("Dry Cough",          "SideEffect", {"severity": "Mild"}),
    ("Dizziness",          "SideEffect", {"severity": "Mild"}),
    ("Fatigue",            "SideEffect", {"severity": "Mild"}),
    ("Severe Pain",        "Symptom",    {}),
    ("Joint Pain",         "Symptom",    {}),
    ("Frequent Urination", "Symptom",    {}),
    ("High Blood Pressure","Symptom",    {}),
]

RELATIONSHIPS = [
    # (source_name, rel_type, target_name)
    ("Aspirin",    "TREATS",          "Headache"),
    ("Aspirin",    "TREATS",          "Heart Disease"),
    ("Aspirin",    "CAUSES",          "Nausea"),
    ("Aspirin",    "CAUSES",          "Stomach Bleeding"),
    ("Aspirin",    "INTERACTS_WITH",  "Warfarin"),
    ("Ibuprofen",  "TREATS",          "Headache"),
    ("Ibuprofen",  "TREATS",          "Arthritis"),
    ("Ibuprofen",  "CAUSES",          "Nausea"),
    ("Ibuprofen",  "CAUSES",          "Kidney Damage"),
    ("Warfarin",   "TREATS",          "Heart Disease"),
    ("Warfarin",   "CAUSES",          "Stomach Bleeding"),
    ("Metformin",  "TREATS",          "Diabetes"),
    ("Metformin",  "CAUSES",          "Nausea"),
    ("Metformin",  "CAUSES",          "Fatigue"),
    ("Lisinopril", "TREATS",          "Hypertension"),
    ("Lisinopril", "CAUSES",          "Dry Cough"),
    ("Lisinopril", "CAUSES",          "Dizziness"),
    ("Headache",   "HAS_SYMPTOM",     "Severe Pain"),
    ("Arthritis",  "HAS_SYMPTOM",     "Joint Pain"),
    ("Diabetes",   "HAS_SYMPTOM",     "Frequent Urination"),
    ("Hypertension", "HAS_SYMPTOM",   "High Blood Pressure"),
]

# ---------------------------------------------------------------------------
# Test queries (query, expected_intent, expected_node_names_subset)
# ---------------------------------------------------------------------------

TEST_QUERIES = [
    (
        "What are the side effects of aspirin?",
        "side_effects",
        {"Aspirin"},
    ),
    (
        "What does ibuprofen treat?",
        "treatment",
        {"Ibuprofen"},
    ),
    (
        "What treats headaches?",
        "treated_by",
        {"Headache"},
    ),
    (
        "Can I take aspirin with warfarin?",
        "interaction",
        {"Aspirin", "Warfarin"},
    ),
    (
        "What are the symptoms of diabetes?",
        "symptoms",
        {"Diabetes"},
    ),
]


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------

def seed_graph(connector: GraphDBConnector) -> None:
    print("\n── Seeding medical knowledge graph ─────────────────────────────────")
    connector.execute_query("MATCH (n) DETACH DELETE n")
    print("  Cleared existing data")

    for name, label, props in NODES:
        connector.execute_query(
            f"MERGE (n:{label} {{name: $name}}) SET n += $props",
            {"name": name, "props": props},
        )
    print(f"  Created {len(NODES)} nodes")

    for src, rel, tgt in RELATIONSHIPS:
        connector.execute_query(
            f"""
            MATCH (a {{name: $src}}), (b {{name: $tgt}})
            MERGE (a)-[:{rel}]->(b)
            """,
            {"src": src, "tgt": tgt},
        )
    print(f"  Created {len(RELATIONSHIPS)} relationships")


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(
    query: str,
    extractor: QueryTimeEntityExtractor,
    classifier: IntentClassifier,
    engine: SmartTraversalEngine,
    gen: ContextGenerator,
    expected_intent: str,
    expected_entry_names: set[str],
    llm=None,
) -> bool:
    """
    Run the full pipeline for one query. Returns True if all assertions pass.
    """
    print(f"\n  Query   : \"{query}\"")
    errors = []

    # Step 1: Entity extraction
    entry_nodes = extractor.extract_entry_nodes(query)
    found_names = {n["name"] for n in entry_nodes}
    print(f"  Nodes   : {found_names}")
    if not expected_entry_names.issubset(found_names):
        errors.append(
            f"    FAIL: expected {expected_entry_names} in entry nodes, got {found_names}"
        )

    # Step 2: Intent classification
    intent = classifier.classify(query)
    print(f"  Intent  : {intent}")
    if intent != expected_intent:
        errors.append(f"    FAIL: expected intent '{expected_intent}', got '{intent}'")

    # Step 3: Traversal
    subgraph = engine.traverse(entry_nodes, intent)
    node_count = len(subgraph["nodes"])
    rel_count  = len(subgraph["relationships"])
    print(f"  Subgraph: {node_count} nodes, {rel_count} relationships")

    # Step 4: Context generation
    context = gen.generate(subgraph)
    print(f"  Context :\n{chr(10).join('    ' + l for l in context.splitlines())}")

    # Step 5: LLM (optional)
    if llm:
        answer = llm.answer(query, context)
        print(f"  LLM     : {answer}")

    if errors:
        for e in errors:
            print(e)
        return False

    print("  PASS")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Connecting to Neo4j Aura …")
    connector = GraphDBConnector()
    connector.check_connection()

    seed_graph(connector)

    extractor  = QueryTimeEntityExtractor(connector, search_properties=["name"])
    classifier = IntentClassifier(CONFIG_PATH)
    engine     = SmartTraversalEngine(connector, CONFIG_PATH)
    gen        = ContextGenerator(CONFIG_PATH)

    # Try to load LLM — skip gracefully if key not configured
    llm = None
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if gemini_key and gemini_key != "your_gemini_api_key_here":
        from llm_interface import LLMInterface
        llm = LLMInterface(api_key=gemini_key)
        print("\nGemini LLM: enabled")
    else:
        print("\nGemini LLM: skipped (set GEMINI_API_KEY in .env to enable)")

    print("\n═══ Running pipeline tests ═══════════════════════════════════════════")
    passed = 0
    failed = 0
    for query, expected_intent, expected_names in TEST_QUERIES:
        ok = run_pipeline(
            query, extractor, classifier, engine, gen,
            expected_intent, expected_names, llm
        )
        if ok:
            passed += 1
        else:
            failed += 1

    print(f"\n═══ Results: {passed} passed, {failed} failed ═════════════════════════")
    connector.close()

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
