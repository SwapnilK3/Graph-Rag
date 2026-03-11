"""
test_complex_pipeline.py
------------------------
Live end-to-end test of all five traversal strategies against Neo4j Aura.

Asserts actual graph results for each strategy, then asks Gemini to
answer a complex multi-hop question from the retrieved context.

Run:
    python test_complex_pipeline.py
"""

import os
import sys
import time

from connector import GraphDBConnector
from entity_extractor import QueryTimeEntityExtractor
from intent_classifier import IntentClassifier
from traversal_engine import SmartTraversalEngine
from context_generator import ContextGenerator
from llm_interface import LLMInterface

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config", "medical_graph.json")

# ---------------------------------------------------------------------------
# Rich medical knowledge graph
# (designed so every hopping strategy has meaningful paths to traverse)
# ---------------------------------------------------------------------------

NODES = [
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

RELATIONSHIPS = [
    # Aspirin
    ("Aspirin",      "TREATS",           "Headache"),
    ("Aspirin",      "TREATS",           "Heart Disease"),
    ("Aspirin",      "CAUSES",           "Nausea"),
    ("Aspirin",      "CAUSES",           "Stomach Bleeding"),
    ("Aspirin",      "INTERACTS_WITH",   "Warfarin"),
    # Ibuprofen
    ("Ibuprofen",    "TREATS",           "Headache"),
    ("Ibuprofen",    "TREATS",           "Arthritis"),
    ("Ibuprofen",    "CAUSES",           "Nausea"),           # shared with Aspirin
    ("Ibuprofen",    "CAUSES",           "Kidney Damage"),
    ("Ibuprofen",    "INTERACTS_WITH",   "Warfarin"),
    # Warfarin
    ("Warfarin",     "TREATS",           "Heart Disease"),
    ("Warfarin",     "CAUSES",           "Stomach Bleeding"),  # shared with Aspirin
    # Metformin
    ("Metformin",    "TREATS",           "Diabetes"),
    ("Metformin",    "CAUSES",           "Nausea"),            # shared with Aspirin, Ibuprofen
    ("Metformin",    "CAUSES",           "Fatigue"),
    # Lisinopril
    ("Lisinopril",   "TREATS",           "Hypertension"),
    ("Lisinopril",   "CAUSES",           "Dry Cough"),
    ("Lisinopril",   "CAUSES",           "Dizziness"),
    # Omeprazole
    ("Omeprazole",   "TREATS",           "Peptic Ulcer"),
    ("Omeprazole",   "CAUSES",           "Diarrhea"),
    # Atorvastatin
    ("Atorvastatin", "TREATS",           "High Cholesterol"),
    ("Atorvastatin", "CAUSES",           "Muscle Pain"),
    ("Atorvastatin", "CAUSES",           "Liver Damage"),
    # Amoxicillin
    ("Amoxicillin",  "TREATS",           "Bacterial Infection"),
    ("Amoxicillin",  "CAUSES",           "Diarrhea"),          # shared with Omeprazole
    ("Amoxicillin",  "INTERACTS_WITH",   "Warfarin"),
    # Disease → Symptom (enables chained: Drug-TREATS->Disease-HAS_SYMPTOM->Symptom)
    ("Headache",          "HAS_SYMPTOM", "Severe Pain"),
    ("Arthritis",         "HAS_SYMPTOM", "Joint Pain"),
    ("Diabetes",          "HAS_SYMPTOM", "Frequent Urination"),
    ("Hypertension",      "HAS_SYMPTOM", "High Blood Pressure"),
    ("Heart Disease",     "HAS_SYMPTOM", "Chest Pain"),
    ("Peptic Ulcer",      "HAS_SYMPTOM", "Burning Sensation"),
    ("Peptic Ulcer",      "HAS_SYMPTOM", "Stomach Bleeding"),
    ("Bacterial Infection","HAS_SYMPTOM","Fever"),
    # SideEffect → Disease (enables drug_risk_chain: Drug-CAUSES->SE-INCREASES_RISK_OF->Disease)
    ("Stomach Bleeding",  "INCREASES_RISK_OF", "Peptic Ulcer"),
    ("Kidney Damage",     "INCREASES_RISK_OF", "Hypertension"),
    ("Muscle Pain",       "INCREASES_RISK_OF", "Heart Disease"),
]


# ---------------------------------------------------------------------------
# Test definitions
# ---------------------------------------------------------------------------

TESTS = [
    # (description, query, expected_intent, expected_strategy,
    #   expected_entry_names, expected_result_names)
    (
        "1-hop side effects",
        "What are the side effects of aspirin?",
        "side_effects", "targeted",
        {"Aspirin"},
        {"Nausea", "Stomach Bleeding"},
    ),
    (
        "1-hop treated-by (reverse)",
        "What treats headaches?",
        "treated_by", "targeted",
        {"Headache"},
        {"Aspirin", "Ibuprofen"},
    ),
    (
        "1-hop drug interaction",
        "Can I take aspirin with warfarin?",
        "interaction", "targeted",
        {"Aspirin", "Warfarin"},
        {"Aspirin", "Warfarin"},
    ),
    (
        "2-hop chained: Drug→Disease→Symptom",
        "What symptoms does aspirin indirectly cause through the diseases it treats?",
        "indirect_symptoms", "chained",
        {"Aspirin"},
        {"Severe Pain", "Chest Pain"},   # Headache→Severe Pain, HeartDisease→Chest Pain
    ),
    (
        "2-hop chained: Drug→SideEffect→Disease (risk chain)",
        "What diseases does aspirin lead to through its side effects?",
        "drug_risk_chain", "chained",
        {"Aspirin"},
        {"Peptic Ulcer"},                # Stomach Bleeding→INCREASES_RISK_OF→Peptic Ulcer
    ),
    (
        "2-hop variable neighborhood",
        "Give me an overview of metformin",
        "neighborhood", "variable_hop",
        {"Metformin"},
        {"Diabetes", "Frequent Urination"},  # Metformin→Diabetes (hop1), Diabetes→Symptom (hop2)
    ),
    (
        "shortest path between two nodes",
        "How is aspirin connected to peptic ulcer?",
        "connection", "shortest_path",
        {"Aspirin", "Peptic Ulcer"},
        {"Stomach Bleeding"},          # path: Aspirin→Stomach Bleeding→Peptic Ulcer
    ),
    (
        "shared neighbor: common side effects",
        "What side effects do aspirin and ibuprofen share?",
        "shared_effects", "shared_neighbor",
        {"Aspirin", "Ibuprofen"},
        {"Nausea"},                    # both cause Nausea
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def seed_graph(connector: GraphDBConnector) -> None:
    print("\n── Seeding rich medical graph ───────────────────────────────────────")
    connector.execute_query("MATCH (n) DETACH DELETE n")
    for name, label, props in NODES:
        connector.execute_query(
            f"MERGE (n:{label} {{name: $name}}) SET n += $props",
            {"name": name, "props": props},
        )
    for src, rel, tgt in RELATIONSHIPS:
        connector.execute_query(
            f"MATCH (a {{name: $src}}), (b {{name: $tgt}}) MERGE (a)-[:{rel}]->(b)",
            {"src": src, "tgt": tgt},
        )
    print(f"  {len(NODES)} nodes, {len(RELATIONSHIPS)} relationships\n")


def run_test(
    desc, query, expected_intent, expected_strategy,
    expected_entry_names, expected_result_names,
    extractor, classifier, engine, gen, llm=None,
) -> bool:
    print(f"  ── {desc}")
    print(f"     Query   : \"{query}\"")
    errors = []

    entry_nodes = extractor.extract_entry_nodes(query)
    found_names = {n["name"] for n in entry_nodes}
    print(f"     Nodes   : {found_names}")

    if not expected_entry_names.issubset(found_names):
        errors.append(f"     FAIL entry nodes: expected {expected_entry_names}, got {found_names}")

    intent = classifier.classify(query)
    print(f"     Intent  : {intent}")
    if intent != expected_intent:
        errors.append(f"     FAIL intent: expected '{expected_intent}', got '{intent}'")

    subgraph = engine.traverse(entry_nodes, intent)
    strategy = subgraph.get("strategy", "?")
    hop_depth = subgraph.get("hop_depth", 0)
    result_names = {n["name"] for n in subgraph["nodes"]}
    print(f"     Strategy: {strategy}  depth={hop_depth}")
    print(f"     Results : {result_names}")

    if strategy != expected_strategy:
        errors.append(f"     FAIL strategy: expected '{expected_strategy}', got '{strategy}'")

    if not expected_result_names.issubset(result_names):
        errors.append(
            f"     FAIL result nodes: expected {expected_result_names} to be in {result_names}"
        )

    context = gen.generate(subgraph)

    if llm:
        for attempt in range(3):
            try:
                answer = llm.answer(query, context)
                break
            except Exception as exc:
                msg = str(exc)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    wait = 60 if attempt == 0 else 90
                    print(f"     (rate limited, waiting {wait}s …)")
                    time.sleep(wait)
                else:
                    raise
        else:
            answer = "(LLM unavailable after retries)"
        wrapped = "\n     ".join(answer.splitlines())
        print(f"     LLM     : {wrapped}")
        time.sleep(13)  # stay under 5 RPM free-tier limit

    if errors:
        for e in errors:
            print(e)
        print("     FAIL ✗")
        return False

    print("     PASS ✓")
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

    llm = None
    key = os.getenv("GEMINI_API_KEY", "")
    if key and key != "your_gemini_api_key_here":
        llm = LLMInterface(api_key=key)
        print("Gemini LLM  : enabled\n")
    else:
        print("Gemini LLM  : skipped (GEMINI_API_KEY not set)\n")

    print("═══ Running complex pipeline tests ══════════════════════════════════")
    passed, failed = 0, 0

    for args in TESTS:
        print()
        ok = run_test(*args, extractor, classifier, engine, gen, llm)
        if ok:
            passed += 1
        else:
            failed += 1

    # ── Bonus: one complex LLM-only question using the richest context ──
    if llm:
        print("\n═══ Bonus: complex multi-hop LLM question ════════════════════════")
        bonus_query = (
            "A patient takes aspirin for heart disease. "
            "What symptoms might they experience, and what drug interaction risks exist?"
        )
        entry_nodes  = extractor.extract_entry_nodes(bonus_query)
        # Use variable_hop so we get the full 2-hop neighbourhood
        subgraph     = engine.traverse(entry_nodes, "neighborhood")
        context      = gen.generate(subgraph)
        try:
            time.sleep(13)
            answer = llm.answer(bonus_query, context)
        except Exception as exc:
            if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                time.sleep(90)
                answer = llm.answer(bonus_query, context)
            else:
                raise
        print(f"  Query : {bonus_query}")
        print(f"  Answer: {answer}")

    print(f"\n═══ Results: {passed} passed, {failed} failed ═════════════════════")

    connector.close()
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
