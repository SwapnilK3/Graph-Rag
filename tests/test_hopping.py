"""
test_hopping.py
---------------
Offline unit tests for all five traversal strategies plus the new
multi-hop intent patterns in the classifier.

All Neo4j calls are mocked — no database required.
Run with:
    pytest test_hopping.py -v
"""

import os
import pytest
from unittest.mock import MagicMock

from intent_classifier import IntentClassifier
from traversal_engine import SmartTraversalEngine

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config", "medical_graph.json")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_node(node_id: str, label: str, name: str) -> dict:
    return {"id": node_id, "label": label, "name": name, "properties": {"name": name}}


def make_row(src_id, src_label, src_name, rel, tgt_id, tgt_label, tgt_name) -> dict:
    return {
        "source_id":    src_id,
        "source_label": src_label,
        "source_props": {"name": src_name},
        "rel_type":     rel,
        "rel_props":    {},
        "target_id":    tgt_id,
        "target_label": tgt_label,
        "target_props": {"name": tgt_name},
    }


@pytest.fixture
def mock_connector():
    conn = MagicMock()
    conn.execute_query.return_value = []
    return conn


@pytest.fixture
def engine(mock_connector):
    return SmartTraversalEngine(mock_connector, CONFIG_PATH)


@pytest.fixture
def clf():
    return IntentClassifier(CONFIG_PATH)


# ===========================================================================
# New intent keywords in the classifier
# ===========================================================================

class TestNewIntents:
    def test_indirect_symptoms_keyword(self, clf):
        assert clf.classify("What symptoms does aspirin indirectly cause?") == "indirect_symptoms"

    def test_neighborhood_keyword(self, clf):
        assert clf.classify("Give me an overview of metformin") == "neighborhood"

    def test_connection_keyword(self, clf):
        assert clf.classify("How is aspirin connected to kidney damage?") == "connection"

    def test_shared_effects_keyword(self, clf):
        assert clf.classify("What side effects do aspirin and ibuprofen share?") == "shared_effects"

    def test_drug_risk_chain_keyword(self, clf):
        assert clf.classify("What does aspirin lead to?") == "drug_risk_chain"

    def test_all_new_intents_listed(self, clf):
        intents = clf.all_intents()
        for name in ("indirect_symptoms", "drug_risk_chain", "neighborhood",
                     "connection", "shared_effects"):
            assert name in intents, f"Intent '{name}' missing from all_intents()"


# ===========================================================================
# Strategy: chained
# ===========================================================================

class TestChainedStrategy:

    def test_strategy_label_is_chained(self, engine, mock_connector):
        entry = [make_node("id:1", "Drug", "Aspirin")]
        result = engine.traverse(entry, "indirect_symptoms")
        assert result["strategy"] == "chained"

    def test_hop_depth_matches_number_of_hops(self, engine, mock_connector):
        # indirect_symptoms has 2 hops: TREATS -> HAS_SYMPTOM
        entry = [make_node("id:1", "Drug", "Aspirin")]
        result = engine.traverse(entry, "indirect_symptoms")
        assert result["hop_depth"] == 2

    def test_cypher_contains_first_relationship(self, engine, mock_connector):
        entry = [make_node("id:1", "Drug", "Aspirin")]
        engine.traverse(entry, "indirect_symptoms")
        cypher = mock_connector.execute_query.call_args[0][0]
        assert "TREATS" in cypher

    def test_cypher_contains_second_relationship(self, engine, mock_connector):
        entry = [make_node("id:1", "Drug", "Aspirin")]
        engine.traverse(entry, "indirect_symptoms")
        cypher = mock_connector.execute_query.call_args[0][0]
        assert "HAS_SYMPTOM" in cypher

    def test_cypher_uses_path_unwind(self, engine, mock_connector):
        entry = [make_node("id:1", "Drug", "Aspirin")]
        engine.traverse(entry, "indirect_symptoms")
        cypher = mock_connector.execute_query.call_args[0][0]
        assert "UNWIND" in cypher
        assert "length(path)" in cypher

    def test_cypher_anchors_on_entry_label(self, engine, mock_connector):
        # entry_label is "Drug" for indirect_symptoms
        entry = [make_node("id:1", "Drug", "Aspirin")]
        engine.traverse(entry, "indirect_symptoms")
        cypher = mock_connector.execute_query.call_args[0][0]
        assert "n0:Drug" in cypher

    def test_multi_hop_results_included_in_subgraph(self, engine, mock_connector):
        # Row representing hop 1: Aspirin -TREATS-> Headache
        row1 = make_row("id:1", "Drug", "Aspirin", "TREATS", "id:2", "Disease", "Headache")
        # Row representing hop 2: Headache -HAS_SYMPTOM-> Severe Pain
        row2 = make_row("id:2", "Disease", "Headache", "HAS_SYMPTOM", "id:3", "Symptom", "Severe Pain")
        mock_connector.execute_query.return_value = [row1, row2]

        entry = [make_node("id:1", "Drug", "Aspirin")]
        result = engine.traverse(entry, "indirect_symptoms")

        names = {n["name"] for n in result["nodes"]}
        assert "Aspirin" in names
        assert "Headache" in names
        assert "Severe Pain" in names

    def test_three_hop_chain_depth(self, engine, mock_connector):
        # drug_risk_chain also has 2 hops but tests a different chain
        entry = [make_node("id:1", "Drug", "Aspirin")]
        result = engine.traverse(entry, "drug_risk_chain")
        assert result["hop_depth"] == 2
        cypher = mock_connector.execute_query.call_args[0][0]
        assert "CAUSES" in cypher
        assert "INCREASES_RISK_OF" in cypher


# ===========================================================================
# Strategy: variable_hop
# ===========================================================================

class TestVariableHopStrategy:

    def test_strategy_label_is_variable_hop(self, engine, mock_connector):
        entry = [make_node("id:1", "Drug", "Metformin")]
        result = engine.traverse(entry, "neighborhood")
        assert result["strategy"] == "variable_hop"

    def test_hop_depth_equals_max_hops(self, engine, mock_connector):
        # neighborhood uses max_hops=2
        entry = [make_node("id:1", "Drug", "Metformin")]
        result = engine.traverse(entry, "neighborhood")
        assert result["hop_depth"] == 2

    def test_cypher_contains_variable_hop_syntax(self, engine, mock_connector):
        entry = [make_node("id:1", "Drug", "Metformin")]
        engine.traverse(entry, "neighborhood")
        cypher = mock_connector.execute_query.call_args[0][0]
        # Should contain [*1..2] style range
        assert "[*" in cypher
        assert ".." in cypher

    def test_cypher_uses_path_unwind(self, engine, mock_connector):
        entry = [make_node("id:1", "Drug", "Metformin")]
        engine.traverse(entry, "neighborhood")
        cypher = mock_connector.execute_query.call_args[0][0]
        assert "UNWIND" in cypher
        assert "length(path)" in cypher

    def test_intermediate_nodes_appear_in_subgraph(self, engine, mock_connector):
        # Simulate 2-hop path: Metformin -> Diabetes -> Frequent Urination
        row1 = make_row("id:1", "Drug", "Metformin",      "TREATS",      "id:2", "Disease", "Diabetes")
        row2 = make_row("id:2", "Disease", "Diabetes",    "HAS_SYMPTOM", "id:3", "Symptom", "Frequent Urination")
        mock_connector.execute_query.return_value = [row1, row2]

        entry = [make_node("id:1", "Drug", "Metformin")]
        result = engine.traverse(entry, "neighborhood")

        names = {n["name"] for n in result["nodes"]}
        assert "Diabetes" in names           # intermediate node at hop 1
        assert "Frequent Urination" in names  # leaf node at hop 2

    def test_entry_node_always_in_subgraph(self, engine, mock_connector):
        mock_connector.execute_query.return_value = []
        entry = [make_node("id:42", "Drug", "Lisinopril")]
        result = engine.traverse(entry, "neighborhood")
        assert any(n["id"] == "id:42" for n in result["nodes"])


# ===========================================================================
# Strategy: shortest_path
# ===========================================================================

class TestShortestPathStrategy:

    def test_strategy_label_is_shortest_path(self, engine, mock_connector):
        entry = [make_node("id:1", "Drug", "Aspirin"),
                 make_node("id:2", "SideEffect", "Kidney Damage")]
        result = engine.traverse(entry, "connection")
        assert result["strategy"] == "shortest_path"

    def test_cypher_uses_shortestpath_function(self, engine, mock_connector):
        entry = [make_node("id:1", "Drug", "Aspirin"),
                 make_node("id:5", "Symptom", "Joint Pain")]
        engine.traverse(entry, "connection")
        cypher = mock_connector.execute_query.call_args[0][0]
        assert "shortestPath" in cypher

    def test_cypher_pairs_ids_lexicographically(self, engine, mock_connector):
        """Both entry-node IDs must appear in the WHERE clause."""
        entry = [make_node("id:1", "Drug", "Aspirin"),
                 make_node("id:2", "Drug", "Warfarin")]
        engine.traverse(entry, "connection")
        cypher = mock_connector.execute_query.call_args[0][0]
        assert "$ids" in cypher

    def test_single_entry_falls_back_to_variable_hop(self, engine, mock_connector):
        """When only one entry node exists, shortest_path degrades to variable_hop."""
        entry = [make_node("id:1", "Drug", "Aspirin")]
        result = engine.traverse(entry, "connection")
        # Falls back: cypher must not contain shortestPath
        cypher = mock_connector.execute_query.call_args[0][0]
        assert "shortestPath" not in cypher
        assert "[*" in cypher  # variable hop range present

    def test_path_nodes_in_subgraph(self, engine, mock_connector):
        # Path: Aspirin -> Peptic Ulcer -> Omeprazole
        row1 = make_row("id:1", "Drug", "Aspirin",      "CAUSES", "id:6", "Disease",    "Peptic Ulcer")
        row2 = make_row("id:6", "Disease", "Peptic Ulcer", "TREATS", "id:7", "Drug", "Omeprazole")
        mock_connector.execute_query.return_value = [row1, row2]

        entry = [make_node("id:1", "Drug", "Aspirin"),
                 make_node("id:7", "Drug", "Omeprazole")]
        result = engine.traverse(entry, "connection")

        names = {n["name"] for n in result["nodes"]}
        assert "Peptic Ulcer" in names  # intermediate node on the path


# ===========================================================================
# Strategy: shared_neighbor
# ===========================================================================

class TestSharedNeighborStrategy:

    def test_strategy_label_is_shared_neighbor(self, engine, mock_connector):
        entry = [make_node("id:1", "Drug", "Aspirin"),
                 make_node("id:2", "Drug", "Ibuprofen")]
        result = engine.traverse(entry, "shared_effects")
        assert result["strategy"] == "shared_neighbor"

    def test_hop_depth_is_one(self, engine, mock_connector):
        entry = [make_node("id:1", "Drug", "Aspirin"),
                 make_node("id:2", "Drug", "Ibuprofen")]
        result = engine.traverse(entry, "shared_effects")
        assert result["hop_depth"] == 1

    def test_cypher_checks_min_connections(self, engine, mock_connector):
        entry = [make_node("id:1", "Drug", "Aspirin"),
                 make_node("id:2", "Drug", "Ibuprofen")]
        engine.traverse(entry, "shared_effects")
        cypher = mock_connector.execute_query.call_args[0][0]
        assert "min_conn" in cypher
        assert "connected_entries" in cypher

    def test_shared_neighbor_in_subgraph(self, engine, mock_connector):
        # Both aspirin and ibuprofen CAUSE Nausea — Nausea should appear
        row1 = make_row("id:1", "Drug", "Aspirin",    "CAUSES", "id:10", "SideEffect", "Nausea")
        row2 = make_row("id:2", "Drug", "Ibuprofen",  "CAUSES", "id:10", "SideEffect", "Nausea")
        mock_connector.execute_query.return_value = [row1, row2]

        entry = [make_node("id:1", "Drug", "Aspirin"),
                 make_node("id:2", "Drug", "Ibuprofen")]
        result = engine.traverse(entry, "shared_effects")

        names = {n["name"] for n in result["nodes"]}
        assert "Nausea" in names

    def test_min_connections_param_passed(self, engine, mock_connector):
        entry = [make_node("id:1", "Drug", "Aspirin"),
                 make_node("id:2", "Drug", "Ibuprofen")]
        engine.traverse(entry, "shared_effects")
        params = mock_connector.execute_query.call_args[0][1]
        assert "min_conn" in params
        assert params["min_conn"] == 2  # min of config(2) and len(entry_ids)=2


# ===========================================================================
# Strategy: general (fallback)
# ===========================================================================

class TestGeneralStrategy:

    def test_general_strategy_on_unknown_intent(self, engine, mock_connector):
        entry = [make_node("id:1", "Drug", "Aspirin")]
        result = engine.traverse(entry, "unknown_intent")
        assert result["strategy"] == "general"

    def test_general_cypher_no_label_restriction(self, engine, mock_connector):
        entry = [make_node("id:1", "Drug", "Aspirin")]
        engine.traverse(entry, "nonexistent")
        cypher = mock_connector.execute_query.call_args[0][0]
        # General should match any node type
        assert "(src)-[r]-(tgt)" in cypher


# ===========================================================================
# Cross-strategy: subgraph structure invariants
# All strategies must produce the same output shape
# ===========================================================================

class TestSubgraphInvariants:
    INTENTS = [
        ("side_effects",     [make_node("id:1", "Drug", "Aspirin")]),
        ("neighborhood",     [make_node("id:1", "Drug", "Aspirin")]),
        ("indirect_symptoms",[make_node("id:1", "Drug", "Aspirin")]),
        ("connection",       [make_node("id:1", "Drug", "Aspirin"),
                               make_node("id:2", "Disease", "Headache")]),
        ("shared_effects",   [make_node("id:1", "Drug", "Aspirin"),
                               make_node("id:2", "Drug", "Ibuprofen")]),
    ]

    @pytest.mark.parametrize("intent,entry", INTENTS)
    def test_always_has_nodes_key(self, engine, mock_connector, intent, entry):
        result = engine.traverse(entry, intent)
        assert "nodes" in result

    @pytest.mark.parametrize("intent,entry", INTENTS)
    def test_always_has_relationships_key(self, engine, mock_connector, intent, entry):
        result = engine.traverse(entry, intent)
        assert "relationships" in result

    @pytest.mark.parametrize("intent,entry", INTENTS)
    def test_always_has_strategy_key(self, engine, mock_connector, intent, entry):
        result = engine.traverse(entry, intent)
        assert "strategy" in result

    @pytest.mark.parametrize("intent,entry", INTENTS)
    def test_always_has_hop_depth_key(self, engine, mock_connector, intent, entry):
        result = engine.traverse(entry, intent)
        assert "hop_depth" in result

    @pytest.mark.parametrize("intent,entry", INTENTS)
    def test_entry_nodes_always_present(self, engine, mock_connector, intent, entry):
        result = engine.traverse(entry, intent)
        result_ids = {n["id"] for n in result["nodes"]}
        for en in entry:
            assert en["id"] in result_ids
