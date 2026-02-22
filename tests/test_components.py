"""
test_components.py
------------------
Offline unit tests for Components 2–5.

All Neo4j calls are mocked — no database connection required.
Run with:
    pytest test_components.py -v
"""

import json
import os
import pytest
from unittest.mock import MagicMock, patch

from intent_classifier import IntentClassifier
from traversal_engine import SmartTraversalEngine
from context_generator import ContextGenerator

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config", "medical_graph.json")


# ===========================================================================
# Helpers
# ===========================================================================

def make_node(node_id: str, label: str, name: str, **props) -> dict:
    properties = {"name": name, **props}
    return {"id": node_id, "label": label, "name": name, "properties": properties}


def make_subgraph(nodes: list[dict], rels: list[dict]) -> dict:
    return {"nodes": nodes, "relationships": rels}


def make_traversal_row(
    src_id, src_label, src_name, rel_type, tgt_id, tgt_label, tgt_name
) -> dict:
    return {
        "source_id":    src_id,
        "source_label": src_label,
        "source_props": {"name": src_name},
        "rel_type":     rel_type,
        "rel_props":    {},
        "target_id":    tgt_id,
        "target_label": tgt_label,
        "target_props": {"name": tgt_name},
    }


# ===========================================================================
# Component 2: IntentClassifier
# ===========================================================================

class TestIntentClassifier:

    @pytest.fixture
    def clf(self):
        return IntentClassifier(CONFIG_PATH)

    def test_side_effects_keyword(self, clf):
        assert clf.classify("What are the side effects of aspirin?") == "side_effects"

    def test_adverse_keyword(self, clf):
        assert clf.classify("Any adverse reactions to ibuprofen?") == "side_effects"

    def test_treatment_keyword(self, clf):
        assert clf.classify("What does aspirin treat?") == "treatment"

    def test_treated_by_keyword(self, clf):
        assert clf.classify("What treats headaches?") == "treated_by"

    def test_interaction_keyword(self, clf):
        assert clf.classify("Can I take aspirin with warfarin?") == "interaction"

    def test_symptoms_keyword(self, clf):
        assert clf.classify("What are the symptoms of diabetes?") == "symptoms"

    def test_fallback_to_general(self, clf):
        assert clf.classify("Tell me about aspirin") == "general"

    def test_case_insensitive(self, clf):
        assert clf.classify("SIDE EFFECTS of aspirin") == "side_effects"

    def test_empty_query_returns_general(self, clf):
        assert clf.classify("") == "general"

    def test_all_intents_listed(self, clf):
        intents = clf.all_intents()
        assert "side_effects" in intents
        assert "treatment" in intents
        assert "treated_by" in intents
        assert "interaction" in intents
        assert "symptoms" in intents


# ===========================================================================
# Component 3: SmartTraversalEngine
# ===========================================================================

class TestSmartTraversalEngine:

    @pytest.fixture
    def mock_connector(self):
        conn = MagicMock()
        conn.execute_query.return_value = []
        return conn

    @pytest.fixture
    def engine(self, mock_connector):
        return SmartTraversalEngine(mock_connector, CONFIG_PATH)

    def test_entry_ids_passed_to_query(self, engine, mock_connector):
        entry = [make_node("id:1", "Drug", "Aspirin")]
        engine.traverse(entry, "side_effects")
        call_params = mock_connector.execute_query.call_args[0][1]
        assert call_params["ids"] == ["id:1"]

    def test_source_anchor_uses_source_where(self, engine, mock_connector):
        entry = [make_node("id:1", "Drug", "Aspirin")]
        engine.traverse(entry, "side_effects")
        cypher = mock_connector.execute_query.call_args[0][0]
        assert "elementId(src) IN $ids" in cypher

    def test_target_anchor_uses_target_where(self, engine, mock_connector):
        # "treated_by" uses entry_anchor = "target"
        entry = [make_node("id:2", "Disease", "Headache")]
        engine.traverse(entry, "treated_by")
        cypher = mock_connector.execute_query.call_args[0][0]
        assert "elementId(tgt) IN $ids" in cypher

    def test_either_anchor_uses_or_clause(self, engine, mock_connector):
        # "interaction" uses entry_anchor = "either"
        entry = [make_node("id:1", "Drug", "Aspirin")]
        engine.traverse(entry, "interaction")
        cypher = mock_connector.execute_query.call_args[0][0]
        assert "OR" in cypher

    def test_unknown_intent_falls_back_to_general(self, engine, mock_connector):
        entry = [make_node("id:1", "Drug", "Aspirin")]
        engine.traverse(entry, "general")
        cypher = mock_connector.execute_query.call_args[0][0]
        assert "MATCH (src)-[r]-(tgt)" in cypher

    def test_empty_entry_nodes_returns_empty_subgraph(self, engine):
        result = engine.traverse([], "side_effects")
        assert result["nodes"] == []
        assert result["relationships"] == []
        assert result["strategy"] == "none"

    def test_subgraph_contains_entry_nodes(self, engine, mock_connector):
        mock_connector.execute_query.return_value = []
        entry = [make_node("id:99", "Drug", "Metformin")]
        result = engine.traverse(entry, "side_effects")
        ids = [n["id"] for n in result["nodes"]]
        assert "id:99" in ids

    def test_subgraph_has_strategy_key(self, engine, mock_connector):
        mock_connector.execute_query.return_value = []
        entry = [make_node("id:1", "Drug", "Aspirin")]
        result = engine.traverse(entry, "side_effects")
        assert "strategy" in result
        assert result["strategy"] == "targeted"

    def test_subgraph_has_hop_depth_key(self, engine, mock_connector):
        mock_connector.execute_query.return_value = []
        entry = [make_node("id:1", "Drug", "Aspirin")]
        result = engine.traverse(entry, "side_effects")
        assert "hop_depth" in result
        assert result["hop_depth"] == 1

    def test_subgraph_deduplicates_nodes(self, engine, mock_connector):
        row = make_traversal_row("id:1", "Drug", "Aspirin", "CAUSES", "id:2", "SideEffect", "Nausea")
        # Return same row twice
        mock_connector.execute_query.return_value = [row, row]
        entry = [make_node("id:1", "Drug", "Aspirin")]
        result = engine.traverse(entry, "side_effects")
        node_ids = [n["id"] for n in result["nodes"]]
        assert len(node_ids) == len(set(node_ids)), "Duplicate nodes in subgraph"

    def test_subgraph_relationships_are_captured(self, engine, mock_connector):
        row = make_traversal_row("id:1", "Drug", "Aspirin", "CAUSES", "id:2", "SideEffect", "Nausea")
        mock_connector.execute_query.return_value = [row]
        entry = [make_node("id:1", "Drug", "Aspirin")]
        result = engine.traverse(entry, "side_effects")
        assert len(result["relationships"]) == 1
        assert result["relationships"][0]["type"] == "CAUSES"

    def test_correct_relationship_type_in_cypher(self, engine, mock_connector):
        entry = [make_node("id:1", "Drug", "Aspirin")]
        engine.traverse(entry, "side_effects")
        cypher = mock_connector.execute_query.call_args[0][0]
        assert "CAUSES" in cypher

    def test_correct_labels_in_cypher_for_treatment(self, engine, mock_connector):
        entry = [make_node("id:1", "Drug", "Aspirin")]
        engine.traverse(entry, "treatment")
        cypher = mock_connector.execute_query.call_args[0][0]
        assert "Drug" in cypher
        assert "Disease" in cypher
        assert "TREATS" in cypher


# ===========================================================================
# Component 4: ContextGenerator
# ===========================================================================

class TestContextGenerator:

    @pytest.fixture
    def gen(self):
        return ContextGenerator(CONFIG_PATH)

    def test_context_header_shows_strategy(self, gen):
        drug = make_node("id:1", "Drug", "Aspirin")
        subgraph = {"nodes": [drug], "relationships": [], "strategy": "targeted", "hop_depth": 1}
        result = gen.generate(subgraph)
        assert "targeted" in result
        assert "depth=1" in result

    def test_empty_subgraph_returns_no_info_message(self, gen):
        result = gen.generate({"nodes": [], "relationships": []})
        assert "No relevant information" in result

    def test_entities_section_present(self, gen):
        subgraph = make_subgraph([make_node("id:1", "Drug", "Aspirin")], [])
        assert "ENTITIES:" in gen.generate(subgraph)

    def test_node_name_and_label_in_output(self, gen):
        subgraph = make_subgraph([make_node("id:1", "Drug", "Aspirin")], [])
        result = gen.generate(subgraph)
        assert "Aspirin" in result
        assert "Drug" in result

    def test_relationships_section_present_when_rels_exist(self, gen):
        drug = make_node("id:1", "Drug", "Aspirin")
        se   = make_node("id:2", "SideEffect", "Nausea")
        rel  = {"source_id": "id:1", "target_id": "id:2", "type": "CAUSES", "properties": {}}
        result = gen.generate(make_subgraph([drug, se], [rel]))
        assert "RELATIONSHIPS:" in result

    def test_treats_template_applied(self, gen):
        drug    = make_node("id:1", "Drug", "Aspirin")
        disease = make_node("id:2", "Disease", "Headache")
        rel     = {"source_id": "id:1", "target_id": "id:2", "type": "TREATS", "properties": {}}
        result  = gen.generate(make_subgraph([drug, disease], [rel]))
        assert "Aspirin treats Headache" in result

    def test_causes_template_applied(self, gen):
        drug = make_node("id:1", "Drug", "Aspirin")
        se   = make_node("id:2", "SideEffect", "Nausea")
        rel  = {"source_id": "id:1", "target_id": "id:2", "type": "CAUSES", "properties": {}}
        result = gen.generate(make_subgraph([drug, se], [rel]))
        assert "Aspirin may cause Nausea" in result

    def test_unknown_rel_type_uses_default_template(self, gen):
        n1 = make_node("id:1", "Drug", "Aspirin")
        n2 = make_node("id:2", "Concept", "Something")
        rel = {"source_id": "id:1", "target_id": "id:2", "type": "UNKNOWN_REL", "properties": {}}
        result = gen.generate(make_subgraph([n1, n2], [rel]))
        # Should not crash; should include both names
        assert "Aspirin" in result
        assert "Something" in result

    def test_extra_properties_shown_in_output(self, gen):
        drug = make_node("id:1", "Drug", "Aspirin", type="NSAID")
        result = gen.generate(make_subgraph([drug], []))
        assert "NSAID" in result

    def test_no_relationships_section_when_none(self, gen):
        subgraph = make_subgraph([make_node("id:1", "Drug", "Aspirin")], [])
        result = gen.generate(subgraph)
        assert "RELATIONSHIPS:" not in result


# ===========================================================================
# Component 5: LLMInterface — import-only & bad-key guard
# ===========================================================================

class TestLLMInterfaceGuards:

    def test_raises_on_missing_key(self):
        """LLMInterface must raise ValueError when no API key is set."""
        from llm_interface import LLMInterface
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=True):
            with pytest.raises(ValueError, match="Gemini API key not found"):
                LLMInterface(api_key=None)

    def test_raises_on_placeholder_key(self):
        """Placeholder value in .env must also be rejected."""
        from llm_interface import LLMInterface
        with pytest.raises(ValueError):
            LLMInterface(api_key="your_gemini_api_key_here")
