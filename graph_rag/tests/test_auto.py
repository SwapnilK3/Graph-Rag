"""
test_auto.py
------------
Comprehensive tests for the automatic schema discovery and config generation.

Part A: Offline unit tests (no DB / API key needed) — tests the logic itself.
Part B: Live integration tests (needs Neo4j Aura) — tests against real graph.

Run offline tests:
    python -m pytest test_auto.py -k "not live" -v

Run all tests:
    python -m pytest test_auto.py -v

Run as script (live demo with full output):
    python test_auto.py
"""

import json
import pytest
from unittest.mock import MagicMock, patch


# ═══════════════════════════════════════════════════════════════════
#  Part A: OFFLINE UNIT TESTS
# ═══════════════════════════════════════════════════════════════════


# --- SchemaDiscovery offline tests ---

class TestSchemaDiscoveryOffline:
    """Test schema discovery logic with mocked Neo4j responses."""

    def _make_discovery(self, execute_side_effect):
        from schema_discovery import SchemaDiscovery
        connector = MagicMock()
        connector.execute_query = MagicMock(side_effect=execute_side_effect)
        return SchemaDiscovery(connector)

    def test_discover_labels(self):
        from schema_discovery import SchemaDiscovery
        connector = MagicMock()
        connector.execute_query.return_value = [
            {"label": "Drug"}, {"label": "Disease"}
        ]
        d = SchemaDiscovery(connector)
        labels = d._discover_labels()
        assert labels == ["Disease", "Drug"]  # sorted

    def test_count_nodes(self):
        from schema_discovery import SchemaDiscovery
        connector = MagicMock()
        connector.execute_query.return_value = [{"cnt": 5}]
        d = SchemaDiscovery(connector)
        counts = d._count_nodes(["Drug"])
        assert counts == {"Drug": 5}

    def test_discover_relationships(self):
        from schema_discovery import SchemaDiscovery
        connector = MagicMock()
        connector.execute_query.return_value = [
            {"rel_type": "TREATS", "source_label": "Drug",
             "target_label": "Disease", "cnt": 10}
        ]
        d = SchemaDiscovery(connector)
        rels = d._discover_relationships()
        assert len(rels) == 1
        assert rels[0]["type"] == "TREATS"
        assert rels[0]["source_label"] == "Drug"

    def test_infer_type_string(self):
        from schema_discovery import SchemaDiscovery
        assert SchemaDiscovery._infer_type("hello") == "String"

    def test_infer_type_int(self):
        from schema_discovery import SchemaDiscovery
        assert SchemaDiscovery._infer_type(42) == "Integer"

    def test_infer_type_float(self):
        from schema_discovery import SchemaDiscovery
        assert SchemaDiscovery._infer_type(3.14) == "Float"

    def test_infer_type_bool(self):
        from schema_discovery import SchemaDiscovery
        assert SchemaDiscovery._infer_type(True) == "Boolean"

    def test_infer_type_list(self):
        from schema_discovery import SchemaDiscovery
        assert SchemaDiscovery._infer_type([1, 2]) == "List"

    def test_infer_type_none(self):
        from schema_discovery import SchemaDiscovery
        assert SchemaDiscovery._infer_type(None) == "Unknown"

    def test_multifactor_scoring_name_plus_string_plus_cardinality(self):
        """A string property named 'name' with high cardinality should score 1.0."""
        from schema_discovery import SchemaDiscovery
        connector = MagicMock()
        d = SchemaDiscovery(connector)
        props = {
            "Drug": [{
                "name": "name", "type": "String", "unique_ratio": 1.0, "sample": ["Aspirin"]
            }]
        }
        result = d._identify_searchable_properties(["Drug"], props)
        assert "name" in result["Drug"]

    def test_multifactor_scoring_low_score_excluded(self):
        """An integer property named 'id' with low cardinality should be excluded."""
        from schema_discovery import SchemaDiscovery
        connector = MagicMock()
        d = SchemaDiscovery(connector)
        props = {
            "Drug": [{
                "name": "internal_id", "type": "Integer", "unique_ratio": 0.1, "sample": [1]
            }]
        }
        result = d._identify_searchable_properties(["Drug"], props)
        # Should fallback to empty (no string properties at all)
        assert result["Drug"] == []

    def test_multifactor_scoring_string_high_cardinality(self):
        """A string property with high cardinality but non-name should score 0.6 (pass)."""
        from schema_discovery import SchemaDiscovery
        connector = MagicMock()
        d = SchemaDiscovery(connector)
        props = {
            "Drug": [{
                "name": "formula", "type": "String", "unique_ratio": 0.9, "sample": ["C9H8O4"]
            }]
        }
        result = d._identify_searchable_properties(["Drug"], props)
        assert "formula" in result["Drug"]

    def test_fallback_to_any_string(self):
        """When nothing scores > 0.5, fallback picks first string property."""
        from schema_discovery import SchemaDiscovery
        connector = MagicMock()
        d = SchemaDiscovery(connector)
        props = {
            "Drug": [
                {"name": "code", "type": "Integer", "unique_ratio": 0.1, "sample": [1]},
                {"name": "desc", "type": "String", "unique_ratio": 0.2, "sample": ["x"]},
            ]
        }
        result = d._identify_searchable_properties(["Drug"], props)
        assert "desc" in result["Drug"]


# --- AutoConfigGenerator offline tests ---

class TestAutoConfigOffline:
    """Test auto config generation with mock schemas."""

    def _make_schema(self):
        return {
            "node_labels":    ["Disease", "Drug", "SideEffect"],
            "relationships":  [
                {"type": "TREATS",  "source_label": "Drug", "target_label": "Disease",    "count": 5},
                {"type": "CAUSES",  "source_label": "Drug", "target_label": "SideEffect", "count": 3},
            ],
            "properties": {
                "Drug":       [{"name": "name", "type": "String", "unique_ratio": 1.0, "sample": []}],
                "Disease":    [{"name": "name", "type": "String", "unique_ratio": 1.0, "sample": []}],
                "SideEffect": [{"name": "name", "type": "String", "unique_ratio": 1.0, "sample": []}],
            },
            "searchable_properties": {"Drug": ["name"], "Disease": ["name"], "SideEffect": ["name"]},
            "node_counts":    {"Drug": 3, "Disease": 3, "SideEffect": 2},
            "total_nodes":    8,
            "total_relationships": 8,
        }

    def test_generate_returns_dict(self):
        from auto_config import AutoConfigGenerator
        config = AutoConfigGenerator(self._make_schema()).generate()
        assert isinstance(config, dict)

    def test_has_intent_patterns(self):
        from auto_config import AutoConfigGenerator
        config = AutoConfigGenerator(self._make_schema()).generate()
        assert "intent_patterns" in config

    def test_has_relationship_templates(self):
        from auto_config import AutoConfigGenerator
        config = AutoConfigGenerator(self._make_schema()).generate()
        assert "relationship_templates" in config
        assert "TREATS" in config["relationship_templates"]
        assert "CAUSES" in config["relationship_templates"]

    def test_forward_intents_created(self):
        from auto_config import AutoConfigGenerator
        config = AutoConfigGenerator(self._make_schema()).generate()
        intents = config["intent_patterns"]
        assert "drug_treats" in intents
        assert "drug_causes" in intents

    def test_reverse_intents_created(self):
        from auto_config import AutoConfigGenerator
        config = AutoConfigGenerator(self._make_schema()).generate()
        intents = config["intent_patterns"]
        assert "disease_treats_reverse" in intents
        assert "sideeffect_causes_reverse" in intents

    def test_structural_intents_created(self):
        from auto_config import AutoConfigGenerator
        config = AutoConfigGenerator(self._make_schema()).generate()
        intents = config["intent_patterns"]
        assert "neighborhood" in intents
        assert "connection" in intents
        assert "shared" in intents

    def test_neighborhood_is_variable_hop(self):
        from auto_config import AutoConfigGenerator
        config = AutoConfigGenerator(self._make_schema()).generate()
        assert config["intent_patterns"]["neighborhood"]["strategy"] == "variable_hop"

    def test_connection_is_shortest_path(self):
        from auto_config import AutoConfigGenerator
        config = AutoConfigGenerator(self._make_schema()).generate()
        assert config["intent_patterns"]["connection"]["strategy"] == "shortest_path"

    def test_shared_is_shared_neighbor(self):
        from auto_config import AutoConfigGenerator
        config = AutoConfigGenerator(self._make_schema()).generate()
        assert config["intent_patterns"]["shared"]["strategy"] == "shared_neighbor"

    def test_chain_detection(self):
        """TREATS(Drug→Disease) + HAS_SYMPTOM(Disease→Symptom) → chained intent."""
        from auto_config import AutoConfigGenerator
        schema = self._make_schema()
        schema["node_labels"].append("Symptom")
        schema["relationships"].append(
            {"type": "HAS_SYMPTOM", "source_label": "Disease", "target_label": "Symptom", "count": 3}
        )
        config = AutoConfigGenerator(schema).generate()
        intents = config["intent_patterns"]
        chain_names = [k for k in intents if k.startswith("chain_")]
        assert len(chain_names) >= 1, f"Expected chain intents, got {list(intents.keys())}"

    def test_intent_ordering_chains_first(self):
        """Chained intents should come before targeted intents."""
        from auto_config import AutoConfigGenerator
        schema = self._make_schema()
        schema["node_labels"].append("Symptom")
        schema["relationships"].append(
            {"type": "HAS_SYMPTOM", "source_label": "Disease", "target_label": "Symptom", "count": 3}
        )
        config = AutoConfigGenerator(schema).generate()
        keys = list(config["intent_patterns"].keys())
        chain_idx = [i for i, k in enumerate(keys) if k.startswith("chain_")]
        targeted_idx = [i for i, k in enumerate(keys)
                        if config["intent_patterns"][k].get("strategy") == "targeted"]
        if chain_idx and targeted_idx:
            assert max(chain_idx) < min(targeted_idx), \
                f"Chains at {chain_idx} should precede targeted at {targeted_idx}"

    def test_auto_generated_flag(self):
        from auto_config import AutoConfigGenerator
        config = AutoConfigGenerator(self._make_schema()).generate()
        assert config["auto_generated"] is True

    def test_keywords_are_lowercase(self):
        from auto_config import AutoConfigGenerator
        config = AutoConfigGenerator(self._make_schema()).generate()
        for intent_name, pattern in config["intent_patterns"].items():
            for kw in pattern.get("keywords", []):
                assert kw == kw.lower(), f"Keyword '{kw}' in {intent_name} not lowercase"

    def test_template_has_source_target_placeholders(self):
        from auto_config import AutoConfigGenerator
        config = AutoConfigGenerator(self._make_schema()).generate()
        for rel_type, template in config["relationship_templates"].items():
            assert "{source}" in template, f"Template for {rel_type} missing {{source}}"
            assert "{target}" in template, f"Template for {rel_type} missing {{target}}"


# --- Updated components accept dict ---

class TestComponentsAcceptDict:
    """Verify IntentClassifier, TraversalEngine, ContextGenerator accept dicts."""

    def _make_config(self):
        return {
            "intent_patterns": {
                "test_intent": {
                    "strategy": "targeted",
                    "keywords": ["test keyword"],
                    "source_label": "A",
                    "relationship": "REL",
                    "target_label": "B",
                    "entry_anchor": "source",
                }
            },
            "relationship_templates": {
                "REL": "{source} relates to {target}"
            },
            "general_traversal": {"node_limit": 10},
        }

    def test_intent_classifier_from_dict(self):
        from intent_classifier import IntentClassifier
        clf = IntentClassifier(self._make_config())
        assert clf.classify("test keyword here") == "test_intent"
        assert clf.classify("no match") == "general"

    def test_traversal_engine_from_dict(self):
        from traversal_engine import SmartTraversalEngine
        connector = MagicMock()
        connector.execute_query.return_value = []
        engine = SmartTraversalEngine(connector, self._make_config())
        result = engine.traverse(
            [{"id": "1", "label": "A", "name": "test", "properties": {}}],
            "test_intent",
        )
        assert result["strategy"] == "targeted"

    def test_context_generator_from_dict(self):
        from context_generator import ContextGenerator
        gen = ContextGenerator(self._make_config())
        subgraph = {
            "nodes": [{"id": "1", "label": "A", "name": "Foo", "properties": {}}],
            "relationships": [],
            "strategy": "targeted",
            "hop_depth": 1,
        }
        text = gen.generate(subgraph)
        assert "Foo" in text


# --- Pipeline offline test ---

class TestPipelineOffline:
    """Test pipeline initialization with mocked connector."""

    def test_pipeline_query_returns_dict_keys(self):
        """Verify the query result dict has all expected keys."""
        from pipeline import GraphRAGPipeline
        connector = MagicMock()
        connector.check_connection = MagicMock()

        # Mock schema discovery responses
        def fake_execute(query, params=None):
            q = query.strip()
            if "db.labels" in q:
                return [{"label": "Drug"}, {"label": "Disease"}]
            if "count(n)" in q and "DISTINCT" not in q:
                return [{"cnt": 3}]
            if "type(r)" in q:
                return [{"rel_type": "TREATS", "source_label": "Drug",
                         "target_label": "Disease", "cnt": 5}]
            if "keys(n)" in q:
                return [{"key": "name"}]
            if "count(DISTINCT" in q or "distinct_count" in q:
                return [{"total": 3, "distinct_count": 3, "samples": ["Aspirin"]}]
            if "toLower(n." in q:
                return [{"id": "1", "label": "Drug",
                         "properties": {"name": "Aspirin"}}]
            return []

        connector.execute_query = MagicMock(side_effect=fake_execute)

        pipeline = GraphRAGPipeline(
            connector=connector, enable_llm=False, verbose=False,
        )
        result = pipeline.query("What does aspirin treat?")

        expected_keys = {
            "query", "entry_nodes", "intent", "strategy",
            "hop_depth", "subgraph", "context", "answer", "timing",
        }
        assert expected_keys.issubset(result.keys())


# ═══════════════════════════════════════════════════════════════════
#  Part B: LIVE INTEGRATION TESTS (need Neo4j Aura)
# ═══════════════════════════════════════════════════════════════════

class TestLiveSchemaDiscovery:
    """Live tests against real Neo4j — run with: pytest test_auto.py -k live"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Seed graph and create discovery instance."""
        from connector import GraphDBConnector
        try:
            self.connector = GraphDBConnector()
            self.connector.check_connection()
        except Exception:
            pytest.skip("Neo4j not available")

        # Ensure graph has data
        count = self.connector.execute_query(
            "MATCH (n) RETURN count(n) AS cnt"
        )[0]["cnt"]
        if count == 0:
            pytest.skip("Graph is empty — run 'python main.py --seed' first")

        from schema_discovery import SchemaDiscovery
        self.discovery = SchemaDiscovery(self.connector)
        yield
        self.connector.close()

    def test_live_discover_labels(self):
        labels = self.discovery._discover_labels()
        assert len(labels) > 0
        assert all(isinstance(l, str) for l in labels)

    def test_live_discover_relationships(self):
        rels = self.discovery._discover_relationships()
        assert len(rels) > 0
        for r in rels:
            assert "type" in r
            assert "source_label" in r
            assert "count" in r

    def test_live_full_discovery(self):
        schema = self.discovery.discover()
        assert schema["total_nodes"] > 0
        assert schema["total_relationships"] > 0
        assert len(schema["searchable_properties"]) > 0

    def test_live_searchable_properties_found(self):
        schema = self.discovery.discover()
        # At least one label should have searchable properties
        has_searchable = any(
            len(props) > 0
            for props in schema["searchable_properties"].values()
        )
        assert has_searchable


class TestLiveAutoConfig:
    """Live auto config generation test."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from connector import GraphDBConnector
        try:
            self.connector = GraphDBConnector()
            self.connector.check_connection()
        except Exception:
            pytest.skip("Neo4j not available")

        count = self.connector.execute_query(
            "MATCH (n) RETURN count(n) AS cnt"
        )[0]["cnt"]
        if count == 0:
            pytest.skip("Graph is empty")

        from schema_discovery import SchemaDiscovery
        self.schema = SchemaDiscovery(self.connector).discover()
        yield
        self.connector.close()

    def test_live_auto_config_generates(self):
        from auto_config import AutoConfigGenerator
        config = AutoConfigGenerator(self.schema).generate()
        assert "intent_patterns" in config
        assert len(config["intent_patterns"]) > 0
        assert "relationship_templates" in config

    def test_live_config_works_with_classifier(self):
        from auto_config import AutoConfigGenerator
        from intent_classifier import IntentClassifier
        config = AutoConfigGenerator(self.schema).generate()
        clf = IntentClassifier(config)
        intents = clf.all_intents()
        assert len(intents) > 0

    def test_live_config_works_with_traversal(self):
        from auto_config import AutoConfigGenerator
        from traversal_engine import SmartTraversalEngine
        config = AutoConfigGenerator(self.schema).generate()
        engine = SmartTraversalEngine(self.connector, config)
        # Traverse with empty nodes should return empty subgraph
        result = engine.traverse([], "neighborhood")
        assert "nodes" in result


class TestLivePipeline:
    """Full end-to-end live pipeline test."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from connector import GraphDBConnector
        try:
            self.connector = GraphDBConnector()
            self.connector.check_connection()
        except Exception:
            pytest.skip("Neo4j not available")

        count = self.connector.execute_query(
            "MATCH (n) RETURN count(n) AS cnt"
        )[0]["cnt"]
        if count == 0:
            pytest.skip("Graph is empty")
        yield
        self.connector.close()

    def test_live_pipeline_query(self):
        from pipeline import GraphRAGPipeline
        pipeline = GraphRAGPipeline(
            connector=self.connector, enable_llm=False, verbose=False,
        )
        result = pipeline.query("What are the side effects of aspirin?")
        assert result["entry_nodes"]  # found aspirin
        assert result["intent"] != "none"
        assert len(result["subgraph"]["nodes"]) > 0

    def test_live_pipeline_no_match_query(self):
        from pipeline import GraphRAGPipeline
        pipeline = GraphRAGPipeline(
            connector=self.connector, enable_llm=False, verbose=False,
        )
        result = pipeline.query("xyzzy nonexistent entity")
        assert result["entry_nodes"] == []
        assert result["intent"] == "none"


# ═══════════════════════════════════════════════════════════════════
#  Script runner (live demo with pretty output)
# ═══════════════════════════════════════════════════════════════════

def run_live_demo():
    """Full live demo: discover → configure → query."""
    import os
    from connector import GraphDBConnector
    from schema_discovery import SchemaDiscovery
    from auto_config import AutoConfigGenerator
    from pipeline import GraphRAGPipeline

    print("=" * 60)
    print("  AUTOMATIC GRAPH-RAG — LIVE INTEGRATION TEST")
    print("=" * 60)

    connector = GraphDBConnector()
    connector.check_connection()

    # Check if graph has data
    count = connector.execute_query("MATCH (n) RETURN count(n) AS cnt")[0]["cnt"]
    if count == 0:
        print("\n  Graph is empty. Seeding sample data …")
        from main import seed_sample_graph
        seed_sample_graph(connector)

    # ── Phase 1: Schema Discovery ─────────────────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 1: SCHEMA DISCOVERY")
    print("=" * 60)
    discovery = SchemaDiscovery(connector)
    schema = discovery.discover()
    discovery.print_schema(schema)

    # ── Phase 2: Auto Config Generation ───────────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 2: AUTO CONFIG GENERATION")
    print("=" * 60)
    gen = AutoConfigGenerator(schema)
    config = gen.generate()
    gen.print_config(config)

    # ── Phase 3: Pipeline Queries ─────────────────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 3: PIPELINE QUERIES")
    print("=" * 60)

    pipeline = GraphRAGPipeline(
        connector=connector,
        config=config,
        enable_llm=True,
        verbose=False,
    )

    test_queries = [
        "What are the side effects of aspirin?",
        "What treats headaches?",
        "Give me an overview of metformin",
        "How is aspirin connected to peptic ulcer?",
        "What do aspirin and ibuprofen have in common?",
    ]

    passed = 0
    for q in test_queries:
        result = pipeline.query(q)
        nodes = result["entry_nodes"]
        intent = result["intent"]
        strategy = result["strategy"]
        n_nodes = len(result["subgraph"].get("nodes", []))

        status = "PASS" if nodes and n_nodes > 0 else "FAIL"
        if status == "PASS":
            passed += 1

        print(f"\n  [{status}] {q}")
        print(f"    Nodes: {nodes}, Intent: {intent}, Strategy: {strategy}")
        print(f"    Subgraph: {n_nodes} nodes")
        if result["answer"]:
            answer_short = result["answer"][:120] + "…" if len(result["answer"] or "") > 120 else result["answer"]
            print(f"    Answer: {answer_short}")

    print(f"\n  Results: {passed}/{len(test_queries)} passed")
    print("=" * 60)

    connector.close()


if __name__ == "__main__":
    run_live_demo()
