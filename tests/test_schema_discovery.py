"""
tests/test_schema_discovery.py
---------------------------------
Offline tests for src/schema_discovery/ components:
- PropertyAnalyzer  (property_analyzer.py)
- CardinalityAnalyzer (cardinality.py)
- HierarchyInferer (hierarchy.py)
- SearchableScorer (searchable_scorer.py)
- SchemaDiscoverer (discoverer.py)

Uses a mock Neo4j connector so tests run offline.
"""

import pytest
from src.schema_discovery.property_analyzer import (
    PropertyAnalyzer,
    PropertyInfo,
    TypeProperties,
)
from src.schema_discovery.cardinality import (
    CardinalityAnalyzer,
    CardinalityInfo,
)
from src.schema_discovery.hierarchy import (
    HierarchyInferer,
    TypeNode,
    TypeHierarchy,
)
from src.schema_discovery.searchable_scorer import (
    SearchableScorer,
    SearchableScore,
)
from src.schema_discovery.discoverer import (
    SchemaDiscoverer,
    DiscoveredSchema,
    NodeTypeSchema,
    EdgeTypeSchema,
)


# ══════════════════════════════════════════════════════════════
# Mock Neo4j Connector
# ══════════════════════════════════════════════════════════════

class MockConnector:
    """
    Mock Neo4jConnector that returns pre-configured query results.
    Register expected results with .register(query_pattern, result).
    Patterns are matched in reverse order (last registered = highest priority).
    """

    def __init__(self):
        self._results: list[tuple[str, list[dict] | dict]] = []

    def register(self, pattern: str, result):
        """Register a query pattern (substring match) and its result."""
        self._results.append((pattern, result))

    def _match(self, cypher: str) -> list[dict]:
        # Search in reverse so later registrations override earlier ones
        for pattern, result in reversed(self._results):
            if pattern in cypher:
                return result if isinstance(result, list) else [result]
        return []

    def run(self, cypher: str, params=None) -> list[dict]:
        return self._match(cypher)

    def run_single(self, cypher: str, params=None) -> dict | None:
        rows = self.run(cypher, params)
        return rows[0] if rows else None

    def write(self, cypher: str, params=None) -> list[dict]:
        return self.run(cypher, params)

    def run_batch(self, cypher, batch):
        pass

    def clear_database(self):
        pass


# ══════════════════════════════════════════════════════════════
# PropertyAnalyzer tests
# ══════════════════════════════════════════════════════════════

class TestPropertyAnalyzer:

    def test_infer_type_integer(self):
        analyzer = PropertyAnalyzer(MockConnector())
        assert analyzer._infer_type([1, 2, 3, 4]) == "INTEGER"

    def test_infer_type_float(self):
        analyzer = PropertyAnalyzer(MockConnector())
        assert analyzer._infer_type([1.5, 2.3, 3.7]) == "FLOAT"

    def test_infer_type_boolean(self):
        analyzer = PropertyAnalyzer(MockConnector())
        assert analyzer._infer_type([True, False, True]) == "BOOLEAN"

    def test_infer_type_string(self):
        analyzer = PropertyAnalyzer(MockConnector())
        assert analyzer._infer_type(["hello", "world"]) == "STRING"

    def test_infer_type_date(self):
        analyzer = PropertyAnalyzer(MockConnector())
        assert analyzer._infer_type(["2024-01-15", "2024-02-20"]) == "DATE"

    def test_infer_type_list(self):
        analyzer = PropertyAnalyzer(MockConnector())
        assert analyzer._infer_type([[1, 2], [3, 4]]) == "LIST"

    def test_infer_type_empty(self):
        analyzer = PropertyAnalyzer(MockConnector())
        assert analyzer._infer_type([]) == "STRING"

    def test_infer_type_mixed_numeric(self):
        analyzer = PropertyAnalyzer(MockConnector())
        # Mix of int and float → FLOAT
        assert analyzer._infer_type([1, 2.5, 3, 4.1]) == "FLOAT"

    def test_classify_value_bool_string(self):
        analyzer = PropertyAnalyzer(MockConnector())
        assert analyzer._classify_value("true") == "BOOLEAN"
        assert analyzer._classify_value("false") == "BOOLEAN"

    def test_classify_value_integer_string(self):
        analyzer = PropertyAnalyzer(MockConnector())
        assert analyzer._classify_value("42") == "INTEGER"

    def test_classify_value_float_string(self):
        analyzer = PropertyAnalyzer(MockConnector())
        assert analyzer._classify_value("3.14") == "FLOAT"

    def test_analyze_node_properties(self):
        conn = MockConnector()
        conn.register("count(n)", {"cnt": 10})
        conn.register("UNWIND keys(n) AS k", [{"k": "name"}, {"k": "age"}])
        conn.register("RETURN count(n) AS cnt", {"cnt": 10})
        conn.register("count(DISTINCT", {"cnt": 10})
        conn.register("AS val LIMIT", [{"val": "Alice"}, {"val": "Bob"}])

        analyzer = PropertyAnalyzer(conn, sample_size=10)
        result = analyzer.analyze_node_properties("Person")
        assert result.label == "Person"
        assert result.instance_count == 10


class TestPropertyInfo:
    def test_defaults(self):
        info = PropertyInfo(name="test")
        assert info.data_type == "STRING"
        assert info.constraint == "OPTIONAL"
        assert info.frequency == 0.0


# ══════════════════════════════════════════════════════════════
# CardinalityAnalyzer tests
# ══════════════════════════════════════════════════════════════

class TestCardinalityClassification:

    def test_one_to_one(self):
        assert CardinalityAnalyzer._classify(1, 1) == "1:1"

    def test_one_to_many(self):
        assert CardinalityAnalyzer._classify(5, 1) == "1:N"

    def test_many_to_one(self):
        assert CardinalityAnalyzer._classify(1, 5) == "N:1"

    def test_many_to_many(self):
        assert CardinalityAnalyzer._classify(3, 4) == "M:N"


class TestCardinalityInfo:
    def test_notation(self):
        info = CardinalityInfo(
            rel_type="R", source_label="A", target_label="B",
            max_out=5, max_in=1, min_out=0, min_in=0,
        )
        assert "[0..5]" in info.notation
        assert "[0..1]" in info.notation


# ══════════════════════════════════════════════════════════════
# HierarchyInferer tests
# ══════════════════════════════════════════════════════════════

class TestTypeHierarchy:
    def test_get_subtypes(self):
        h = TypeHierarchy()
        h.types["Entity"] = TypeNode(label="Entity", subtypes=["Person", "Drug"])
        h.types["Person"] = TypeNode(label="Person", supertypes=["Entity"], subtypes=["Doctor"])
        h.types["Drug"] = TypeNode(label="Drug", supertypes=["Entity"])
        h.types["Doctor"] = TypeNode(label="Doctor", supertypes=["Person"])

        direct = h.get_subtypes("Entity")
        assert "Person" in direct
        assert "Drug" in direct
        assert "Doctor" not in direct

    def test_get_subtypes_recursive(self):
        h = TypeHierarchy()
        h.types["Entity"] = TypeNode(label="Entity", subtypes=["Person"])
        h.types["Person"] = TypeNode(label="Person", supertypes=["Entity"], subtypes=["Doctor"])
        h.types["Doctor"] = TypeNode(label="Doctor", supertypes=["Person"])

        recursive = h.get_subtypes("Entity", recursive=True)
        assert "Person" in recursive
        assert "Doctor" in recursive

    def test_get_supertypes(self):
        h = TypeHierarchy()
        h.types["Entity"] = TypeNode(label="Entity", subtypes=["Person"])
        h.types["Person"] = TypeNode(label="Person", supertypes=["Entity"])

        assert h.get_supertypes("Person") == ["Entity"]

    def test_get_supertypes_recursive(self):
        h = TypeHierarchy()
        h.types["Entity"] = TypeNode(label="Entity", subtypes=["Person"])
        h.types["Person"] = TypeNode(label="Person", supertypes=["Entity"], subtypes=["Doctor"])
        h.types["Doctor"] = TypeNode(label="Doctor", supertypes=["Person"])

        recursive = h.get_supertypes("Doctor", recursive=True)
        assert "Person" in recursive
        assert "Entity" in recursive

    def test_depth(self):
        h = TypeHierarchy()
        h.types["Root"] = TypeNode(label="Root", depth=0)
        h.types["Child"] = TypeNode(label="Child", depth=1)
        assert h.get_depth("Root") == 0
        assert h.get_depth("Child") == 1
        assert h.get_depth("Unknown") == -1


# ══════════════════════════════════════════════════════════════
# SearchableScorer tests
# ══════════════════════════════════════════════════════════════

class TestSearchableScorer:

    def test_name_property_scores_high(self):
        scorer = SearchableScorer()
        prop = PropertyInfo(name="name", data_type="STRING", unique_ratio=0.95)
        score = scorer.score_property("Person", prop)
        assert score.is_searchable
        assert score.total_score >= 0.7

    def test_name_pattern_match(self):
        scorer = SearchableScorer()
        assert scorer._name_pattern_score("name") > 0
        assert scorer._name_pattern_score("title") > 0
        assert scorer._name_pattern_score("label") > 0
        assert scorer._name_pattern_score("description") > 0
        assert scorer._name_pattern_score("email") > 0

    def test_name_pattern_no_match(self):
        scorer = SearchableScorer()
        assert scorer._name_pattern_score("age") == 0
        assert scorer._name_pattern_score("created_at") == 0

    def test_cardinality_score_high_unique(self):
        scorer = SearchableScorer()
        assert scorer._cardinality_score(0.9) > 0
        assert scorer._cardinality_score(1.0) > 0

    def test_cardinality_score_low_unique(self):
        scorer = SearchableScorer()
        assert scorer._cardinality_score(0.1) == 0
        assert scorer._cardinality_score(0.3) == 0

    def test_string_type_score(self):
        scorer = SearchableScorer()
        assert scorer._string_type_score("STRING") > 0
        assert scorer._string_type_score("INTEGER") == 0

    def test_non_searchable_property(self):
        scorer = SearchableScorer()
        prop = PropertyInfo(name="count", data_type="INTEGER", unique_ratio=0.1)
        score = scorer.score_property("Stats", prop)
        assert not score.is_searchable

    def test_get_searchable_properties(self):
        scorer = SearchableScorer()
        props = {
            "name": PropertyInfo(name="name", data_type="STRING", unique_ratio=0.95),
            "age": PropertyInfo(name="age", data_type="INTEGER", unique_ratio=0.3),
            "title": PropertyInfo(name="title", data_type="STRING", unique_ratio=0.85),
        }
        searchable = scorer.get_searchable_properties("Person", props)
        assert "name" in searchable
        assert "title" in searchable
        assert "age" not in searchable


# ══════════════════════════════════════════════════════════════
# DiscoveredSchema serialization tests
# ══════════════════════════════════════════════════════════════

class TestDiscoveredSchema:

    def test_to_dict(self):
        schema = DiscoveredSchema()
        schema.node_types["Drug"] = NodeTypeSchema(
            label="Drug", instance_count=5,
            properties={"name": {"data_type": "STRING"}},
            searchable_properties=["name"],
        )
        schema.edge_types["TREATS"] = EdgeTypeSchema(
            rel_type="TREATS",
            source_labels=["Drug"],
            target_labels=["Disease"],
            cardinality="M:N",
        )
        d = schema.to_dict()
        assert "Drug" in d["node_types"]
        assert "TREATS" in d["edge_types"]
        assert d["node_types"]["Drug"]["instance_count"] == 5

    def test_to_json(self):
        schema = DiscoveredSchema()
        schema.node_types["Test"] = NodeTypeSchema(label="Test")
        json_str = schema.to_json()
        assert '"Test"' in json_str

    def test_empty_schema(self):
        schema = DiscoveredSchema()
        d = schema.to_dict()
        assert d["node_types"] == {}
        assert d["edge_types"] == {}
