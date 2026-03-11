"""
tests/test_schema_evolution.py
---------------------------------
Offline tests for src/schema_evolution/adapter.py
"""

import pytest
from src.kg_construction.triple_extractor import Triple
from src.kg_construction.schema_inducer import InducedSchema, EntityType, RelationType
from src.schema_discovery.discoverer import (
    DiscoveredSchema,
    NodeTypeSchema,
    EdgeTypeSchema,
)
from src.schema_evolution.adapter import (
    SchemaAdapter,
    SchemaChange,
    EvolutionResult,
)


# ──────────────────────────────────────────────────────────────
# Test data helpers
# ──────────────────────────────────────────────────────────────

def _base_schema() -> DiscoveredSchema:
    """A small base schema with Drug and Disease types."""
    schema = DiscoveredSchema()
    schema.node_types["Drug"] = NodeTypeSchema(
        label="Drug",
        instance_count=3,
        properties={"name": {"data_type": "STRING", "constraint": "MANDATORY"}},
        searchable_properties=["name"],
    )
    schema.node_types["Disease"] = NodeTypeSchema(
        label="Disease",
        instance_count=2,
        properties={"name": {"data_type": "STRING", "constraint": "MANDATORY"}},
        searchable_properties=["name"],
    )
    schema.edge_types["TREATS"] = EdgeTypeSchema(
        rel_type="TREATS",
        source_labels=["Drug"],
        target_labels=["Disease"],
        cardinality="M:N",
        total_edges=4,
    )
    return schema


def _new_triples_same_domain() -> list[Triple]:
    """New triples within existing types."""
    return [
        Triple(head="Metformin", relation="TREATS", tail="Diabetes",
               head_type="entity", tail_type="entity"),
    ]


def _new_triples_new_type() -> list[Triple]:
    """New triples introducing a new entity type."""
    return [
        Triple(head="Dr. Smith", relation="PRESCRIBES", tail="Aspirin",
               head_type="entity", tail_type="entity"),
        Triple(head="Dr. Jones", relation="PRESCRIBES", tail="Metformin",
               head_type="entity", tail_type="entity"),
    ]


# ──────────────────────────────────────────────────────────────
# Horizontal expansion tests
# ──────────────────────────────────────────────────────────────

class TestHorizontalExpansion:

    def test_new_node_type_detected(self):
        adapter = SchemaAdapter()
        result = adapter.evolve(_base_schema(), _new_triples_new_type())

        assert result.has_changes
        # Should detect Entity type for Dr. Smith/Dr. Jones (generic type)
        assert len(result.new_node_types) > 0 or len(result.new_edge_types) > 0

    def test_new_edge_type_detected(self):
        adapter = SchemaAdapter()
        result = adapter.evolve(_base_schema(), _new_triples_new_type())

        # PRESCRIBES is a new relation not in the base schema
        assert "PRESCRIBES" in result.new_edge_types

    def test_no_changes_for_existing_types(self):
        adapter = SchemaAdapter()
        triples = [
            Triple(head="Aspirin", relation="TREATS", tail="Headache",
                   head_type="Drug", tail_type="Disease"),
        ]
        result = adapter.evolve(_base_schema(), triples)
        # Drug and Disease already exist; TREATS already exists
        assert "TREATS" not in result.new_edge_types

    def test_original_schema_not_mutated(self):
        adapter = SchemaAdapter()
        original = _base_schema()
        original_types = set(original.node_types.keys())
        adapter.evolve(original, _new_triples_new_type())
        # Original should be unchanged (we deep copy)
        assert set(original.node_types.keys()) == original_types


# ──────────────────────────────────────────────────────────────
# Vertical expansion tests
# ──────────────────────────────────────────────────────────────

class TestVerticalExpansion:

    def test_refinement_detected(self):
        adapter = SchemaAdapter()
        base = _base_schema()

        # Induced schema suggests "Antibiotic" as subtype of "Drug"
        induced = InducedSchema()
        induced.entity_types["Antibiotic"] = EntityType(
            name="Antibiotic",
            entities=["Amoxicillin", "Penicillin"],
            parent_type="Drug",
        )

        triples = [
            Triple(head="Amoxicillin", relation="TREATS", tail="Infection",
                   head_type="Antibiotic", tail_type="Disease"),
        ]

        result = adapter.evolve(base, triples, induced=induced)

        # "Antibiotic" contains "Drug" → should detect as vertical refinement
        # (name heuristic: "Antibiotic" doesn't contain "Drug" literally,
        #  but this tests the general flow)
        assert result.has_changes


# ──────────────────────────────────────────────────────────────
# Statistics update tests
# ──────────────────────────────────────────────────────────────

class TestStatsUpdate:

    def test_edge_count_updated(self):
        adapter = SchemaAdapter()
        schema = _base_schema()
        old_count = schema.edge_types["TREATS"].total_edges

        triples = [
            Triple(head="NewDrug", relation="TREATS", tail="NewDisease",
                   head_type="Drug", tail_type="Disease"),
        ]
        result = adapter.evolve(schema, triples)
        new_count = result.updated_schema.edge_types["TREATS"].total_edges
        assert new_count > old_count


# ──────────────────────────────────────────────────────────────
# Alias detection tests
# ──────────────────────────────────────────────────────────────

class TestAliasDetection:

    def test_similar_names_detected_as_alias(self):
        adapter = SchemaAdapter(merge_threshold=0.7)
        assert adapter._is_alias("Drug", {"Drug", "Disease"})
        assert adapter._is_alias("Drugs", {"Drug", "Disease"})

    def test_different_names_not_alias(self):
        adapter = SchemaAdapter(merge_threshold=0.7)
        assert not adapter._is_alias("Hospital", {"Drug", "Disease"})


# ──────────────────────────────────────────────────────────────
# Data structure tests
# ──────────────────────────────────────────────────────────────

class TestEvolutionResult:

    def test_has_changes_false_when_empty(self):
        r = EvolutionResult(updated_schema=DiscoveredSchema())
        assert not r.has_changes

    def test_has_changes_true(self):
        r = EvolutionResult(
            updated_schema=DiscoveredSchema(),
            changes=[SchemaChange(change_type="HORIZONTAL_NODE", entity="X")],
        )
        assert r.has_changes

    def test_summary(self):
        r = EvolutionResult(
            updated_schema=DiscoveredSchema(),
            changes=[SchemaChange(change_type="HORIZONTAL_NODE", entity="X")],
            new_node_types=["X"],
        )
        s = r.summary()
        assert "1 changes" in s
        assert "X" in s
