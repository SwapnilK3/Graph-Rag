"""
tests/test_schema_inducer.py
------------------------------
Offline tests for src/kg_construction/schema_inducer.py

Tests schema induction logic with mock LLM and heuristic fallback.
"""

import pytest
from src.kg_construction.triple_extractor import Triple
from src.kg_construction.schema_inducer import (
    SchemaInducer,
    InducedSchema,
    EntityType,
    RelationType,
)


# ──────────────────────────────────────────────────────────────
# Mock LLM
# ──────────────────────────────────────────────────────────────

class MockLLMClient:
    def __init__(self, response=None):
        self.response = response or {
            "types": [
                {
                    "name": "Drug",
                    "description": "Pharmaceutical substances",
                    "entities": ["Aspirin", "Ibuprofen"],
                    "is_event_type": False,
                    "parent_type": None,
                },
                {
                    "name": "Disease",
                    "description": "Medical conditions",
                    "entities": ["Headache", "Fever"],
                    "is_event_type": False,
                    "parent_type": None,
                },
            ],
            "relation_types": [
                {
                    "name": "TREATS",
                    "description": "Drug treats a disease",
                    "source_types": ["Drug"],
                    "target_types": ["Disease"],
                },
            ],
        }

    def generate_json(self, prompt, system=None):
        return self.response

    def generate(self, prompt, system=None):
        return str(self.response)


# ──────────────────────────────────────────────────────────────
# Test data
# ──────────────────────────────────────────────────────────────

def _sample_triples():
    return [
        Triple(head="Aspirin", relation="TREATS", tail="Headache",
               head_type="entity", tail_type="entity"),
        Triple(head="Ibuprofen", relation="TREATS", tail="Fever",
               head_type="entity", tail_type="entity"),
        Triple(head="Aspirin", relation="HAS_SIDE_EFFECT", tail="Nausea",
               head_type="entity", tail_type="entity"),
    ]


# ──────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────

class TestSchemaInducer:

    def test_induce_with_llm(self):
        llm = MockLLMClient()
        inducer = SchemaInducer(llm=llm)
        schema = inducer.induce(_sample_triples())

        assert isinstance(schema, InducedSchema)
        assert "Drug" in schema.entity_types
        assert "Disease" in schema.entity_types
        assert "TREATS" in schema.relation_types

    def test_induce_empty_triples(self):
        llm = MockLLMClient()
        inducer = SchemaInducer(llm=llm)
        schema = inducer.induce([])
        assert len(schema.entity_types) == 0
        assert len(schema.relation_types) == 0

    def test_entity_type_structure(self):
        llm = MockLLMClient()
        inducer = SchemaInducer(llm=llm)
        schema = inducer.induce(_sample_triples())

        drug = schema.entity_types.get("Drug")
        assert drug is not None
        assert "Aspirin" in drug.entities
        assert "Ibuprofen" in drug.entities
        assert drug.is_event_type is False

    def test_relation_constraints_rebuilt(self):
        """After induction, relation constraints should reflect actual triples."""
        llm = MockLLMClient()
        inducer = SchemaInducer(llm=llm)
        schema = inducer.induce(_sample_triples())

        treats = schema.relation_types.get("TREATS")
        assert treats is not None
        assert treats.count >= 2

    def test_heuristic_fallback(self):
        """When LLM fails, heuristic induction should still produce types."""
        class FailingLLM:
            def generate_json(self, prompt, system=None):
                raise RuntimeError("LLM unavailable")
            def generate(self, prompt, system=None):
                raise RuntimeError("LLM unavailable")

        inducer = SchemaInducer(llm=FailingLLM())
        schema = inducer.induce(_sample_triples())

        # Should have some types from heuristic grouping
        assert len(schema.entity_types) > 0
        # Should have relation constraints
        assert len(schema.relation_types) > 0


class TestJaccardMerging:
    """Test PG-HIVE Algorithm 2: merging similar types."""

    def test_merge_identical_entity_sets(self):
        llm = MockLLMClient(response={
            "types": [
                {"name": "TypeA", "entities": ["X", "Y", "Z"], "is_event_type": False},
                {"name": "TypeB", "entities": ["X", "Y", "Z"], "is_event_type": False},
            ],
            "relation_types": [],
        })
        inducer = SchemaInducer(llm=llm, merge_threshold=0.7)
        schema = inducer.induce([
            Triple(head="X", relation="R", tail="Y"),
            Triple(head="Y", relation="R", tail="Z"),
        ])
        # TypeA and TypeB should be merged (Jaccard = 1.0)
        assert len(schema.entity_types) == 1

    def test_no_merge_below_threshold(self):
        llm = MockLLMClient(response={
            "types": [
                {"name": "TypeA", "entities": ["X", "Y"], "is_event_type": False},
                {"name": "TypeB", "entities": ["Z", "W"], "is_event_type": False},
            ],
            "relation_types": [],
        })
        inducer = SchemaInducer(llm=llm, merge_threshold=0.7)
        schema = inducer.induce([
            Triple(head="X", relation="R", tail="Y"),
            Triple(head="Z", relation="R", tail="W"),
        ])
        # Disjoint entity sets → no merge
        assert len(schema.entity_types) == 2


class TestEntityType:
    def test_defaults(self):
        et = EntityType(name="Test")
        assert et.entities == []
        assert et.is_event_type is False
        assert et.parent_type is None

    def test_event_type(self):
        et = EntityType(name="Action", is_event_type=True)
        assert et.is_event_type is True


class TestRelationType:
    def test_defaults(self):
        rt = RelationType(name="REL")
        assert rt.source_types == []
        assert rt.target_types == []
        assert rt.count == 0
