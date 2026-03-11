"""
tests/test_entity_resolver.py
-------------------------------
Offline tests for src/kg_construction/entity_resolver.py
"""

import pytest
from src.kg_construction.triple_extractor import Triple
from src.kg_construction.entity_resolver import (
    EntityResolver,
    ResolutionResult,
    _normalise_key,
    _string_similarity,
)


# ──────────────────────────────────────────────────────────────
# Normalisation key tests
# ──────────────────────────────────────────────────────────────

class TestNormaliseKey:
    def test_lowercase(self):
        assert _normalise_key("ASPIRIN") == "aspirin"

    def test_remove_articles(self):
        assert _normalise_key("the flu") == "flu"
        assert _normalise_key("a Doctor") == "doctor"
        assert _normalise_key("an Apple") == "apple"

    def test_remove_special_chars(self):
        assert _normalise_key("high-blood pressure") == "highblood pressure"

    def test_collapse_whitespace(self):
        assert _normalise_key("  foo   bar  ") == "foo bar"


# ──────────────────────────────────────────────────────────────
# String similarity tests
# ──────────────────────────────────────────────────────────────

class TestStringSimilarity:
    def test_identical(self):
        assert _string_similarity("hello", "hello") == 1.0

    def test_completely_different(self):
        sim = _string_similarity("abc", "xyz")
        assert sim < 0.3

    def test_similar(self):
        sim = _string_similarity("aspirin", "Aspirin")
        assert sim > 0.9

    def test_empty_strings(self):
        # Both empty → equal
        sim = _string_similarity("", "")
        assert sim == 1.0


# ──────────────────────────────────────────────────────────────
# Entity resolution tests
# ──────────────────────────────────────────────────────────────

class TestEntityResolver:

    def test_exact_duplicates(self):
        triples = [
            Triple(head="Aspirin", relation="TREATS", tail="Headache"),
            Triple(head="Aspirin", relation="TREATS", tail="Headache"),
        ]
        resolver = EntityResolver()
        result = resolver.resolve(triples)
        assert result.resolved_count == 1

    def test_case_normalisation(self):
        triples = [
            Triple(head="aspirin", relation="TREATS", tail="headache"),
            Triple(head="ASPIRIN", relation="TREATS", tail="HEADACHE"),
            Triple(head="Aspirin", relation="TREATS", tail="Headache"),
        ]
        resolver = EntityResolver()
        result = resolver.resolve(triples)
        # All three refer to the same entities
        assert result.resolved_count == 1

    def test_article_removal(self):
        triples = [
            Triple(head="the flu", relation="AFFECTS", tail="patient"),
            Triple(head="flu", relation="AFFECTS", tail="patient"),
        ]
        resolver = EntityResolver()
        result = resolver.resolve(triples)
        assert result.resolved_count == 1

    def test_canonical_map(self):
        triples = [
            Triple(head="aspirin", relation="R", tail="B"),
            Triple(head="ASPIRIN", relation="R", tail="B"),
            Triple(head="Aspirin", relation="R", tail="B"),
        ]
        resolver = EntityResolver()
        result = resolver.resolve(triples)
        # Canonical should be Title Case
        canonical = result.canonical_map.get("aspirin")
        assert canonical is not None
        assert canonical == canonical.title()  # Title case

    def test_different_entities_not_merged(self):
        triples = [
            Triple(head="Aspirin", relation="TREATS", tail="Headache"),
            Triple(head="Ibuprofen", relation="TREATS", tail="Fever"),
        ]
        resolver = EntityResolver()
        result = resolver.resolve(triples)
        assert result.resolved_count == 2
        assert result.entities_after >= 4  # 4 distinct entities

    def test_entity_counts(self):
        triples = [
            Triple(head="A", relation="R1", tail="B"),
            Triple(head="A", relation="R2", tail="C"),
            Triple(head="B", relation="R3", tail="C"),
        ]
        resolver = EntityResolver()
        result = resolver.resolve(triples)
        assert result.original_count == 3
        assert result.entities_before == 3  # A, B, C
        assert result.entities_after == 3

    def test_empty_triples(self):
        resolver = EntityResolver()
        result = resolver.resolve([])
        assert result.resolved_count == 0
        assert result.entities_before == 0


class TestResolutionResult:
    def test_defaults(self):
        r = ResolutionResult()
        assert r.resolved_triples == []
        assert r.canonical_map == {}
        assert r.merge_groups == {}
