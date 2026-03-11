"""
tests/test_triple_extractor.py
-------------------------------
Offline tests for src/kg_construction/triple_extractor.py

Tests normalisation and data structures WITHOUT calling the LLM.
"""

import pytest
from src.kg_construction.triple_extractor import (
    Triple,
    ExtractionResult,
    _normalise_entity,
    _normalise_relation,
)


# ──────────────────────────────────────────────────────────────
# Normalisation
# ──────────────────────────────────────────────────────────────

class TestNormaliseEntity:
    def test_basic_title_case(self):
        assert _normalise_entity("aspirin") == "Aspirin"

    def test_multi_word(self):
        assert _normalise_entity("heart attack") == "Heart Attack"

    def test_extra_whitespace(self):
        assert _normalise_entity("  high  blood  pressure  ") == "High Blood Pressure"

    def test_empty(self):
        assert _normalise_entity("") == ""

    def test_already_title_case(self):
        assert _normalise_entity("New York") == "New York"


class TestNormaliseRelation:
    def test_basic(self):
        assert _normalise_relation("treats") == "TREATS"

    def test_multi_word(self):
        assert _normalise_relation("treats disease") == "TREATS_DISEASE"

    def test_extra_whitespace(self):
        assert _normalise_relation("  is   located in  ") == "IS_LOCATED_IN"

    def test_special_chars_removed(self):
        result = _normalise_relation("treats/cures")
        assert result == "TREATSCURES"

    def test_empty(self):
        assert _normalise_relation("") == ""


# ──────────────────────────────────────────────────────────────
# Triple data class
# ──────────────────────────────────────────────────────────────

class TestTriple:
    def test_creation(self):
        t = Triple(head="Aspirin", relation="TREATS", tail="Headache")
        assert t.head == "Aspirin"
        assert t.relation == "TREATS"
        assert t.tail == "Headache"

    def test_defaults(self):
        t = Triple(head="A", relation="R", tail="B")
        assert t.head_type == ""
        assert t.tail_type == ""
        assert t.confidence == 1.0
        assert t.source_chunk == ""

    def test_normalised(self):
        t = Triple(
            head="aspirin",
            relation="treats disease",
            tail="headache",
            head_type="entity",
            tail_type="entity",
            confidence=0.9,
        )
        n = t.normalised()
        assert n.head == "Aspirin"
        assert n.relation == "TREATS_DISEASE"
        assert n.tail == "Headache"
        assert n.confidence == 0.9  # preserved

    def test_normalised_preserves_types(self):
        t = Triple(
            head="taking medication",
            relation="PREVENTS",
            tail="Disease",
            head_type="event",
            tail_type="entity",
        )
        n = t.normalised()
        assert n.head_type == "event"
        assert n.tail_type == "entity"


class TestExtractionResult:
    def test_empty(self):
        r = ExtractionResult()
        assert r.triples == []
        assert r.entity_triples == []
        assert r.event_triples == []
        assert r.chunks_processed == 0
        assert r.total_chunks == 0

    def test_counting(self):
        r = ExtractionResult(total_chunks=5, chunks_processed=3)
        assert r.total_chunks == 5
        assert r.chunks_processed == 3


# ──────────────────────────────────────────────────────────────
# Mock LLM extraction test
# ──────────────────────────────────────────────────────────────

class MockLLMClient:
    """Mock LLM that returns canned triples."""

    def __init__(self, response):
        self.response = response
        self.call_count = 0

    def generate_json(self, prompt, system=None):
        self.call_count += 1
        return self.response

    def generate(self, prompt, system=None):
        self.call_count += 1
        return str(self.response)


class TestTripleExtractorWithMock:
    def _make_extractor(self, llm_response):
        from src.kg_construction.triple_extractor import TripleExtractor
        mock = MockLLMClient(llm_response)
        return TripleExtractor(llm=mock, chunk_size=500, confidence_threshold=0.5), mock

    def test_extracts_entity_triples(self):
        response = [
            {
                "head": "aspirin",
                "relation": "TREATS",
                "tail": "headache",
                "head_type": "entity",
                "tail_type": "entity",
                "confidence": 0.95,
            }
        ]
        ext, mock = self._make_extractor(response)
        result = ext.extract("Aspirin is used to treat headache.")
        assert len(result.triples) >= 1
        assert result.triples[0].head == "Aspirin"
        assert result.triples[0].relation == "TREATS"
        assert result.triples[0].tail == "Headache"
        assert result.entity_triples  # should have entity triples

    def test_extracts_event_triples(self):
        response = [
            {
                "head": "taking aspirin daily",
                "relation": "REDUCES_RISK_OF",
                "tail": "heart attack",
                "head_type": "event",
                "tail_type": "entity",
                "confidence": 0.85,
            }
        ]
        ext, _ = self._make_extractor(response)
        result = ext.extract("Taking aspirin daily reduces risk of heart attack.")
        assert len(result.event_triples) >= 1

    def test_filters_low_confidence(self):
        response = [
            {"head": "A", "relation": "R", "tail": "B", "head_type": "entity", "tail_type": "entity", "confidence": 0.3},
            {"head": "C", "relation": "R", "tail": "D", "head_type": "entity", "tail_type": "entity", "confidence": 0.8},
        ]
        ext, _ = self._make_extractor(response)
        result = ext.extract("Some text.")
        assert len(result.triples) == 1
        assert result.triples[0].head == "C"

    def test_handles_empty_response(self):
        ext, _ = self._make_extractor([])
        result = ext.extract("Some text.")
        assert len(result.triples) == 0

    def test_handles_malformed_items(self):
        response = [
            "not a dict",
            {"head": "", "relation": "R", "tail": "B", "confidence": 0.9},
            {"head": "A", "relation": "R", "tail": "B", "confidence": 0.9,
             "head_type": "entity", "tail_type": "entity"},
        ]
        ext, _ = self._make_extractor(response)
        result = ext.extract("Some text.")
        # Only the last valid item should survive
        assert len(result.triples) == 1

    def test_extract_from_chunks(self):
        response = [
            {"head": "X", "relation": "REL", "tail": "Y",
             "head_type": "entity", "tail_type": "entity", "confidence": 0.9}
        ]
        ext, mock = self._make_extractor(response)
        chunks = ["Chunk 1", "Chunk 2", "Chunk 3"]
        result = ext.extract_from_chunks(chunks)
        assert mock.call_count == 3
        assert result.chunks_processed == 3
        assert result.total_chunks == 3
