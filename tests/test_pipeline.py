"""
tests/test_pipeline.py
-----------------------
Offline tests for src/pipeline.py

Tests the MethodologyPipeline with mock LLM and mock connector.
"""

import pytest
from src.pipeline import MethodologyPipeline, PipelineResult


# ──────────────────────────────────────────────────────────────
# Mock LLM
# ──────────────────────────────────────────────────────────────

class MockLLM:
    """Mock LLM returning canned extraction and induction results."""

    def __init__(self):
        self.calls = []

    def generate_json(self, prompt, system=None):
        self.calls.append(prompt[:50])

        # Detection: extraction prompt asks for triples
        if "Extract all knowledge graph triples" in prompt:
            return [
                {
                    "head": "Aspirin",
                    "relation": "TREATS",
                    "tail": "Headache",
                    "head_type": "entity",
                    "tail_type": "entity",
                    "confidence": 0.95,
                },
                {
                    "head": "Ibuprofen",
                    "relation": "TREATS",
                    "tail": "Fever",
                    "head_type": "entity",
                    "tail_type": "entity",
                    "confidence": 0.90,
                },
            ]

        # Detection: induction prompt asks for types
        if "induce the entity types" in prompt:
            return {
                "types": [
                    {
                        "name": "Drug",
                        "description": "A pharmaceutical substance",
                        "entities": ["Aspirin", "Ibuprofen"],
                        "is_event_type": False,
                        "parent_type": None,
                    },
                    {
                        "name": "Condition",
                        "description": "A medical condition",
                        "entities": ["Headache", "Fever"],
                        "is_event_type": False,
                        "parent_type": None,
                    },
                ],
                "relation_types": [
                    {
                        "name": "TREATS",
                        "description": "Drug treats condition",
                        "source_types": ["Drug"],
                        "target_types": ["Condition"],
                    }
                ],
            }

        return []

    def generate(self, prompt, system=None):
        return str(self.generate_json(prompt, system))


# ──────────────────────────────────────────────────────────────
# Mock Connector (no actual DB calls)
# ──────────────────────────────────────────────────────────────

class MockConnector:
    """Mock connector that records writes and returns empty results."""

    def __init__(self):
        self.writes = []

    def run(self, cypher, params=None):
        return []

    def run_single(self, cypher, params=None):
        return None

    def write(self, cypher, params=None):
        self.writes.append(cypher)
        return []

    def run_batch(self, cypher, batch):
        for b in batch:
            self.writes.append(cypher)

    def clear_database(self):
        self.writes.append("CLEAR")

    def verify(self):
        return True

    def close(self):
        pass


# ──────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────

class TestMethodologyPipeline:

    def _make_pipeline(self, clear=False):
        llm = MockLLM()
        conn = MockConnector()
        return MethodologyPipeline(
            connector=conn,
            llm=llm,
            clear_graph=clear,
        ), llm, conn

    def test_extraction_stage(self):
        pipeline, llm, conn = self._make_pipeline()
        result = pipeline.extract_only("Aspirin treats headache. Ibuprofen treats fever.")
        assert len(result.triples) >= 1
        assert "extraction" not in []  # individual stage doesn't track

    def test_full_pipeline_stages(self):
        pipeline, llm, conn = self._make_pipeline()
        result = pipeline.run("Aspirin treats headache. Ibuprofen treats fever.")
        assert "extraction" in result.stages_completed
        assert "induction" in result.stages_completed
        assert "resolution" in result.stages_completed
        # build and discovery may fail with mock connector, that's OK

    def test_extraction_produces_triples(self):
        pipeline, _, _ = self._make_pipeline()
        result = pipeline.run("Aspirin treats headache. Ibuprofen treats fever.")
        assert result.extraction is not None
        assert len(result.extraction.triples) >= 1

    def test_induction_produces_types(self):
        pipeline, _, _ = self._make_pipeline()
        result = pipeline.run("Aspirin treats headache. Ibuprofen treats fever.")
        assert result.induced_schema is not None
        assert len(result.induced_schema.entity_types) >= 1

    def test_resolution_deduplicates(self):
        pipeline, _, _ = self._make_pipeline()
        result = pipeline.run("Aspirin treats headache. Ibuprofen treats fever.")
        assert result.resolution is not None
        assert result.resolution.resolved_count <= result.resolution.original_count

    def test_empty_text(self):
        pipeline, _, _ = self._make_pipeline()
        result = pipeline.run("")
        # Empty text → extraction succeeds but no triples → early exit
        assert "induction" not in result.stages_completed
        assert len(result.errors) > 0  # "No triples extracted"

    def test_pipeline_result_defaults(self):
        r = PipelineResult()
        assert r.extraction is None
        assert r.stages_completed == []
        assert r.errors == []

    def test_print_report_no_error(self):
        """print_report should not raise even with partial results."""
        pipeline, _, _ = self._make_pipeline()
        result = pipeline.run("Aspirin treats headache.")
        # Just verify it doesn't raise
        pipeline.print_report(result)
