"""
tests/test_extractor.py
-----------------------
Unit tests for Component 1: QueryTimeEntityExtractor

Run with:
    pytest tests/test_extractor.py -v

The Neo4j calls are mocked so these tests run entirely offline.
"""

import pytest
from unittest.mock import MagicMock, patch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from entity_extractor import QueryTimeEntityExtractor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_node(node_id: str, label: str, name: str) -> dict:
    """Helper: construct a fake graph-query result row."""
    return {
        "id": node_id,
        "label": label,
        "properties": {"name": name},
        "candidate_name": name.lower(),
    }


@pytest.fixture
def mock_connector():
    """A GraphDBConnector whose execute_query is fully controllable."""
    conn = MagicMock()
    conn.execute_query.return_value = []   # default: empty graph
    return conn


@pytest.fixture
def extractor(mock_connector):
    return QueryTimeEntityExtractor(mock_connector)


# ---------------------------------------------------------------------------
# _extract_keywords tests
# ---------------------------------------------------------------------------

class TestExtractKeywords:
    def test_removes_stop_words(self, extractor):
        keywords = extractor._extract_keywords("What are the side effects of aspirin?")
        # "what", "are", "the", "of" should all be stripped
        assert "what" not in keywords
        assert "the" not in keywords
        assert "of" not in keywords

    def test_aspirin_present(self, extractor):
        keywords = extractor._extract_keywords("What are the side effects of aspirin?")
        assert "aspirin" in keywords

    def test_multi_word_entity_bigram(self, extractor):
        keywords = extractor._extract_keywords("stomach bleeding is a side effect")
        assert "stomach bleeding" in keywords

    def test_trigrams_before_unigrams(self, extractor):
        """Longer n-grams must appear earlier in the list."""
        keywords = extractor._extract_keywords("blood pressure medication")
        # "blood pressure medication" (trigram) should come before
        # "blood pressure" (bigram) and before "blood" (unigram)
        idx_trigram = keywords.index("blood pressure medication")
        idx_bigram  = keywords.index("blood pressure")
        idx_unigram = keywords.index("blood")
        assert idx_trigram < idx_bigram < idx_unigram

    def test_empty_query_returns_empty(self, extractor):
        assert extractor._extract_keywords("") == []

    def test_only_stop_words_falls_back(self, extractor):
        """If every token is a stop-word the fallback returns tokens anyway."""
        keywords = extractor._extract_keywords("what is the")
        assert len(keywords) > 0   # must not crash or return empty silently

    def test_hyphenated_drug_name(self, extractor):
        keywords = extractor._extract_keywords("Tell me about co-codamol")
        assert "co-codamol" in keywords


# ---------------------------------------------------------------------------
# _exact_match / _partial_match tests
# ---------------------------------------------------------------------------

class TestGraphLookup:
    def test_exact_match_hit(self, extractor, mock_connector):
        """Connector returns one row → one node dict back."""
        mock_connector.execute_query.return_value = [
            {"id": "4:abc:1", "label": "Drug", "properties": {"name": "Aspirin"}}
        ]
        nodes = extractor._exact_match("aspirin")
        assert len(nodes) == 1
        assert nodes[0]["name"] == "Aspirin"
        assert nodes[0]["label"] == "Drug"

    def test_exact_match_miss_returns_empty(self, extractor, mock_connector):
        mock_connector.execute_query.return_value = []
        nodes = extractor._exact_match("nonexistentdrug")
        assert nodes == []

    def test_partial_match_returns_results(self, extractor, mock_connector):
        mock_connector.execute_query.return_value = [
            {"id": "4:abc:2", "label": "Drug", "properties": {"name": "Ibuprofen"}}
        ]
        nodes = extractor._partial_match("ibu")
        assert len(nodes) == 1
        assert nodes[0]["name"] == "Ibuprofen"


# ---------------------------------------------------------------------------
# extract_entry_nodes (integration of all steps)
# ---------------------------------------------------------------------------

class TestExtractEntryNodes:
    def test_finds_aspirin(self, extractor, mock_connector):
        aspirin_row = {
            "id": "4:abc:10",
            "label": "Drug",
            "properties": {"name": "Aspirin"},
        }

        def side_effect(cypher, params):
            # Only return a result when the keyword matches "aspirin"
            keyword = params.get("keyword", "")
            if "aspirin" in keyword:
                return [aspirin_row]
            return []

        mock_connector.execute_query.side_effect = side_effect

        nodes = extractor.extract_entry_nodes("What are the side effects of aspirin?")
        assert len(nodes) == 1
        assert nodes[0]["name"] == "Aspirin"

    def test_no_entity_returns_empty(self, extractor, mock_connector):
        mock_connector.run_query.return_value = []
        nodes = extractor.extract_entry_nodes("Tell me something interesting")
        assert nodes == []

    def test_deduplication(self, extractor, mock_connector):
        """Same node matched by multiple keywords must appear only once."""
        shared_row = {
            "id": "4:abc:99",
            "label": "Drug",
            "properties": {"name": "Aspirin"},
        }
        # Every call returns the same node
        mock_connector.execute_query.return_value = [shared_row]

        nodes = extractor.extract_entry_nodes("aspirin aspirin again")
        ids = [n["id"] for n in nodes]
        assert len(ids) == len(set(ids)), "Duplicate nodes returned!"

    def test_multi_entity_query(self, extractor, mock_connector):
        """'aspirin' and 'warfarin' should each find their node."""
        def side_effect(cypher, params):
            keyword = params.get("keyword", "")
            if "aspirin" in keyword:
                return [{"id": "4:abc:1", "label": "Drug", "properties": {"name": "Aspirin"}}]
            if "warfarin" in keyword:
                return [{"id": "4:abc:2", "label": "Drug", "properties": {"name": "Warfarin"}}]
            return []

        mock_connector.execute_query.side_effect = side_effect
        nodes = extractor.extract_entry_nodes("Can I take aspirin with warfarin?")
        names = {n["name"] for n in nodes}
        assert "Aspirin" in names
        assert "Warfarin" in names

    def test_empty_query_returns_empty(self, extractor):
        nodes = extractor.extract_entry_nodes("")
        assert nodes == []

    def test_whitespace_only_query(self, extractor):
        nodes = extractor.extract_entry_nodes("   ")
        assert nodes == []


# ---------------------------------------------------------------------------
# Configuration options
# ---------------------------------------------------------------------------

class TestConfiguration:
    def test_custom_search_properties(self, mock_connector):
        """Extractor should use custom search_properties in its Cypher."""
        ext = QueryTimeEntityExtractor(
            mock_connector,
            search_properties=["name", "generic_name", "brand_name"],
        )
        mock_connector.execute_query.return_value = []
        ext.extract_entry_nodes("tylenol")

        # Verify all three properties appear in at least one Cypher call
        all_cyphers = " ".join(
            str(call.args[0]) for call in mock_connector.execute_query.call_args_list
        )
        assert "generic_name" in all_cyphers
        assert "brand_name" in all_cyphers

    def test_label_restriction(self, mock_connector):
        """With node_labels set, Cypher should include a label filter."""
        ext = QueryTimeEntityExtractor(
            mock_connector,
            node_labels=["Drug", "Disease"],
        )
        mock_connector.execute_query.return_value = []
        ext.extract_entry_nodes("aspirin")

        all_cyphers = " ".join(
            str(call.args[0]) for call in mock_connector.execute_query.call_args_list
        )
        assert "Drug|Disease" in all_cyphers

    def test_extra_stop_words(self, mock_connector):
        """Words added via extra_stop_words must be excluded from keywords."""
        ext = QueryTimeEntityExtractor(
            mock_connector,
            extra_stop_words=["drug", "medication"],
        )
        keywords = ext._extract_keywords("what is this drug medication aspirin")
        assert "drug" not in keywords
        assert "medication" not in keywords
        assert "aspirin" in keywords
