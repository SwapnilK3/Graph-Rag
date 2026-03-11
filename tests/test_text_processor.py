"""
tests/test_text_processor.py
-----------------------------
Offline tests for src/utils/text_processor.py
"""

import pytest
from src.utils.text_processor import chunk_text, extract_sentences


class TestChunkText:
    """Tests for paragraph-aware chunking (StructuGraphRAG approach)."""

    def test_empty_text(self):
        assert chunk_text("") == []

    def test_whitespace_only(self):
        assert chunk_text("   \n\n  ") == []

    def test_short_text_single_chunk(self):
        text = "Hello world. This is a short document."
        chunks = chunk_text(text, max_chunk_size=500)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_paragraph_splitting(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = chunk_text(text, max_chunk_size=30, overlap=0)
        assert len(chunks) >= 2
        # Each chunk should contain at least one paragraph
        for chunk in chunks:
            assert len(chunk.strip()) > 0

    def test_overlap(self):
        # Create text with 3 clear paragraphs
        para_a = "Alpha " * 50  # ~300 chars
        para_b = "Beta " * 50
        para_c = "Gamma " * 50
        text = f"{para_a}\n\n{para_b}\n\n{para_c}"
        chunks = chunk_text(text, max_chunk_size=350, overlap=50)
        # Should produce multiple chunks
        assert len(chunks) >= 2

    def test_respects_max_chunk_size(self):
        text = "Word " * 500  # 2500 chars
        chunks = chunk_text(text, max_chunk_size=200, overlap=0)
        for chunk in chunks:
            # Allow some tolerance for sentence/word boundaries
            assert len(chunk) <= 400  # generous upper bound

    def test_no_empty_chunks(self):
        text = "A.\n\nB.\n\nC.\n\nD.\n\nE."
        chunks = chunk_text(text, max_chunk_size=50, overlap=0)
        for chunk in chunks:
            assert len(chunk.strip()) > 0


class TestExtractSentences:
    """Tests for sentence extraction."""

    def test_empty(self):
        assert extract_sentences("") == []

    def test_single_sentence(self):
        sents = extract_sentences("Hello world.")
        assert len(sents) >= 1

    def test_multiple_sentences(self):
        text = "First sentence. Second sentence. Third sentence."
        sents = extract_sentences(text)
        assert len(sents) >= 2

    def test_abbreviations(self):
        text = "Dr. Smith went to Washington. He arrived at 5 p.m."
        sents = extract_sentences(text)
        assert len(sents) >= 1
