"""
src/utils/text_processor.py
---------------------------
Text ingestion and chunking utilities.

Based on StructuGraphRAG's approach: preserve document structure,
chunk by meaningful boundaries (paragraphs, sections) rather than
arbitrary token windows.
"""

import re


def chunk_text(
    text: str,
    max_chunk_size: int = 1500,
    overlap: int = 200,
    separator: str | None = None,
) -> list[str]:
    """
    Split *text* into overlapping chunks.

    Strategy (from StructuGraphRAG):
      1. Split on paragraph boundaries (double newlines).
      2. If a paragraph exceeds *max_chunk_size*, split on sentences.
      3. Merge small paragraphs until reaching *max_chunk_size*.
      4. Apply *overlap* characters of trailing context from the
         previous chunk to maintain continuity.

    Parameters
    ----------
    text : str
        Input document text.
    max_chunk_size : int
        Maximum characters per chunk (default 1500 ≈ 375 tokens).
    overlap : int
        Characters of overlap between consecutive chunks.
    separator : str, optional
        Custom separator regex. Default: double-newline.

    Returns
    -------
    list[str]
        Ordered list of text chunks.
    """
    if not text or not text.strip():
        return []

    # ---- step 1: split on paragraphs ----
    sep = separator or r"\n\s*\n"
    paragraphs = re.split(sep, text.strip())
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # ---- step 2: break oversized paragraphs into sentences ----
    segments: list[str] = []
    for para in paragraphs:
        if len(para) <= max_chunk_size:
            segments.append(para)
        else:
            # sentence-level split
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sent in sentences:
                if len(sent) <= max_chunk_size:
                    segments.append(sent)
                else:
                    # Hard split on word boundaries for very long runs
                    words = sent.split()
                    buf = ""
                    for w in words:
                        candidate = (buf + " " + w).strip() if buf else w
                        if len(candidate) <= max_chunk_size:
                            buf = candidate
                        else:
                            if buf:
                                segments.append(buf)
                            buf = w
                    if buf:
                        segments.append(buf)

    # ---- step 3: merge small segments ----
    chunks: list[str] = []
    current = ""
    for seg in segments:
        candidate = (current + "\n\n" + seg).strip() if current else seg
        if len(candidate) <= max_chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = seg
    if current:
        chunks.append(current)

    # ---- step 4: add overlap ----
    if overlap > 0 and len(chunks) > 1:
        overlapped: list[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-overlap:]
            overlapped.append(prev_tail + "\n" + chunks[i])
        chunks = overlapped

    return chunks


def extract_sentences(text: str) -> list[str]:
    """Split *text* into individual sentences."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]
