"""
src/kg_construction/triple_extractor.py
---------------------------------------
LLM-based triple extraction from text.

**Paper basis**: AutoSchemaKG (Bai et al., 2025)
  - Extracts TWO kinds of triples:
    1. Entity-level: (entity, relation, entity)
       e.g. ("Aspirin", "TREATS", "Headache")
    2. Event-level:  (event, relation, entity)
       e.g. ("Taking Aspirin daily", "REDUCES_RISK_OF", "Heart Attack")
  - Uses LLM for extraction with structured output
  - Achieves >90% precision/recall/F1

**Paper basis**: StructuGraphRAG (Zhu et al., 2024)
  - Multi-agent approach: corpus preparation → extraction → ontology
  - Leverages document structure for better extraction

Algorithm
---------
For each text chunk:
  1. Send chunk to LLM with extraction prompt
  2. LLM returns JSON array of triples
  3. Normalise entity/relation names
  4. Score confidence per triple
  5. Filter by confidence threshold
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field

from src.utils.llm_client import LLMClient
from src.utils.text_processor import chunk_text

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────

@dataclass
class Triple:
    """A single (head, relation, tail) triple extracted from text."""
    head: str
    relation: str
    tail: str
    head_type: str = ""      # entity | event
    tail_type: str = ""      # entity | event
    confidence: float = 1.0
    source_chunk: str = ""

    def normalised(self) -> "Triple":
        """Return a copy with normalised names."""
        return Triple(
            head=_normalise_entity(self.head),
            relation=_normalise_relation(self.relation),
            tail=_normalise_entity(self.tail),
            head_type=self.head_type,
            tail_type=self.tail_type,
            confidence=self.confidence,
            source_chunk=self.source_chunk,
        )


@dataclass
class ExtractionResult:
    """Result of extracting triples from a document."""
    triples: list[Triple] = field(default_factory=list)
    entity_triples: list[Triple] = field(default_factory=list)
    event_triples: list[Triple] = field(default_factory=list)
    chunks_processed: int = 0
    total_chunks: int = 0


# ──────────────────────────────────────────────────────────────
# Normalisation helpers
# ──────────────────────────────────────────────────────────────

def _normalise_entity(name: str) -> str:
    """
    Normalise an entity name: title-case, strip extra whitespace.
    e.g. "  aspirin  " → "Aspirin"
    """
    name = re.sub(r"\s+", " ", name.strip())
    return name.title() if name else name


def _normalise_relation(name: str) -> str:
    """
    Normalise a relation to UPPER_SNAKE_CASE.
    e.g. "treats disease" → "TREATS_DISEASE"
    """
    name = re.sub(r"\s+", " ", name.strip())
    name = re.sub(r"[^a-zA-Z0-9 _]", "", name)
    return name.upper().replace(" ", "_")


# ──────────────────────────────────────────────────────────────
# Extraction prompts (AutoSchemaKG-inspired)
# ──────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert knowledge graph extraction system.
Your task is to extract ALL factual triples from the given text.

Extract TWO types of triples (from AutoSchemaKG methodology):

1. ENTITY-ENTITY triples: (entity, relation, entity)
   - Entities are concrete nouns: people, places, things, concepts
   - Example: ("Aspirin", "TREATS", "Headache")

2. EVENT-ENTITY triples: (event/action, relation, entity)
   - Events are actions, processes, or occurrences
   - Example: ("Taking Aspirin daily", "REDUCES_RISK_OF", "Heart Attack")

Rules:
- Extract ALL meaningful triples — do not omit any.
- Relations should be concise verbs or verb phrases in UPPER_SNAKE_CASE.
- Entities should be noun phrases (no articles).
- Assign a confidence score 0.0-1.0 to each triple.
- Classify head/tail as "entity" or "event".
"""

_EXTRACTION_PROMPT = """\
Extract all knowledge graph triples from the following text.

TEXT:
{text}

Return a JSON array of objects, each with:
{{
  "head": "entity or event name",
  "relation": "RELATION_NAME",
  "tail": "entity or event name",
  "head_type": "entity" or "event",
  "tail_type": "entity" or "event",
  "confidence": 0.0-1.0
}}

Return ONLY the JSON array, no other text.
"""


# ──────────────────────────────────────────────────────────────
# Main extractor
# ──────────────────────────────────────────────────────────────

class TripleExtractor:
    """
    Extracts (head, relation, tail) triples from text using an LLM.

    Implements the AutoSchemaKG extraction methodology:
      1. Chunk text into segments
      2. Send each chunk to LLM for triple extraction
      3. Parse structured JSON output
      4. Normalise and deduplicate

    Parameters
    ----------
    llm : LLMClient
        The LLM client for generation.
    chunk_size : int
        Maximum characters per chunk (default 1500).
    chunk_overlap : int
        Overlap characters between chunks (default 200).
    confidence_threshold : float
        Minimum confidence to keep a triple (default 0.5).
    """

    def __init__(
        self,
        llm: LLMClient,
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
        confidence_threshold: float = 0.5,
    ):
        self.llm = llm
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.confidence_threshold = confidence_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, text: str) -> ExtractionResult:
        """
        Extract all triples from a document.

        Parameters
        ----------
        text : str
            Full document text.

        Returns
        -------
        ExtractionResult
            Contains all triples separated into entity/event categories.
        """
        chunks = chunk_text(
            text,
            max_chunk_size=self.chunk_size,
            overlap=self.chunk_overlap,
        )
        result = ExtractionResult(total_chunks=len(chunks))

        for chunk in chunks:
            triples = self._extract_chunk(chunk)
            result.triples.extend(triples)
            result.chunks_processed += 1
            logger.info(
                "Chunk %d/%d: extracted %d triples",
                result.chunks_processed, result.total_chunks, len(triples),
            )

        # Separate entity vs event triples
        for t in result.triples:
            if t.head_type == "event" or t.tail_type == "event":
                result.event_triples.append(t)
            else:
                result.entity_triples.append(t)

        logger.info(
            "Extraction complete: %d entity triples, %d event triples",
            len(result.entity_triples), len(result.event_triples),
        )
        return result

    def extract_from_chunks(self, chunks: list[str]) -> ExtractionResult:
        """Extract triples from pre-chunked text segments."""
        result = ExtractionResult(total_chunks=len(chunks))
        for chunk in chunks:
            triples = self._extract_chunk(chunk)
            result.triples.extend(triples)
            result.chunks_processed += 1

        for t in result.triples:
            if t.head_type == "event" or t.tail_type == "event":
                result.event_triples.append(t)
            else:
                result.entity_triples.append(t)

        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _extract_chunk(self, chunk: str) -> list[Triple]:
        """Send a single chunk to the LLM and parse triples."""
        prompt = _EXTRACTION_PROMPT.format(text=chunk)

        try:
            raw = self.llm.generate_json(prompt, system=_SYSTEM_PROMPT)
        except Exception as e:
            logger.warning("LLM extraction failed for chunk: %s", e)
            return []

        if not isinstance(raw, list):
            raw = [raw]

        triples: list[Triple] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                t = Triple(
                    head=item.get("head", ""),
                    relation=item.get("relation", ""),
                    tail=item.get("tail", ""),
                    head_type=item.get("head_type", "entity"),
                    tail_type=item.get("tail_type", "entity"),
                    confidence=float(item.get("confidence", 0.8)),
                    source_chunk=chunk[:200],
                )
                t = t.normalised()

                # Filter by confidence
                if (
                    t.confidence >= self.confidence_threshold
                    and t.head
                    and t.relation
                    and t.tail
                ):
                    triples.append(t)
            except (ValueError, TypeError) as e:
                logger.debug("Skipping malformed triple: %s", e)

        return triples
