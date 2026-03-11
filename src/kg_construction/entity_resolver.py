"""
src/kg_construction/entity_resolver.py
--------------------------------------
Entity resolution and deduplication for extracted triples.

**Paper basis**: AutoSchemaKG (Bai et al., 2025)
  - Entity deduplication across chunks
  - Co-reference resolution
  - Normalisation of entity mentions

**Paper basis**: StructuGraphRAG (Zhu et al., 2024)
  - Entity mapping to ontology classes
  - Embedding-based entity clustering

Algorithm
---------
1. Normalise all entity names (case, whitespace, articles)
2. Build similarity groups using string similarity
3. For each group, pick canonical form (most frequent or longest)
4. Rewrite all triples to use canonical entities
5. Merge duplicate triples
"""

from __future__ import annotations

import re
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from src.kg_construction.triple_extractor import Triple

logger = logging.getLogger(__name__)

# Try to import Levenshtein for fast fuzzy matching
try:
    from Levenshtein import ratio as levenshtein_ratio
    HAS_LEVENSHTEIN = True
except ImportError:
    HAS_LEVENSHTEIN = False


# ──────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────

@dataclass
class ResolutionResult:
    """Result of entity resolution."""
    resolved_triples: list[Triple] = field(default_factory=list)
    canonical_map: dict[str, str] = field(default_factory=dict)
    merge_groups: dict[str, list[str]] = field(default_factory=dict)
    original_count: int = 0
    resolved_count: int = 0
    entities_before: int = 0
    entities_after: int = 0


# ──────────────────────────────────────────────────────────────
# Similarity functions
# ──────────────────────────────────────────────────────────────

def _normalise_key(name: str) -> str:
    """Create a normalisation key: lowercase, no articles, no extra spaces."""
    name = name.lower().strip()
    name = re.sub(r"\b(the|a|an)\b", "", name)
    name = re.sub(r"[^a-z0-9 ]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def _string_similarity(a: str, b: str) -> float:
    """
    Compute similarity between two strings.
    Uses Levenshtein ratio if available, else Jaccard on character bigrams.
    """
    if HAS_LEVENSHTEIN:
        return levenshtein_ratio(a.lower(), b.lower())

    # Fallback: Jaccard on character bigrams
    def bigrams(s):
        s = s.lower()
        return set(s[i:i+2] for i in range(len(s) - 1))

    bg_a, bg_b = bigrams(a), bigrams(b)
    if not bg_a and not bg_b:
        return 1.0
    if not bg_a or not bg_b:
        return 0.0
    return len(bg_a & bg_b) / len(bg_a | bg_b)


# ──────────────────────────────────────────────────────────────
# Main resolver
# ──────────────────────────────────────────────────────────────

class EntityResolver:
    """
    Resolves and deduplicates entities across extracted triples.

    Parameters
    ----------
    similarity_threshold : float
        Minimum similarity to consider two entities the same (default 0.85).
    use_normalisation : bool
        Whether to use normalisation-key matching in addition to
        similarity-based matching (default True).
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        use_normalisation: bool = True,
    ):
        self.similarity_threshold = similarity_threshold
        self.use_normalisation = use_normalisation

    def resolve(self, triples: list[Triple]) -> ResolutionResult:
        """
        Resolve entities across all triples.

        Steps:
        1. Collect all unique entity mentions
        2. Group by normalisation key (exact-after-normalisation matches)
        3. Within remaining ungrouped, fuzzy match by string similarity
        4. Pick canonical form for each group
        5. Rewrite triples and deduplicate
        """
        result = ResolutionResult(original_count=len(triples))

        # Step 1: Collect all unique entity mentions
        all_entities: set[str] = set()
        for t in triples:
            all_entities.add(t.head)
            all_entities.add(t.tail)
        result.entities_before = len(all_entities)

        # Step 2+3: Build merge groups
        canonical_map = self._build_canonical_map(all_entities)
        result.canonical_map = canonical_map

        # Build merge groups for reporting
        groups: dict[str, list[str]] = defaultdict(list)
        for original, canonical in canonical_map.items():
            if original != canonical:
                groups[canonical].append(original)
        result.merge_groups = dict(groups)

        # Step 4+5: Rewrite triples
        seen: set[tuple] = set()
        for t in triples:
            new_triple = Triple(
                head=canonical_map.get(t.head, t.head),
                relation=t.relation,
                tail=canonical_map.get(t.tail, t.tail),
                head_type=t.head_type,
                tail_type=t.tail_type,
                confidence=t.confidence,
                source_chunk=t.source_chunk,
            )
            # Deduplicate
            key = (new_triple.head, new_triple.relation, new_triple.tail)
            if key not in seen:
                seen.add(key)
                result.resolved_triples.append(new_triple)

        result.resolved_count = len(result.resolved_triples)
        result.entities_after = len(
            set(t.head for t in result.resolved_triples)
            | set(t.tail for t in result.resolved_triples)
        )

        logger.info(
            "Entity resolution: %d→%d triples, %d→%d entities",
            result.original_count, result.resolved_count,
            result.entities_before, result.entities_after,
        )
        return result

    # ------------------------------------------------------------------
    # Build canonical map
    # ------------------------------------------------------------------

    def _build_canonical_map(self, entities: set[str]) -> dict[str, str]:
        """
        Build a mapping from each entity mention to its canonical form.
        """
        # Phase 1: Group by normalisation key
        norm_groups: dict[str, list[str]] = defaultdict(list)
        for entity in entities:
            key = _normalise_key(entity)
            norm_groups[key].append(entity)

        # Phase 2: Fuzzy merge groups with similar keys
        keys = list(norm_groups.keys())
        merged = set()
        for i in range(len(keys)):
            if keys[i] in merged:
                continue
            for j in range(i + 1, len(keys)):
                if keys[j] in merged:
                    continue
                sim = _string_similarity(keys[i], keys[j])
                if sim >= self.similarity_threshold:
                    # Merge j into i
                    norm_groups[keys[i]].extend(norm_groups[keys[j]])
                    merged.add(keys[j])

        for key in merged:
            del norm_groups[key]

        # Phase 3: Pick canonical form for each group
        canonical_map: dict[str, str] = {}
        for key, group in norm_groups.items():
            canonical = self._pick_canonical(group)
            for entity in group:
                canonical_map[entity] = canonical

        return canonical_map

    def _pick_canonical(self, group: list[str]) -> str:
        """
        Pick the canonical form from a group of entity mentions.
        Strategy: prefer Title Case, then longest, then most frequent.
        """
        if len(group) == 1:
            return group[0]

        # Count frequency
        freq = Counter(group)

        # Prefer Title Case version
        title_versions = [e for e in group if e == e.title()]
        if title_versions:
            return max(title_versions, key=lambda e: (freq[e], len(e)))

        # Otherwise, prefer longest
        return max(group, key=lambda e: (freq[e], len(e)))
