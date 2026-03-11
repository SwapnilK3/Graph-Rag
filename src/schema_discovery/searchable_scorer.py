"""
src/schema_discovery/searchable_scorer.py
------------------------------------------
Multi-factor scoring to identify "searchable" properties — those
most useful for entity lookup in a Graph-RAG query layer.

Based on the **user's draft paper** §3.1 "Automatic Schema Discovery":
  - Name pattern bonus     : +0.40 (property name matches common name
    patterns like 'name', 'title', 'label', 'id', etc.)
  - High cardinality bonus : +0.30 (many distinct values ⟹ likely an
    identifier)
  - String type bonus      : +0.30 (strings are searchable by nature)

  Score > 0.5 → property is marked as searchable.

Additionally incorporates ideas from PG-HIVE (uniqueness) and
Schema Inference (kind-equivalence) for more robust scoring.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field

from src.schema_discovery.property_analyzer import PropertyInfo

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

# Weights from the user's draft paper §3.1
W_NAME_PATTERN = 0.40
W_CARDINALITY = 0.30
W_STRING_TYPE = 0.30

# Threshold above which a property is considered searchable
SEARCHABLE_THRESHOLD = 0.5

# Patterns for property names likely to be identifiers/names
_NAME_PATTERNS = [
    re.compile(r"\bname\b", re.I),
    re.compile(r"\btitle\b", re.I),
    re.compile(r"\blabel\b", re.I),
    re.compile(r"\bdescription\b", re.I),
    re.compile(r"\bsummary\b", re.I),
    re.compile(r"\bidentifier\b", re.I),
    re.compile(r"\bcode\b", re.I),
    re.compile(r"\babbreviation\b", re.I),
    re.compile(r"\bslug\b", re.I),
    re.compile(r"\bemail\b", re.I),
    re.compile(r"\busername\b", re.I),
    re.compile(r"\bdisplay\b", re.I),
    re.compile(r"^id$", re.I),
]


# ──────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────

@dataclass
class SearchableScore:
    """Scoring breakdown for one property."""
    property_name: str
    label: str                          # node/edge type
    name_pattern_score: float = 0.0     # 0 or W_NAME_PATTERN
    cardinality_score: float = 0.0      # 0..W_CARDINALITY
    string_type_score: float = 0.0      # 0 or W_STRING_TYPE
    total_score: float = 0.0
    is_searchable: bool = False

    def __repr__(self):
        star = " ★" if self.is_searchable else ""
        return (
            f"  {self.property_name}: {self.total_score:.2f}"
            f" (name={self.name_pattern_score:.2f}"
            f", card={self.cardinality_score:.2f}"
            f", str={self.string_type_score:.2f}){star}"
        )


# ──────────────────────────────────────────────────────────────
# Scorer
# ──────────────────────────────────────────────────────────────

class SearchableScorer:
    """
    Score properties to determine which are most useful for entity
    lookup (i.e. searchable).

    Parameters
    ----------
    threshold : float
        Score above which a property is considered searchable.
    w_name : float
        Weight for name-pattern match.
    w_cardinality : float
        Weight for high-cardinality (many distinct values).
    w_string : float
        Weight for string-type properties.
    """

    def __init__(
        self,
        threshold: float = SEARCHABLE_THRESHOLD,
        w_name: float = W_NAME_PATTERN,
        w_cardinality: float = W_CARDINALITY,
        w_string: float = W_STRING_TYPE,
    ):
        self.threshold = threshold
        self.w_name = w_name
        self.w_cardinality = w_cardinality
        self.w_string = w_string

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_properties(
        self, label: str, properties: dict[str, PropertyInfo]
    ) -> list[SearchableScore]:
        """
        Score all properties of a type and return sorted results
        (highest score first).
        """
        scores = []
        for key, info in properties.items():
            s = self.score_property(label, info)
            scores.append(s)
        scores.sort(key=lambda x: x.total_score, reverse=True)
        return scores

    def score_property(
        self, label: str, prop: PropertyInfo
    ) -> SearchableScore:
        """
        Compute the searchable score for a single property.

        Three factors (user's draft paper §3.1):
        1. Name pattern: does the property name match known identifier
           patterns? → +w_name
        2. Cardinality: unique_ratio > 0.5 → scale by w_cardinality
        3. String type: is the data type STRING? → +w_string
        """
        s = SearchableScore(property_name=prop.name, label=label)

        # Factor 1: Name pattern
        s.name_pattern_score = self._name_pattern_score(prop.name)

        # Factor 2: Cardinality (high distinct ratio = identifier-like)
        s.cardinality_score = self._cardinality_score(prop.unique_ratio)

        # Factor 3: String type
        s.string_type_score = self._string_type_score(prop.data_type)

        # Total
        s.total_score = (
            s.name_pattern_score + s.cardinality_score + s.string_type_score
        )
        s.is_searchable = s.total_score >= self.threshold

        return s

    def get_searchable_properties(
        self, label: str, properties: dict[str, PropertyInfo]
    ) -> list[str]:
        """Return only property names that pass the searchable threshold."""
        scores = self.score_properties(label, properties)
        return [s.property_name for s in scores if s.is_searchable]

    # ------------------------------------------------------------------
    # Scoring factors
    # ------------------------------------------------------------------

    def _name_pattern_score(self, name: str) -> float:
        """Check if property name matches known identifier/name patterns."""
        for pattern in _NAME_PATTERNS:
            if pattern.search(name):
                return self.w_name
        return 0.0

    def _cardinality_score(self, unique_ratio: float) -> float:
        """
        Scale cardinality score by unique ratio.
        - unique_ratio > 0.8  → full weight (highly unique values)
        - unique_ratio > 0.5  → partial weight
        - unique_ratio ≤ 0.5  → zero
        """
        if unique_ratio >= 0.8:
            return self.w_cardinality
        if unique_ratio >= 0.5:
            # Linear interpolation between 0 and w_cardinality
            return self.w_cardinality * ((unique_ratio - 0.5) / 0.3)
        return 0.0

    def _string_type_score(self, data_type: str) -> float:
        """String properties get the full string-type bonus."""
        return self.w_string if data_type == "STRING" else 0.0
