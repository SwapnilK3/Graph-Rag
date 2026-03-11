"""
src/schema_discovery/cardinality.py
------------------------------------
Edge cardinality inference following PG-HIVE (Sideri et al., 2024) §4.5
and Schema Inference (Lbath et al., 2021) §5.

PG-HIVE Algorithm (Cardinality Inference)
------------------------------------------
For each edge type E from source type S to target type T:
  max_out = max over all s ∈ S of  |{t : (s)-[E]->(t)}|
  max_in  = max over all t ∈ T of  |{s : (s)-[E]->(t)}|

  If max_out ≤ 1 and max_in ≤ 1  → ONE_TO_ONE   (1:1)
  If max_out ≤ 1 and max_in  > 1  → MANY_TO_ONE  (N:1)
  If max_out  > 1 and max_in ≤ 1  → ONE_TO_MANY  (1:N)
  If max_out  > 1 and max_in  > 1  → MANY_TO_MANY (M:N)

Also determines if an edge is optional (some nodes lack it).

Schema Inference (Lbath) additions:
  - Hierarchical cardinality across subtypes
  - Minimum multiplicity (0 = optional, 1 = mandatory participation)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.utils.connector import Neo4jConnector

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────

@dataclass
class CardinalityInfo:
    """Cardinality information for one (source_label)-[rel_type]->(target_label) triple."""
    rel_type: str
    source_label: str
    target_label: str
    max_out: int = 0            # max outgoing per source node
    max_in: int = 0             # max incoming per target node
    avg_out: float = 0.0
    avg_in: float = 0.0
    min_out: int = 0            # 0 = optional participation from source
    min_in: int = 0             # 0 = optional participation from target
    total_edges: int = 0
    cardinality: str = "M:N"    # 1:1 | 1:N | N:1 | M:N
    source_optional: bool = True   # do some source nodes lack this edge?
    target_optional: bool = True

    @property
    def notation(self) -> str:
        """Human-readable cardinality string like '1..* → 0..1'."""
        src_min = self.min_out
        src_max = self.max_out if self.max_out < 1000 else "*"
        tgt_min = self.min_in
        tgt_max = self.max_in if self.max_in < 1000 else "*"
        return f"[{src_min}..{src_max}] → [{tgt_min}..{tgt_max}]"


# ──────────────────────────────────────────────────────────────
# Analyzer
# ──────────────────────────────────────────────────────────────

class CardinalityAnalyzer:
    """
    Infer edge cardinalities from a populated Neo4j graph.

    Parameters
    ----------
    connector : Neo4jConnector
    """

    def __init__(self, connector: Neo4jConnector):
        self.connector = connector

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        rel_type: str,
        source_label: str = "",
        target_label: str = "",
    ) -> CardinalityInfo:
        """
        Compute cardinality for a specific edge type.

        Parameters
        ----------
        rel_type : str
            Relationship type.
        source_label, target_label : str (optional)
            Restrict to edges between specific labels.
        """
        info = CardinalityInfo(
            rel_type=rel_type,
            source_label=source_label,
            target_label=target_label,
        )

        src = f":`{source_label}`" if source_label else ""
        tgt = f":`{target_label}`" if target_label else ""
        pattern = f"(s{src})-[r:`{rel_type}`]->(t{tgt})"

        # Total edge count
        row = self.connector.run_single(
            f"MATCH {pattern} RETURN count(r) AS cnt"
        )
        info.total_edges = row["cnt"] if row else 0
        if info.total_edges == 0:
            return info

        # Out-degree stats per source node
        out_stats = self.connector.run_single(
            f"MATCH {pattern} "
            f"WITH s, count(t) AS deg "
            f"RETURN max(deg) AS mx, min(deg) AS mn, avg(deg) AS av"
        )
        if out_stats:
            info.max_out = out_stats["mx"] or 0
            info.min_out = out_stats["mn"] or 0
            info.avg_out = float(out_stats["av"] or 0)

        # In-degree stats per target node
        in_stats = self.connector.run_single(
            f"MATCH {pattern} "
            f"WITH t, count(s) AS deg "
            f"RETURN max(deg) AS mx, min(deg) AS mn, avg(deg) AS av"
        )
        if in_stats:
            info.max_in = in_stats["mx"] or 0
            info.min_in = in_stats["mn"] or 0
            info.avg_in = float(in_stats["av"] or 0)

        # Cardinality classification (PG-HIVE §4.5)
        info.cardinality = self._classify(info.max_out, info.max_in)

        # Optional participation
        info.source_optional = self._check_source_optional(
            source_label, rel_type, target_label,
        )
        info.target_optional = self._check_target_optional(
            source_label, rel_type, target_label,
        )

        return info

    def analyze_all(self) -> list[CardinalityInfo]:
        """Analyze cardinality for every (source_label, rel_type, target_label) triple in the graph."""
        # Discover all (source_label, rel_type, target_label) triples
        rows = self.connector.run(
            "MATCH (s)-[r]->(t) "
            "WITH labels(s)[0] AS sl, type(r) AS rt, labels(t)[0] AS tl "
            "RETURN DISTINCT sl, rt, tl ORDER BY sl, rt, tl"
        )

        results = []
        for r in rows:
            info = self.analyze(r["rt"], r["sl"], r["tl"])
            results.append(info)

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classify(max_out: int, max_in: int) -> str:
        """PG-HIVE cardinality classification."""
        if max_out <= 1 and max_in <= 1:
            return "1:1"
        if max_out <= 1 and max_in > 1:
            return "N:1"
        if max_out > 1 and max_in <= 1:
            return "1:N"
        return "M:N"

    def _check_source_optional(
        self, source_label: str, rel_type: str, target_label: str
    ) -> bool:
        """
        Are there source nodes that do NOT have this outgoing edge?
        If yes → participation is optional.
        """
        if not source_label:
            return True

        tgt = f":`{target_label}`" if target_label else ""

        total_src = self.connector.run_single(
            f"MATCH (s:`{source_label}`) RETURN count(s) AS cnt"
        )
        with_edge = self.connector.run_single(
            f"MATCH (s:`{source_label}`)-[:`{rel_type}`]->({tgt}) "
            f"RETURN count(DISTINCT s) AS cnt"
        )

        total = total_src["cnt"] if total_src else 0
        connected = with_edge["cnt"] if with_edge else 0
        return connected < total

    def _check_target_optional(
        self, source_label: str, rel_type: str, target_label: str
    ) -> bool:
        """
        Are there target nodes that do NOT have this incoming edge?
        If yes → participation is optional.
        """
        if not target_label:
            return True

        src = f":`{source_label}`" if source_label else ""

        total_tgt = self.connector.run_single(
            f"MATCH (t:`{target_label}`) RETURN count(t) AS cnt"
        )
        with_edge = self.connector.run_single(
            f"MATCH ({src})-[:`{rel_type}`]->(t:`{target_label}`) "
            f"RETURN count(DISTINCT t) AS cnt"
        )

        total = total_tgt["cnt"] if total_tgt else 0
        connected = with_edge["cnt"] if with_edge else 0
        return connected < total
