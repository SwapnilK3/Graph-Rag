"""
path_scorer.py
--------------
V3 Component: Path Relevance Scorer

Scores and ranks paths in a subgraph by query relevance.
Used between traversal and context generation to prune low-value
edges and keep only high-signal paths.

Scoring Factors
---------------
1. Relationship type match (does rel type match the intent pattern?)
2. Entity name similarity (do path nodes mention query terms?)
3. Path length penalty (shorter = more relevant)
4. Node degree normalization (high-degree hubs carry less signal per edge)
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class PathScorer:
    """
    Scores subgraph edges by query relevance.

    Parameters
    ----------
    config : dict
        Pipeline config (must contain 'intent_patterns').
    """

    def __init__(self, config: dict):
        self._intent_patterns = config.get("intent_patterns", {})

    def score_and_rank(
        self,
        subgraph: dict,
        intent: str,
        query: str,
        max_edges: int = 20,
    ) -> dict:
        """
        Score each edge in the subgraph and return a pruned, ranked version.

        Parameters
        ----------
        subgraph : dict
            Raw subgraph from traversal.
        intent : str
            Classified intent.
        query : str
            Original user query.
        max_edges : int
            Maximum edges to retain.

        Returns
        -------
        dict
            Subgraph with edges sorted by relevance (highest first), pruned to max_edges.
        """
        relationships = subgraph.get("relationships", [])
        nodes = subgraph.get("nodes", [])

        if not relationships:
            return subgraph

        # Build lookup
        id_to_name = {n["id"]: (n.get("name") or "").lower() for n in nodes}
        query_terms = set(query.lower().split())
        pattern = self._intent_patterns.get(intent, {})
        target_rel = pattern.get("relationship")

        # Score each edge
        scored_edges = []
        for rel in relationships:
            score = self._score_edge(rel, target_rel, id_to_name, query_terms)
            scored_edges.append((score, rel))

        # Sort descending by score
        scored_edges.sort(key=lambda x: x[0], reverse=True)

        # Prune to max_edges
        kept = scored_edges[:max_edges]

        # Rebuild subgraph with only surviving edges
        surviving_rels = [rel for _, rel in kept]
        used_ids = set()
        for rel in surviving_rels:
            used_ids.add(rel["source_id"])
            used_ids.add(rel["target_id"])

        # Keep entry nodes (first batch) always
        entry_ids = {n["id"] for n in nodes[:10]}
        keep_ids = used_ids | entry_ids
        surviving_nodes = [n for n in nodes if n["id"] in keep_ids]

        return {
            "nodes": surviving_nodes,
            "relationships": surviving_rels,
            "strategy": subgraph.get("strategy"),
            "hop_depth": subgraph.get("hop_depth"),
        }

    def _score_edge(
        self,
        rel: dict,
        target_rel: Optional[str],
        id_to_name: dict,
        query_terms: set,
    ) -> float:
        """
        Score a single edge on a 0-1 scale.

        Factors:
        - 0.4: Relationship type match (full match to intent pattern)
        - 0.3: Entity name overlap with query terms
        - 0.2: Relationship type is common/meaningful
        - 0.1: Base score (every edge has some value)
        """
        score = 0.1  # Base score

        # Factor 1: Relationship type match (0.4)
        rel_type = rel.get("type", "")
        if target_rel and rel_type == target_rel:
            score += 0.4
        elif target_rel:
            # Partial match for related relationship types
            if any(word in rel_type.lower() for word in target_rel.lower().split("_")):
                score += 0.2

        # Factor 2: Entity name overlap with query (0.3)
        src_name = id_to_name.get(rel.get("source_id"), "")
        tgt_name = id_to_name.get(rel.get("target_id"), "")
        name_terms = set(src_name.split()) | set(tgt_name.split())
        overlap = len(query_terms & name_terms)
        if overlap > 0:
            score += min(0.3, overlap * 0.15)

        # Factor 3: Meaningful relationship (0.2)
        # Longer relationship names tend to be more specific/meaningful
        if len(rel_type) > 3:
            score += 0.1
        if "_" in rel_type:  # Multi-word rels like HAS_SIDE_EFFECT
            score += 0.1

        return min(1.0, score)
