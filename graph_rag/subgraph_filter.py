"""
subgraph_filter.py
------------------
Module 4: Subgraph Filter

Filters a raw traversal subgraph to retain only query-relevant edges
before it reaches the context generator. This eliminates noise and
reduces token waste in the LLM context window.

Filtering Strategy
------------------
1. Intent-based hard filter: If the intent pattern specifies a relationship
   type (e.g., "CAUSES"), keep only edges of that type.
2. For general/neighborhood intents: Keep all edge types but enforce
   deduplication and cap.
3. Always preserve entry nodes even if they have no surviving edges.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class SubgraphFilter:
    """
    Filters a raw subgraph to retain only query-relevant edges.

    Parameters
    ----------
    config : dict
        The pipeline config (must contain 'intent_patterns').
    """

    def __init__(self, config: dict):
        self._intent_patterns = config.get("intent_patterns", {})

    def filter(
        self,
        subgraph: dict,
        intent: str,
        query: str,
        max_edges: int = 25,
    ) -> dict:
        """
        Filter the subgraph to query-relevant edges.

        Parameters
        ----------
        subgraph : dict
            Raw subgraph from traversal engine.
        intent : str
            Classified intent name.
        query : str
            Original user query (for future semantic relevance scoring).
        max_edges : int
            Maximum number of edges to retain in the filtered subgraph.

        Returns
        -------
        dict
            Filtered subgraph with the same schema.
        """
        pattern = self._intent_patterns.get(intent, {})
        raw_rels = subgraph.get("relationships", [])
        raw_nodes = subgraph.get("nodes", [])

        # Step 1: Intent-based hard filter
        target_rel = pattern.get("relationship")
        if target_rel:
            # Targeted intent — keep only the specified relationship type
            filtered_rels = [r for r in raw_rels if r["type"] == target_rel]
            logger.debug(
                "Intent '%s' filtered to rel '%s': %d -> %d edges",
                intent, target_rel, len(raw_rels), len(filtered_rels),
            )
        else:
            # General/neighborhood — keep all types
            filtered_rels = raw_rels

        # Step 2: Deduplicate edges (same source + target + type = redundant)
        seen_keys: set[tuple] = set()
        deduped_rels: list[dict] = []
        for rel in filtered_rels:
            key = (rel["source_id"], rel["target_id"], rel["type"])
            if key not in seen_keys:
                seen_keys.add(key)
                deduped_rels.append(rel)

        # Step 3: Cap at max_edges
        if len(deduped_rels) > max_edges:
            logger.debug(
                "Capping edges from %d to %d", len(deduped_rels), max_edges
            )
            deduped_rels = deduped_rels[:max_edges]

        # Step 4: Prune nodes — keep only those referenced by surviving edges
        used_ids: set[str] = set()
        for rel in deduped_rels:
            used_ids.add(rel["source_id"])
            used_ids.add(rel["target_id"])

        # Always keep entry nodes (nodes from the original extraction)
        # They appear first in the node list by convention from _build_subgraph 
        entry_ids = {n["id"] for n in raw_nodes[:10]}  # first batch = entry nodes
        keep_ids = used_ids | entry_ids

        filtered_nodes = [n for n in raw_nodes if n["id"] in keep_ids]

        return {
            "nodes": filtered_nodes,
            "relationships": deduped_rels,
            "strategy": subgraph.get("strategy"),
            "hop_depth": subgraph.get("hop_depth"),
        }
