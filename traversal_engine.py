"""
traversal_engine.py
-------------------
Component 3: Smart Traversal Engine

Supports five traversal strategies driven entirely by the domain config JSON.
All strategies produce the same standardised subgraph dict so the rest of the
pipeline (ContextGenerator, LLMInterface) never needs to know which strategy ran.

Strategies
----------
targeted      — 1-hop over a single named relationship (original behaviour)
chained       — fixed multi-hop chain following a sequence of relationships
variable_hop  — 1..N free hops; intermediate nodes included via path UNWIND
shortest_path — Neo4j shortestPath() between multiple entry nodes
shared_neighbor — nodes reachable from ALL entry nodes (intersection)

Subgraph format (returned by every strategy)
--------------------------------------------
{
    "nodes":         [{"id", "label", "name", "properties"}, ...],
    "relationships": [{"source_id", "target_id", "type", "properties"}, ...],
    "strategy":      str,   # which strategy was used
    "hop_depth":     int,   # maximum graph distance traversed
}
"""

import json
from connector import GraphDBConnector


# ---------------------------------------------------------------------------
# Shared RETURN clause used by single-edge queries and path UNWIND queries.
# All strategies produce rows conforming to this shape.
# ---------------------------------------------------------------------------
_EDGE_RETURN = """
    RETURN DISTINCT
        elementId(src)   AS source_id,
        labels(src)[0]   AS source_label,
        properties(src)  AS source_props,
        type(r)          AS rel_type,
        properties(r)    AS rel_props,
        elementId(tgt)   AS target_id,
        labels(tgt)[0]   AS target_label,
        properties(tgt)  AS target_props
"""

# UNWIND a Cypher path variable into individual edges.
# Caller must alias the path as `path` before this clause.
_PATH_UNWIND = """
    UNWIND range(0, length(path) - 1) AS i
    WITH nodes(path)[i]         AS src,
         relationships(path)[i] AS r,
         nodes(path)[i + 1]     AS tgt
"""


class SmartTraversalEngine:
    """
    Executes intent-driven graph traversal.

    Parameters
    ----------
    connector  : Active GraphDBConnector.
    config_path: Path to the domain config JSON.
    """

    def __init__(self, connector: GraphDBConnector, config_path: str):
        self.connector = connector
        with open(config_path) as f:
            self._config = json.load(f)
        self._intent_patterns: dict = self._config["intent_patterns"]
        self._general_limit: int = (
            self._config.get("general_traversal", {}).get("node_limit", 30)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def traverse(self, entry_nodes: list[dict], intent: str) -> dict:
        """
        Traverse the graph from entry_nodes using the strategy defined by intent.

        Returns a subgraph dict with keys: nodes, relationships, strategy, hop_depth.
        """
        if not entry_nodes:
            return {"nodes": [], "relationships": [], "strategy": "none", "hop_depth": 0}

        entry_ids = [n["id"] for n in entry_nodes]
        pattern   = self._intent_patterns.get(intent)
        strategy  = pattern.get("strategy", "targeted") if pattern else "general"

        dispatch = {
            "targeted":       self._targeted,
            "chained":        self._chained,
            "variable_hop":   self._variable_hop,
            "shortest_path":  self._shortest_path,
            "shared_neighbor":self._shared_neighbor,
        }

        fn = dispatch.get(strategy, self._general)
        rows, hop_depth = fn(entry_ids, pattern or {})

        subgraph = self._build_subgraph(rows, entry_nodes)
        subgraph["strategy"]  = strategy
        subgraph["hop_depth"] = hop_depth
        return subgraph

    # ------------------------------------------------------------------
    # Strategy: targeted  (1-hop, original behaviour)
    # ------------------------------------------------------------------

    def _targeted(self, entry_ids: list[str], pattern: dict) -> tuple[list[dict], int]:
        """Single named relationship, directional or symmetric."""
        src_label = pattern.get("source_label", "")
        rel_name  = pattern["relationship"]
        tgt_label = pattern.get("target_label", "")
        anchor    = pattern.get("entry_anchor", "source")

        src = f"(src:{src_label})" if src_label else "(src)"
        tgt = f"(tgt:{tgt_label})" if tgt_label else "(tgt)"

        if anchor == "either":
            where = "elementId(src) IN $ids OR elementId(tgt) IN $ids"
            match = f"MATCH {src}-[r:{rel_name}]-{tgt}"
        elif anchor == "target":
            where = "elementId(tgt) IN $ids"
            match = f"MATCH {src}-[r:{rel_name}]->{tgt}"
        else:
            where = "elementId(src) IN $ids"
            match = f"MATCH {src}-[r:{rel_name}]->{tgt}"

        cypher = f"{match}\nWHERE {where}\n{_EDGE_RETURN}\nLIMIT 50"
        return self.connector.execute_query(cypher, {"ids": entry_ids}), 1

    # ------------------------------------------------------------------
    # Strategy: chained  (fixed multi-hop sequence)
    # ------------------------------------------------------------------

    def _chained(self, entry_ids: list[str], pattern: dict) -> tuple[list[dict], int]:
        """
        Follow a fixed sequence of relationships from the entry nodes.

        Config shape:
            "hops": [
                {"relationship": "TREATS",      "target_label": "Disease"},
                {"relationship": "HAS_SYMPTOM", "target_label": "Symptom"}
            ],
            "entry_label": "Drug"   (optional)
        """
        hops        = pattern["hops"]
        entry_label = pattern.get("entry_label", "")
        n_hops      = len(hops)

        # Build: (n0:Drug)-[:TREATS]->(n1:Disease)-[:HAS_SYMPTOM]->(n2:Symptom)
        entry_node  = f"(n0:{entry_label})" if entry_label else "(n0)"
        path_tokens = [entry_node]
        for i, hop in enumerate(hops):
            rel   = hop["relationship"]
            label = hop.get("target_label", "")
            node  = f"(n{i + 1}:{label})" if label else f"(n{i + 1})"
            path_tokens.append(f"-[:{rel}]->")
            path_tokens.append(node)

        match_pattern = "".join(path_tokens)

        cypher = f"""
            MATCH path = {match_pattern}
            WHERE elementId(n0) IN $ids
            WITH path LIMIT 100
            {_PATH_UNWIND}
            {_EDGE_RETURN}
        """
        return self.connector.execute_query(cypher, {"ids": entry_ids}), n_hops

    # ------------------------------------------------------------------
    # Strategy: variable_hop  (free 1..N neighbourhood)
    # ------------------------------------------------------------------

    def _variable_hop(self, entry_ids: list[str], pattern: dict) -> tuple[list[dict], int]:
        """
        Explore up to max_hops hops in any direction, unrolling every
        intermediate edge so all nodes in the path appear in the subgraph.

        Config shape:
            "min_hops": 1,
            "max_hops": 2
        """
        min_h = pattern.get("min_hops", 1)
        max_h = pattern.get("max_hops", 2)

        cypher = f"""
            MATCH path = (source)-[*{min_h}..{max_h}]-(target)
            WHERE elementId(source) IN $ids AND source <> target
            WITH path LIMIT 60
            {_PATH_UNWIND}
            {_EDGE_RETURN}
        """
        return self.connector.execute_query(cypher, {"ids": entry_ids}), max_h

    # ------------------------------------------------------------------
    # Strategy: shortest_path  (connect multiple entry nodes)
    # ------------------------------------------------------------------

    def _shortest_path(self, entry_ids: list[str], pattern: dict) -> tuple[list[dict], int]:
        """
        Find the shortest undirected path between every pair of entry nodes.
        Also finds paths from each entry node to any node named in the query
        context (handled by the caller, who includes all matched nodes in
        entry_ids).

        Falls back to variable_hop when only one entry node is given.
        """
        max_h = pattern.get("max_hops", 6)

        if len(entry_ids) < 2:
            # Single entry: just do a neighbourhood exploration
            return self._variable_hop(entry_ids, {"min_hops": 1, "max_hops": max_h})

        cypher = f"""
            MATCH (a), (b)
            WHERE elementId(a) IN $ids
              AND elementId(b) IN $ids
              AND elementId(a) < elementId(b)
            MATCH path = shortestPath((a)-[*..{max_h}]-(b))
            WITH path LIMIT 20
            {_PATH_UNWIND}
            {_EDGE_RETURN}
        """
        rows = self.connector.execute_query(cypher, {"ids": entry_ids})
        hop_depth = max(len(r.get("path_lens", [])) for r in rows) if rows else max_h
        return rows, max_h

    # ------------------------------------------------------------------
    # Strategy: shared_neighbor  (intersection of neighbourhoods)
    # ------------------------------------------------------------------

    def _shared_neighbor(self, entry_ids: list[str], pattern: dict) -> tuple[list[dict], int]:
        """
        Find nodes that are direct neighbours of ALL (or most) entry nodes.
        Returns the edges connecting each entry node to those shared neighbours.

        Useful for queries like "what do aspirin and ibuprofen have in common?".

        min_connections defaults to 2 but will not exceed len(entry_ids).
        """
        min_conn = min(pattern.get("min_connections", 2), len(entry_ids))

        cypher = """
            MATCH (entry)-[r]-(neighbor)
            WHERE elementId(entry) IN $ids
            WITH neighbor, collect(DISTINCT elementId(entry)) AS connected_entries
            WHERE size(connected_entries) >= $min_conn
            WITH collect(elementId(neighbor)) AS shared_ids
            MATCH (src)-[r2]-(tgt)
            WHERE elementId(src) IN $ids AND elementId(tgt) IN shared_ids
            RETURN DISTINCT
                elementId(src)  AS source_id,
                labels(src)[0]  AS source_label,
                properties(src) AS source_props,
                type(r2)        AS rel_type,
                properties(r2)  AS rel_props,
                elementId(tgt)  AS target_id,
                labels(tgt)[0]  AS target_label,
                properties(tgt) AS target_props
        """
        return (
            self.connector.execute_query(cypher, {"ids": entry_ids, "min_conn": min_conn}),
            1,
        )

    # ------------------------------------------------------------------
    # Strategy: general  (fallback 1-hop in any direction)
    # ------------------------------------------------------------------

    def _general(self, entry_ids: list[str], _pattern: dict) -> tuple[list[dict], int]:
        """One-hop exploration in both directions, no relationship filter."""
        cypher = f"""
            MATCH (src)-[r]-(tgt)
            WHERE elementId(src) IN $ids
            {_EDGE_RETURN}
            LIMIT {self._general_limit}
        """
        return self.connector.execute_query(cypher, {"ids": entry_ids}), 1

    # ------------------------------------------------------------------
    # Subgraph assembly  (shared by all strategies)
    # ------------------------------------------------------------------

    def _build_subgraph(self, rows: list[dict], entry_nodes: list[dict]) -> dict:
        """
        Deduplicate and assemble nodes + relationships from flat result rows.
        Entry nodes are always present even when traversal returns nothing.
        """
        seen_ids: set[str] = set()
        nodes:    list[dict] = []
        rels:     list[dict] = []

        for en in entry_nodes:
            if en["id"] not in seen_ids:
                seen_ids.add(en["id"])
                nodes.append(en)

        for row in rows:
            for side in ("source", "target"):
                nid = row.get(f"{side}_id")
                if nid and nid not in seen_ids:
                    seen_ids.add(nid)
                    props = row.get(f"{side}_props") or {}
                    nodes.append({
                        "id":         nid,
                        "label":      row.get(f"{side}_label") or "Unknown",
                        "name":       props.get("name", ""),
                        "properties": props,
                    })

            if row.get("source_id") and row.get("target_id"):
                rels.append({
                    "source_id": row["source_id"],
                    "target_id": row["target_id"],
                    "type":      row.get("rel_type", "RELATED_TO"),
                    "properties":row.get("rel_props") or {},
                })

        return {"nodes": nodes, "relationships": rels}
