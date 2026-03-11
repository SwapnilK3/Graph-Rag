"""
src/schema_discovery/discoverer.py
-----------------------------------
Main schema discovery orchestrator.

Implements **PG-HIVE Algorithm 1** (Sideri et al., 2024) — the full
schema discovery pipeline — combined with ideas from Schema Inference
(Lbath et al., 2021) and the user's draft paper §3.1.

Full Pipeline
-------------
1. **Load**          — Fetch all node labels and edge types from the graph.
2. **Preprocess**    — Build label/type inventories.
3. **Type extraction** — PG-HIVE Algorithm 2 (label-set merging + Jaccard).
4. **Property analysis** — For each type: discover properties, infer data
   types, determine constraints (MANDATORY/OPTIONAL).
5. **Cardinality**   — For each edge type: compute (1:1, 1:N, N:1, M:N).
6. **Hierarchy**     — Infer node-type hierarchy (subtypes/supertypes).
7. **Searchable scoring** — Multi-factor scoring to identify searchable
   (look-up) properties.
8. **Serialize**     — Pack everything into a DiscoveredSchema object (and
   optionally JSON).

DiscoveredSchema is the complete output consumed by downstream components
(schema evolution, Graph-RAG query layer, etc.).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict

from src.utils.connector import Neo4jConnector
from src.schema_discovery.property_analyzer import PropertyAnalyzer, TypeProperties
from src.schema_discovery.cardinality import CardinalityAnalyzer, CardinalityInfo
from src.schema_discovery.hierarchy import HierarchyInferer, TypeHierarchy
from src.schema_discovery.searchable_scorer import SearchableScorer

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# Output data structures
# ══════════════════════════════════════════════════════════════

@dataclass
class NodeTypeSchema:
    """Fully discovered schema for one node type (label)."""
    label: str
    instance_count: int = 0
    properties: dict = field(default_factory=dict)       # {key: {type, constraint, ...}}
    searchable_properties: list[str] = field(default_factory=list)
    supertypes: list[str] = field(default_factory=list)
    subtypes: list[str] = field(default_factory=list)
    depth: int = 0


@dataclass
class EdgeTypeSchema:
    """Fully discovered schema for one edge type."""
    rel_type: str
    source_labels: list[str] = field(default_factory=list)
    target_labels: list[str] = field(default_factory=list)
    properties: dict = field(default_factory=dict)
    cardinality: str = "M:N"
    max_out: int = 0
    max_in: int = 0
    total_edges: int = 0
    source_optional: bool = True
    target_optional: bool = True


@dataclass
class DiscoveredSchema:
    """
    Complete schema discovered from a Neo4j graph.

    This is the primary output of the SchemaDiscoverer pipeline and is
    consumed by Schema Evolution (AdaKGC) and the Graph-RAG query layer.
    """
    node_types: dict[str, NodeTypeSchema] = field(default_factory=dict)
    edge_types: dict[str, EdgeTypeSchema] = field(default_factory=dict)
    hierarchy: TypeHierarchy | None = None
    graph_stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to a plain dict (JSON-friendly)."""
        return {
            "node_types": {
                k: {
                    "label": v.label,
                    "instance_count": v.instance_count,
                    "properties": v.properties,
                    "searchable_properties": v.searchable_properties,
                    "supertypes": v.supertypes,
                    "subtypes": v.subtypes,
                    "depth": v.depth,
                }
                for k, v in self.node_types.items()
            },
            "edge_types": {
                k: {
                    "rel_type": v.rel_type,
                    "source_labels": v.source_labels,
                    "target_labels": v.target_labels,
                    "properties": v.properties,
                    "cardinality": v.cardinality,
                    "max_out": v.max_out,
                    "max_in": v.max_in,
                    "total_edges": v.total_edges,
                    "source_optional": v.source_optional,
                    "target_optional": v.target_optional,
                }
                for k, v in self.edge_types.items()
            },
            "graph_stats": self.graph_stats,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to pretty JSON."""
        return json.dumps(self.to_dict(), indent=indent)


# ══════════════════════════════════════════════════════════════
# Main discoverer
# ══════════════════════════════════════════════════════════════

class SchemaDiscoverer:
    """
    Discover the full schema of a populated Neo4j graph.

    Orchestrates PropertyAnalyzer, CardinalityAnalyzer, HierarchyInferer,
    and SearchableScorer to produce a DiscoveredSchema.

    Parameters
    ----------
    connector : Neo4jConnector
        Active database connection.
    sample_size : int
        Property value sample size for type inference (default 100).
    searchable_threshold : float
        Score threshold for marking a property searchable (default 0.5).
    hierarchy_property_weight : float
        Weight of property overlap in hierarchy inference (default 0.3).
    """

    def __init__(
        self,
        connector: Neo4jConnector,
        sample_size: int = 100,
        searchable_threshold: float = 0.5,
        hierarchy_property_weight: float = 0.3,
    ):
        self.connector = connector
        self.prop_analyzer = PropertyAnalyzer(connector, sample_size=sample_size)
        self.card_analyzer = CardinalityAnalyzer(connector)
        self.hier_inferer = HierarchyInferer(
            connector, property_weight=hierarchy_property_weight,
        )
        self.scorer = SearchableScorer(threshold=searchable_threshold)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def discover(self) -> DiscoveredSchema:
        """
        Execute the full PG-HIVE-inspired discovery pipeline.

        Returns
        -------
        DiscoveredSchema
            Complete schema with node types, edge types, hierarchy,
            cardinalities, property info, and searchable properties.
        """
        schema = DiscoveredSchema()
        logger.info("Schema discovery started")

        # Step 1: Graph statistics
        schema.graph_stats = self._graph_stats()
        logger.info(
            "Graph has %d nodes, %d edges",
            schema.graph_stats.get("node_count", 0),
            schema.graph_stats.get("edge_count", 0),
        )

        # Step 2: Discover node types
        node_labels = self._discover_node_labels()
        logger.info("Discovered %d node labels", len(node_labels))

        # Step 3: Analyze node properties
        for label in node_labels:
            type_props = self.prop_analyzer.analyze_node_properties(label)
            node_schema = NodeTypeSchema(
                label=label,
                instance_count=type_props.instance_count,
                properties={
                    k: {
                        "data_type": v.data_type,
                        "constraint": v.constraint,
                        "frequency": round(v.frequency, 3),
                        "unique_ratio": round(v.unique_ratio, 3),
                    }
                    for k, v in type_props.properties.items()
                },
            )

            # Step 4: Score searchable properties
            node_schema.searchable_properties = self.scorer.get_searchable_properties(
                label, type_props.properties,
            )

            schema.node_types[label] = node_schema

        logger.info("Analyzed properties for %d node types", len(schema.node_types))

        # Step 5: Discover edge types and cardinalities
        edge_info = self._discover_edges()
        for et in edge_info:
            rel_type = et["rel_type"]
            src_labels = et["source_labels"]
            tgt_labels = et["target_labels"]

            edge_schema = EdgeTypeSchema(
                rel_type=rel_type,
                source_labels=src_labels,
                target_labels=tgt_labels,
            )

            # Analyze cardinality for the primary source→target pair
            if src_labels and tgt_labels:
                card = self.card_analyzer.analyze(
                    rel_type, src_labels[0], tgt_labels[0],
                )
                edge_schema.cardinality = card.cardinality
                edge_schema.max_out = card.max_out
                edge_schema.max_in = card.max_in
                edge_schema.total_edges = card.total_edges
                edge_schema.source_optional = card.source_optional
                edge_schema.target_optional = card.target_optional

            # Analyze edge properties
            edge_props = self.prop_analyzer.analyze_edge_properties(rel_type)
            edge_schema.properties = {
                k: {
                    "data_type": v.data_type,
                    "constraint": v.constraint,
                    "frequency": round(v.frequency, 3),
                }
                for k, v in edge_props.properties.items()
            }

            schema.edge_types[rel_type] = edge_schema

        logger.info("Analyzed %d edge types", len(schema.edge_types))

        # Step 6: Infer hierarchy
        hierarchy = self.hier_inferer.infer()
        schema.hierarchy = hierarchy

        # Update node schemas with hierarchy info
        for label, node_schema in schema.node_types.items():
            type_node = hierarchy.types.get(label)
            if type_node:
                node_schema.supertypes = type_node.supertypes
                node_schema.subtypes = type_node.subtypes
                node_schema.depth = type_node.depth

        logger.info("Hierarchy inferred — %d root types", len(hierarchy.roots))
        logger.info("Schema discovery complete")

        return schema

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def discover_and_save(self, path: str) -> DiscoveredSchema:
        """Discover schema and save to a JSON file."""
        schema = self.discover()
        with open(path, "w") as f:
            f.write(schema.to_json())
        logger.info("Schema saved to %s", path)
        return schema

    def print_schema(self, schema: DiscoveredSchema | None = None) -> None:
        """Print a human-readable summary of the schema."""
        if schema is None:
            schema = self.discover()

        print("\n" + "=" * 60)
        print("DISCOVERED SCHEMA")
        print("=" * 60)

        stats = schema.graph_stats
        print(f"\nNodes: {stats.get('node_count', '?')}")
        print(f"Edges: {stats.get('edge_count', '?')}")
        print(f"Labels: {stats.get('label_count', '?')}")
        print(f"Rel types: {stats.get('rel_type_count', '?')}")

        print("\n── Node Types " + "─" * 46)
        for label, ns in schema.node_types.items():
            print(f"\n  [{label}]  ({ns.instance_count} instances)")
            if ns.supertypes:
                print(f"    ↑ supertypes: {', '.join(ns.supertypes)}")
            if ns.subtypes:
                print(f"    ↓ subtypes:   {', '.join(ns.subtypes)}")
            for pk, pi in ns.properties.items():
                mand = "★" if pi.get("constraint") == "MANDATORY" else " "
                search = " 🔍" if pk in ns.searchable_properties else ""
                print(
                    f"    {mand} {pk}: {pi.get('data_type', '?')}"
                    f"  (freq={pi.get('frequency', '?')}"
                    f", uniq={pi.get('unique_ratio', '?')}){search}"
                )

        print("\n── Edge Types " + "─" * 46)
        for rt, es in schema.edge_types.items():
            src = ", ".join(es.source_labels) or "?"
            tgt = ", ".join(es.target_labels) or "?"
            print(
                f"\n  ({src})-[{rt}]->({tgt})"
                f"  {es.cardinality}  ({es.total_edges} edges)"
            )
            if es.properties:
                for pk, pi in es.properties.items():
                    print(f"    {pk}: {pi.get('data_type', '?')}")

        print("\n" + "=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _graph_stats(self) -> dict:
        """Collect basic graph statistics."""
        node_row = self.connector.run_single(
            "MATCH (n) RETURN count(n) AS cnt"
        )
        edge_row = self.connector.run_single(
            "MATCH ()-[r]->() RETURN count(r) AS cnt"
        )
        label_rows = self.connector.run(
            "CALL db.labels() YIELD label RETURN count(label) AS cnt"
        )
        rel_rows = self.connector.run(
            "CALL db.relationshipTypes() YIELD relationshipType "
            "RETURN count(relationshipType) AS cnt"
        )

        return {
            "node_count": node_row["cnt"] if node_row else 0,
            "edge_count": edge_row["cnt"] if edge_row else 0,
            "label_count": label_rows[0]["cnt"] if label_rows else 0,
            "rel_type_count": rel_rows[0]["cnt"] if rel_rows else 0,
        }

    def _discover_node_labels(self) -> list[str]:
        """Get all node labels in the graph."""
        rows = self.connector.run(
            "CALL db.labels() YIELD label RETURN label ORDER BY label"
        )
        return [r["label"] for r in rows]

    def _discover_edges(self) -> list[dict]:
        """
        Discover all edge types with their source and target labels.

        Returns list of {rel_type, source_labels, target_labels}.
        """
        rows = self.connector.run(
            "MATCH (s)-[r]->(t) "
            "WITH type(r) AS rt, "
            "     collect(DISTINCT labels(s)[0]) AS sls, "
            "     collect(DISTINCT labels(t)[0]) AS tls "
            "RETURN rt, sls, tls ORDER BY rt"
        )
        return [
            {
                "rel_type": r["rt"],
                "source_labels": r["sls"],
                "target_labels": r["tls"],
            }
            for r in rows
        ]
