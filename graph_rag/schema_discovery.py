"""
schema_discovery.py
-------------------
Component 0: Automatic Graph Schema Discovery

Connects to any Neo4j database and discovers its complete schema:
  - Node labels
  - Relationship types (with source/target label pairs)
  - Properties per label (with types, cardinality, sample values)
  - Searchable property ranking via multi-factor scoring

This is the key innovation: the system becomes schema-agnostic.
No one has to manually describe the graph — the system figures it out.

Algorithm (Multi-Factor Scoring for Searchable Properties):
-----------------------------------------------------------
For each property on each label, compute:
  +0.40  if property name matches a known name pattern (name, title, label, …)
  +0.30  if property has high cardinality (>50% unique values)
  +0.30  if property is a string type
Properties scoring >0.5 are marked as searchable.
"""

from __future__ import annotations

import logging
from typing import Optional

from connector import GraphDBConnector

logger = logging.getLogger(__name__)

# Property names that are very likely entity identifiers.
_NAME_PATTERNS: frozenset[str] = frozenset({
    "name", "title", "label", "display_name", "displayname",
    "full_name", "fullname", "common_name", "commonname",
    "brand_name", "brandname", "generic_name", "genericname",
    "short_name", "shortname", "identifier", "id_name",
    "heading", "subject", "description", "summary",
})


class SchemaDiscovery:
    """
    Automatically discovers the full schema of a Neo4j knowledge graph.

    Usage
    -----
        discovery = SchemaDiscovery(connector)
        schema = discovery.discover()

    Returns a dict:
        {
            "node_labels":    ["Drug", "Disease", ...],
            "relationships":  [
                {"type": "TREATS", "source_label": "Drug", "target_label": "Disease", "count": 5},
                ...
            ],
            "properties":     {
                "Drug":    [{"name": "name", "type": "String", "unique_ratio": 1.0, "sample": ["Aspirin"]}, ...],
                "Disease": [...],
            },
            "searchable_properties": {
                "Drug":    ["name", "generic_name"],
                "Disease": ["name"],
            },
            "node_counts":  {"Drug": 8, "Disease": 6, ...},
            "total_nodes":  32,
            "total_relationships": 37
        }
    """

    def __init__(self, connector: GraphDBConnector):
        self.connector = connector

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def discover(self) -> dict:
        """
        Run full schema discovery.  Returns a comprehensive schema dict.
        """
        logger.info("Starting automatic schema discovery …")

        labels      = self._discover_labels()
        node_counts = self._count_nodes(labels)
        rels        = self._discover_relationships()
        properties  = self._discover_properties(labels)
        searchable  = self._identify_searchable_properties(labels, properties)

        total_nodes = sum(node_counts.values())
        total_rels  = sum(r["count"] for r in rels)

        schema = {
            "node_labels":           labels,
            "relationships":         rels,
            "properties":            properties,
            "searchable_properties": searchable,
            "node_counts":           node_counts,
            "total_nodes":           total_nodes,
            "total_relationships":   total_rels,
        }

        logger.info(
            "Schema discovery complete: %d labels, %d relationship types, "
            "%d total nodes, %d total relationships",
            len(labels), len(rels), total_nodes, total_rels,
        )
        return schema

    # ------------------------------------------------------------------
    # Step 1: Discover node labels
    # ------------------------------------------------------------------

    def _discover_labels(self) -> list[str]:
        """Return all node labels present in the graph."""
        rows = self.connector.execute_query("CALL db.labels() YIELD label RETURN label")
        labels = sorted(row["label"] for row in rows)
        logger.debug("Discovered labels: %s", labels)
        return labels

    # ------------------------------------------------------------------
    # Step 2: Count nodes per label
    # ------------------------------------------------------------------

    def _count_nodes(self, labels: list[str]) -> dict[str, int]:
        """Return {label: count} for each label."""
        counts = {}
        for label in labels:
            rows = self.connector.execute_query(
                f"MATCH (n:`{label}`) RETURN count(n) AS cnt"
            )
            counts[label] = rows[0]["cnt"] if rows else 0
        return counts

    # ------------------------------------------------------------------
    # Step 3: Discover relationship types with source/target labels
    # ------------------------------------------------------------------

    def _discover_relationships(self) -> list[dict]:
        """
        Discover all relationship types and which label pairs they connect.

        Returns a list of dicts:
            {"type": "TREATS", "source_label": "Drug", "target_label": "Disease", "count": 5}
        """
        cypher = """
            MATCH (a)-[r]->(b)
            RETURN
                type(r)      AS rel_type,
                labels(a)[0] AS source_label,
                labels(b)[0] AS target_label,
                count(*)     AS cnt
            ORDER BY cnt DESC
        """
        rows = self.connector.execute_query(cypher)
        rels = []
        for row in rows:
            rels.append({
                "type":         row["rel_type"],
                "source_label": row["source_label"],
                "target_label": row["target_label"],
                "count":        row["cnt"],
            })
        logger.debug("Discovered %d relationship patterns", len(rels))
        return rels

    # ------------------------------------------------------------------
    # Step 4: Discover properties per label
    # ------------------------------------------------------------------

    def _discover_properties(self, labels: list[str]) -> dict[str, list[dict]]:
        """
        For each label, discover all properties with:
          - type (String, Integer, Float, Boolean, List, Unknown)
          - unique_ratio (distinct_values / total_nodes)
          - sample values (up to 3)
        """
        all_props: dict[str, list[dict]] = {}

        for label in labels:
            # Get all property keys used by nodes of this label
            key_rows = self.connector.execute_query(f"""
                MATCH (n:`{label}`)
                UNWIND keys(n) AS key
                RETURN DISTINCT key
                ORDER BY key
            """)
            prop_keys = [r["key"] for r in key_rows]

            label_props = []
            for prop_key in prop_keys:
                info = self._analyze_property(label, prop_key)
                label_props.append(info)

            all_props[label] = label_props

        return all_props

    def _analyze_property(self, label: str, prop_key: str) -> dict:
        """Analyze a single property: type, cardinality, samples."""
        # Use backtick quoting in case prop_key has special chars
        escaped_prop = f"`{prop_key}`"

        # Get total count, distinct count, and samples in one query
        rows = self.connector.execute_query(f"""
            MATCH (n:`{label}`)
            WHERE n.{escaped_prop} IS NOT NULL
            WITH
                count(n) AS total,
                count(DISTINCT n.{escaped_prop}) AS distinct_count,
                collect(DISTINCT n.{escaped_prop})[0..3] AS samples
            RETURN total, distinct_count, samples
        """)

        if not rows or rows[0]["total"] == 0:
            return {
                "name": prop_key,
                "type": "Unknown",
                "unique_ratio": 0.0,
                "sample": [],
            }

        total    = rows[0]["total"]
        distinct = rows[0]["distinct_count"]
        samples  = rows[0]["samples"] or []

        # Infer type from first non-null sample
        prop_type = self._infer_type(samples[0] if samples else None)

        return {
            "name":         prop_key,
            "type":         prop_type,
            "unique_ratio": distinct / total if total > 0 else 0.0,
            "sample":       [str(s) for s in samples],
        }

    @staticmethod
    def _infer_type(value) -> str:
        """Infer property type from a sample value."""
        if value is None:
            return "Unknown"
        if isinstance(value, bool):
            return "Boolean"
        if isinstance(value, int):
            return "Integer"
        if isinstance(value, float):
            return "Float"
        if isinstance(value, str):
            return "String"
        if isinstance(value, list):
            return "List"
        return "Unknown"

    # ------------------------------------------------------------------
    # Step 5: Multi-factor scoring for searchable properties
    # ------------------------------------------------------------------

    def _identify_searchable_properties(
        self,
        labels: list[str],
        properties: dict[str, list[dict]],
    ) -> dict[str, list[str]]:
        """
        For each label, score every property and keep those scoring > 0.5.

        Scoring:
          +0.40  Name pattern match (property name is "name", "title", etc.)
          +0.30  High cardinality (unique_ratio > 0.5)
          +0.30  String type

        This is the multi-factor scoring algorithm from the research paper.
        """
        searchable: dict[str, list[str]] = {}

        for label in labels:
            label_searchable = []
            for prop in properties.get(label, []):
                score = 0.0

                # Check 1: Name pattern
                if prop["name"].lower().replace("_", "").replace("-", "") in _NAME_PATTERNS or \
                   prop["name"].lower() in _NAME_PATTERNS:
                    score += 0.40

                # Check 2: Cardinality (>50% unique)
                if prop["unique_ratio"] > 0.5:
                    score += 0.30

                # Check 3: String type
                if prop["type"] == "String":
                    score += 0.30

                logger.debug(
                    "  %s.%s: score=%.2f (pattern=%s, cardinality=%.2f, type=%s)",
                    label, prop["name"], score,
                    prop["name"].lower() in _NAME_PATTERNS,
                    prop["unique_ratio"], prop["type"],
                )

                if score > 0.5:
                    label_searchable.append(prop["name"])

            # Fallback: if nothing scored high enough, use any string property
            if not label_searchable:
                for prop in properties.get(label, []):
                    if prop["type"] == "String":
                        label_searchable.append(prop["name"])
                        break

            searchable[label] = label_searchable

        return searchable

    # ------------------------------------------------------------------
    # Pretty print (for debugging / demo)
    # ------------------------------------------------------------------

    def print_schema(self, schema: dict) -> None:
        """Human-readable schema summary."""
        print(f"\n{'═' * 60}")
        print(f"  GRAPH SCHEMA DISCOVERY REPORT")
        print(f"  {schema['total_nodes']} nodes, {schema['total_relationships']} relationships")
        print(f"{'═' * 60}")

        print(f"\n  NODE LABELS ({len(schema['node_labels'])}):")
        for label in schema["node_labels"]:
            count = schema["node_counts"].get(label, 0)
            props = schema["properties"].get(label, [])
            searchable = schema["searchable_properties"].get(label, [])
            print(f"    [{label}] — {count} nodes")
            for p in props:
                marker = " ★" if p["name"] in searchable else ""
                print(f"      .{p['name']} ({p['type']}, {p['unique_ratio']:.0%} unique){marker}")

        print(f"\n  RELATIONSHIPS ({len(schema['relationships'])}):")
        for rel in schema["relationships"]:
            print(f"    ({rel['source_label']})-[:{rel['type']}]->({rel['target_label']})  ×{rel['count']}")

        print(f"\n  SEARCHABLE PROPERTIES (★):")
        for label, props in schema["searchable_properties"].items():
            if props:
                print(f"    {label}: {', '.join(props)}")

        print(f"{'═' * 60}\n")
