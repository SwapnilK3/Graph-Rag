"""
src/schema_discovery/hierarchy.py
----------------------------------
Node type hierarchy inference following:

**PG-HIVE (Sideri et al., 2024)** — §4.3:
  - Pairwise label-set comparison
  - If labels(A) ⊂ labels(B) → A is a subtype of B
  - If labels(A) ∩ labels(B) ≠ ∅ → potential shared supertype

**Schema Inference (Lbath et al., 2021)** — §3.4:
  - Node hierarchy from label inclusion
  - Kind-equivalence: two nodes are kind-equivalent if they share the
    same set of labels → same abstract type
  - Vertical hierarchy based on property overlap (subtype has all
    properties of supertype plus extras)
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
class TypeNode:
    """Represents a type in the hierarchy tree."""
    label: str
    label_set: frozenset[str] = field(default_factory=frozenset)
    property_keys: set[str] = field(default_factory=set)
    supertypes: list[str] = field(default_factory=list)
    subtypes: list[str] = field(default_factory=list)
    instance_count: int = 0
    depth: int = 0                # 0 = root level


@dataclass
class TypeHierarchy:
    """Complete type hierarchy for the graph."""
    types: dict[str, TypeNode] = field(default_factory=dict)
    roots: list[str] = field(default_factory=list)      # types with no supertypes

    def get_depth(self, label: str) -> int:
        """Get depth of a type in the hierarchy tree."""
        node = self.types.get(label)
        return node.depth if node else -1

    def get_subtypes(self, label: str, recursive: bool = False) -> list[str]:
        """Get direct (or recursive) subtypes of a label."""
        node = self.types.get(label)
        if not node:
            return []
        if not recursive:
            return list(node.subtypes)
        result = []
        stack = list(node.subtypes)
        while stack:
            sub = stack.pop()
            result.append(sub)
            sub_node = self.types.get(sub)
            if sub_node:
                stack.extend(sub_node.subtypes)
        return result

    def get_supertypes(self, label: str, recursive: bool = False) -> list[str]:
        """Get direct (or recursive) supertypes of a label."""
        node = self.types.get(label)
        if not node:
            return []
        if not recursive:
            return list(node.supertypes)
        result = []
        stack = list(node.supertypes)
        while stack:
            sup = stack.pop()
            result.append(sup)
            sup_node = self.types.get(sup)
            if sup_node:
                stack.extend(sup_node.supertypes)
        return result


# ──────────────────────────────────────────────────────────────
# Hierarchy inference
# ──────────────────────────────────────────────────────────────

class HierarchyInferer:
    """
    Infer a type hierarchy from the Neo4j graph.

    Two complementary strategies (per PG-HIVE and Lbath):

    1. **Label-based hierarchy**: If all nodes labelled A also carry
       label B (but not all B have A), then A is a subtype of B.

    2. **Property-based hierarchy**: If the property set of A is a
       strict superset of that of B, A may be a subtype of B
       (more specific = more properties).

    Parameters
    ----------
    connector : Neo4jConnector
    property_weight : float
        Relative weight of property-overlap signal (0 = label-only).
    min_overlap : float
        Minimum Jaccard overlap of properties to consider a relationship.
    """

    def __init__(
        self,
        connector: Neo4jConnector,
        property_weight: float = 0.3,
        min_overlap: float = 0.7,
    ):
        self.connector = connector
        self.property_weight = property_weight
        self.min_overlap = min_overlap

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def infer(self) -> TypeHierarchy:
        """
        Infer the full type hierarchy from the graph.

        Returns a TypeHierarchy with parent-child relationships.
        """
        hierarchy = TypeHierarchy()

        # Step 1: Collect all labels and their multi-label profiles
        label_profiles = self._collect_label_profiles()

        # Step 2: Collect property keys per label
        prop_profiles = self._collect_property_profiles()

        # Step 3: Build TypeNode for every label
        for label, info in label_profiles.items():
            node = TypeNode(
                label=label,
                label_set=info["label_set"],
                property_keys=prop_profiles.get(label, set()),
                instance_count=info["count"],
            )
            hierarchy.types[label] = node

        # Step 4: Label-based subtyping (PG-HIVE §4.3)
        self._infer_from_labels(hierarchy, label_profiles)

        # Step 5: Property-based subtyping (Lbath §3.4)
        if self.property_weight > 0:
            self._infer_from_properties(hierarchy)

        # Step 6: Compute depths and find roots
        self._compute_depths(hierarchy)

        return hierarchy

    # ------------------------------------------------------------------
    # Label profile collection
    # ------------------------------------------------------------------

    def _collect_label_profiles(self) -> dict[str, dict]:
        """
        For each label, find how many nodes carry it and what
        co-occurring labels exist.

        Returns {label: {"count": int, "label_set": frozenset, "co_labels": {label: count}}}
        """
        # All distinct label combinations
        rows = self.connector.run(
            "MATCH (n) "
            "RETURN labels(n) AS lbls, count(*) AS cnt"
        )

        label_stats: dict[str, dict] = {}

        for row in rows:
            labels = row["lbls"]
            count = row["cnt"]
            label_fs = frozenset(labels)

            for lbl in labels:
                if lbl not in label_stats:
                    label_stats[lbl] = {
                        "count": 0,
                        "label_set": frozenset(),
                        "co_labels": {},
                    }
                label_stats[lbl]["count"] += count
                label_stats[lbl]["label_set"] = frozenset(
                    label_stats[lbl]["label_set"] | label_fs
                )
                for other in labels:
                    if other != lbl:
                        label_stats[lbl]["co_labels"][other] = (
                            label_stats[lbl]["co_labels"].get(other, 0) + count
                        )

        return label_stats

    def _collect_property_profiles(self) -> dict[str, set[str]]:
        """For each label, discover property keys used by nodes with that label."""
        rows = self.connector.run(
            "MATCH (n) "
            "WITH labels(n) AS lbls, keys(n) AS ks "
            "UNWIND lbls AS lbl "
            "UNWIND ks AS k "
            "RETURN DISTINCT lbl, collect(DISTINCT k) AS props"
        )
        return {r["lbl"]: set(r["props"]) for r in rows}

    # ------------------------------------------------------------------
    # Label-based inference (PG-HIVE §4.3)
    # ------------------------------------------------------------------

    def _infer_from_labels(self, hierarchy: TypeHierarchy, profiles: dict):
        """
        If ALL nodes labelled A also have label B, but not all B have
        label A, then A is a **subtype** of B.

        This means A is more specific (carries extra label) → subtype.
        """
        labels = list(profiles.keys())

        for a in labels:
            a_count = profiles[a]["count"]
            co_labels = profiles[a].get("co_labels", {})

            for b in labels:
                if a == b:
                    continue
                # How many A-nodes also carry label B?
                a_with_b = co_labels.get(b, 0)

                if a_with_b == a_count:
                    # Every A also has B
                    b_count = profiles[b]["count"]
                    b_with_a = profiles[b].get("co_labels", {}).get(a, 0)

                    if b_with_a < b_count:
                        # B has nodes without A → A is subtype of B
                        a_node = hierarchy.types.get(a)
                        b_node = hierarchy.types.get(b)
                        if a_node and b_node:
                            if b not in a_node.supertypes:
                                a_node.supertypes.append(b)
                            if a not in b_node.subtypes:
                                b_node.subtypes.append(a)

    # ------------------------------------------------------------------
    # Property-based inference (Lbath §3.4)
    # ------------------------------------------------------------------

    def _infer_from_properties(self, hierarchy: TypeHierarchy):
        """
        If the property set of A is a strict superset of B and
        no label-based relationship already exists, consider A a subtype
        of B (more specific = more properties).
        """
        labels = list(hierarchy.types.keys())

        for a in labels:
            a_node = hierarchy.types[a]
            if not a_node.property_keys:
                continue

            for b in labels:
                if a == b:
                    continue
                # Skip if label-based relationship already established
                if b in a_node.supertypes or b in a_node.subtypes:
                    continue

                b_node = hierarchy.types[b]
                if not b_node.property_keys:
                    continue

                # Jaccard overlap
                intersection = a_node.property_keys & b_node.property_keys
                union = a_node.property_keys | b_node.property_keys
                if not union:
                    continue
                jaccard = len(intersection) / len(union)

                if jaccard < self.min_overlap:
                    continue

                # A's properties are a strict superset of B → A is subtype of B
                if a_node.property_keys > b_node.property_keys:
                    if b not in a_node.supertypes:
                        a_node.supertypes.append(b)
                    if a not in b_node.subtypes:
                        b_node.subtypes.append(a)

    # ------------------------------------------------------------------
    # Depth computation
    # ------------------------------------------------------------------

    def _compute_depths(self, hierarchy: TypeHierarchy):
        """Compute depth of each node (roots = 0) and identify roots."""
        # Find roots: types with no supertypes
        for label, node in hierarchy.types.items():
            if not node.supertypes:
                hierarchy.roots.append(label)
                node.depth = 0

        # BFS from roots to assign depths
        visited = set()
        queue = [(r, 0) for r in hierarchy.roots]

        while queue:
            label, depth = queue.pop(0)
            if label in visited:
                continue
            visited.add(label)

            node = hierarchy.types.get(label)
            if node:
                node.depth = depth
                for sub in node.subtypes:
                    queue.append((sub, depth + 1))
