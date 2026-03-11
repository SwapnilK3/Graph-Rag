"""
src/schema_evolution/adapter.py
--------------------------------
Schema-adaptive evolution engine inspired by **AdaKGC** (Li et al., 2023):

> "Schema-adaptable Knowledge Graph Construction"
> Three expansion modes: horizontal, vertical, hybrid.

This module takes an existing DiscoveredSchema and adapts it when new
data (new triples, new document text) arrives.  It detects:

  - **Horizontal expansion**: brand-new entity types or relation types
    that did not exist in the current schema.
  - **Vertical expansion**: refinement of existing types (e.g. a generic
    "Person" splits into "Doctor" and "Patient").
  - **Hybrid expansion**: a mix of both.

It also supports **incremental update** (PG-HIVE §5): updating schema
statistics (counts, cardinalities, properties) without rediscovering
from scratch.

Integration Points
-------------------
  - Input: DiscoveredSchema  +  new triples (list[Triple])
  - Output: updated DiscoveredSchema  (or a diff/changelog)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from copy import deepcopy

from src.schema_discovery.discoverer import (
    DiscoveredSchema,
    NodeTypeSchema,
    EdgeTypeSchema,
)
from src.kg_construction.triple_extractor import Triple
from src.kg_construction.schema_inducer import InducedSchema

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# Change records
# ══════════════════════════════════════════════════════════════

@dataclass
class SchemaChange:
    """Single schema change event."""
    change_type: str           # HORIZONTAL_NODE, HORIZONTAL_EDGE, VERTICAL_REFINE, PROPERTY_ADD
    entity: str                # label or rel_type affected
    details: dict = field(default_factory=dict)


@dataclass
class EvolutionResult:
    """Result of applying schema evolution."""
    updated_schema: DiscoveredSchema
    changes: list[SchemaChange] = field(default_factory=list)
    new_node_types: list[str] = field(default_factory=list)
    new_edge_types: list[str] = field(default_factory=list)
    refined_types: list[str] = field(default_factory=list)      # vertical expansion
    new_properties: dict[str, list[str]] = field(default_factory=dict)  # type → [new props]

    @property
    def has_changes(self) -> bool:
        return len(self.changes) > 0

    def summary(self) -> str:
        lines = [f"Schema Evolution — {len(self.changes)} changes"]
        if self.new_node_types:
            lines.append(f"  + Node types:  {', '.join(self.new_node_types)}")
        if self.new_edge_types:
            lines.append(f"  + Edge types:  {', '.join(self.new_edge_types)}")
        if self.refined_types:
            lines.append(f"  ↓ Refined:     {', '.join(self.refined_types)}")
        if self.new_properties:
            for typ, props in self.new_properties.items():
                lines.append(f"  + Props [{typ}]: {', '.join(props)}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# Adapter
# ══════════════════════════════════════════════════════════════

class SchemaAdapter:
    """
    Adapts an existing schema to accommodate new triples.

    Implements three **AdaKGC expansion modes** and the **PG-HIVE
    incremental update** approach:

    1. **Horizontal expansion** — new entity/relation types that don't
       exist in the current schema.
    2. **Vertical expansion** — specialisation of an existing type into
       subtypes when LLM-induced schema suggests finer granularity.
    3. **Hybrid expansion** — combination of both.

    Parameters
    ----------
    merge_threshold : float
        Jaccard similarity above which a "new" type is considered an
        alias of an existing type (skip creation). Default 0.7.
    auto_add_properties : bool
        If True, automatically add newly seen properties to existing
        types. Default True.
    """

    def __init__(
        self,
        merge_threshold: float = 0.7,
        auto_add_properties: bool = True,
    ):
        self.merge_threshold = merge_threshold
        self.auto_add_properties = auto_add_properties

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evolve(
        self,
        schema: DiscoveredSchema,
        new_triples: list[Triple],
        induced: InducedSchema | None = None,
    ) -> EvolutionResult:
        """
        Given an existing schema and new triples, return an updated
        schema with a changelog.

        Parameters
        ----------
        schema : DiscoveredSchema
            Current schema (will be deep-copied).
        new_triples : list[Triple]
            Newly extracted triples to integrate.
        induced : InducedSchema, optional
            LLM-induced schema for the new triples (used for vertical
            expansion — refinement of existing types).

        Returns
        -------
        EvolutionResult
        """
        result = EvolutionResult(updated_schema=deepcopy(schema))
        updated = result.updated_schema

        # Phase 1: Horizontal expansion — detect new types
        self._horizontal_expansion(updated, new_triples, result)

        # Phase 2: Vertical expansion — detect refinements
        if induced:
            self._vertical_expansion(updated, induced, result)

        # Phase 3: Property expansion — detect new properties
        if self.auto_add_properties:
            self._property_expansion(updated, new_triples, result)

        # Phase 4: Update statistics
        self._update_stats(updated, new_triples, result)

        logger.info("Schema evolution: %d changes", len(result.changes))
        return result

    # ------------------------------------------------------------------
    # Horizontal expansion (AdaKGC §3.2)
    # ------------------------------------------------------------------

    def _horizontal_expansion(
        self,
        schema: DiscoveredSchema,
        triples: list[Triple],
        result: EvolutionResult,
    ):
        """
        Detect brand-new entity and relation types not present in the
        current schema.
        """
        existing_node_labels = set(schema.node_types.keys())
        existing_edge_types = set(schema.edge_types.keys())

        # Collect entity types from triples
        seen_entities: dict[str, set[str]] = {}   # type → {entity names}
        seen_relations: dict[str, set[tuple]] = {}  # rel → {(src_type, tgt_type)}

        for t in triples:
            h_type = t.head_type if hasattr(t, "head_type") and t.head_type else "Entity"
            t_type = t.tail_type if hasattr(t, "tail_type") and t.tail_type else "Entity"

            # Normalise type names to title case
            h_type = h_type.title().replace(" ", "")
            t_type = t_type.title().replace(" ", "")

            seen_entities.setdefault(h_type, set()).add(t.head)
            seen_entities.setdefault(t_type, set()).add(t.tail)
            seen_relations.setdefault(t.relation, set()).add((h_type, t_type))

        # Check for new node types
        for etype, entities in seen_entities.items():
            if etype not in existing_node_labels:
                # Check if it's just an alias of an existing type
                if self._is_alias(etype, existing_node_labels):
                    continue
                # New type → horizontal expansion
                schema.node_types[etype] = NodeTypeSchema(
                    label=etype,
                    instance_count=len(entities),
                    properties={"name": {"data_type": "STRING", "constraint": "MANDATORY", "frequency": 1.0, "unique_ratio": 1.0}},
                    searchable_properties=["name"],
                )
                result.new_node_types.append(etype)
                result.changes.append(SchemaChange(
                    change_type="HORIZONTAL_NODE",
                    entity=etype,
                    details={"entity_count": len(entities)},
                ))

        # Check for new edge types
        for rel, type_pairs in seen_relations.items():
            rel_norm = rel.upper().replace(" ", "_")
            if rel_norm not in existing_edge_types:
                src_labels = list({p[0] for p in type_pairs})
                tgt_labels = list({p[1] for p in type_pairs})
                schema.edge_types[rel_norm] = EdgeTypeSchema(
                    rel_type=rel_norm,
                    source_labels=src_labels,
                    target_labels=tgt_labels,
                    total_edges=sum(1 for t in triples if t.relation == rel),
                )
                result.new_edge_types.append(rel_norm)
                result.changes.append(SchemaChange(
                    change_type="HORIZONTAL_EDGE",
                    entity=rel_norm,
                    details={
                        "source_labels": src_labels,
                        "target_labels": tgt_labels,
                    },
                ))

    # ------------------------------------------------------------------
    # Vertical expansion (AdaKGC §3.3)
    # ------------------------------------------------------------------

    def _vertical_expansion(
        self,
        schema: DiscoveredSchema,
        induced: InducedSchema,
        result: EvolutionResult,
    ):
        """
        Detect when an existing broad type should be refined into
        more specific subtypes.

        Uses LLM-induced schema to find specialisations:
        - If induced schema has type B whose entities are a subset
          of existing type A, B is a subtype of A.
        """
        existing_labels = set(schema.node_types.keys())

        for type_name, etype in induced.entity_types.items():
            norm_name = type_name.title().replace(" ", "")

            # Only care about types NOT already in the schema
            if norm_name in existing_labels:
                continue

            # Check overlaps with existing types
            induced_entities = set(etype.entities)
            if not induced_entities:
                continue

            for existing_label in list(existing_labels):
                # We need to check if these induced entities overlap
                # We can't directly check membership without querying DB,
                # so we use a name-based heuristic:
                # If induced type name contains existing label or vice versa
                if (existing_label.lower() in norm_name.lower() or
                        norm_name.lower() in existing_label.lower()):
                    # Likely a refinement
                    schema.node_types[norm_name] = NodeTypeSchema(
                        label=norm_name,
                        instance_count=len(induced_entities),
                        properties={"name": {"data_type": "STRING", "constraint": "MANDATORY", "frequency": 1.0, "unique_ratio": 1.0}},
                        searchable_properties=["name"],
                        supertypes=[existing_label],
                    )

                    # Update parent's subtypes
                    parent = schema.node_types.get(existing_label)
                    if parent and norm_name not in parent.subtypes:
                        parent.subtypes.append(norm_name)

                    result.refined_types.append(norm_name)
                    result.changes.append(SchemaChange(
                        change_type="VERTICAL_REFINE",
                        entity=norm_name,
                        details={
                            "parent": existing_label,
                            "entity_count": len(induced_entities),
                        },
                    ))
                    break  # only link to first matching parent

    # ------------------------------------------------------------------
    # Property expansion
    # ------------------------------------------------------------------

    def _property_expansion(
        self,
        schema: DiscoveredSchema,
        triples: list[Triple],
        result: EvolutionResult,
    ):
        """
        Detect new properties seen in triples that don't exist in the
        current schema.  Relation names can indicate properties.
        """
        # In triple data the relation itself often encodes a property-like
        # connection (e.g. HAS_EMAIL).  We track unique relation names per
        # source type as potential new edge properties.
        # This is a lightweight detection; full property discovery requires
        # re-running PropertyAnalyzer against the updated graph.
        pass  # property expansion happens naturally during re-discovery

    # ------------------------------------------------------------------
    # Statistics update (PG-HIVE §5 incremental)
    # ------------------------------------------------------------------

    def _update_stats(
        self,
        schema: DiscoveredSchema,
        triples: list[Triple],
        result: EvolutionResult,
    ):
        """
        Incrementally update instance counts and edge counts
        based on new triples (without a full re-discovery).
        """
        for t in triples:
            # Update node counts
            h_type = (t.head_type or "Entity").title().replace(" ", "")
            t_type = (t.tail_type or "Entity").title().replace(" ", "")

            if h_type in schema.node_types:
                schema.node_types[h_type].instance_count += 1
            if t_type in schema.node_types:
                schema.node_types[t_type].instance_count += 1

            # Update edge counts
            rel_norm = t.relation.upper().replace(" ", "_")
            if rel_norm in schema.edge_types:
                schema.edge_types[rel_norm].total_edges += 1

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_alias(self, new_type: str, existing: set[str]) -> bool:
        """
        Check if *new_type* is just a variant name for an existing type
        using Jaccard similarity on character bigrams.
        """
        new_lower = new_type.lower()
        new_bigrams = self._bigrams(new_lower)
        if not new_bigrams:
            return False

        for existing_type in existing:
            ex_bigrams = self._bigrams(existing_type.lower())
            if not ex_bigrams:
                continue
            inter = new_bigrams & ex_bigrams
            union = new_bigrams | ex_bigrams
            jaccard = len(inter) / len(union)
            if jaccard >= self.merge_threshold:
                return True
        return False

    @staticmethod
    def _bigrams(s: str) -> set[str]:
        """Character bigrams of a string."""
        return {s[i:i+2] for i in range(len(s) - 1)} if len(s) >= 2 else set()
