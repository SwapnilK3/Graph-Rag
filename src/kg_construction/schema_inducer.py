"""
src/kg_construction/schema_inducer.py
-------------------------------------
Automatic schema induction from extracted triples.

**Paper basis**: AutoSchemaKG (Bai et al., 2025)
  - Schema induction from extracted triples using LLM
  - Two-level type system: entity types & event types
  - 92% semantic alignment with human-defined schemas
  - Batch processing (Bs=5 triples per induction batch)

**Paper basis**: PG-HIVE (Sideri et al., 2024)
  - Clustering-based type discovery
  - LSH for scalable similarity computation
  - Jaccard-based merging of similar types

Algorithm (from AutoSchemaKG)
-----------------------------
1. Collect all unique entities and relations from triples
2. Group entities by co-occurrence patterns
3. Use LLM to induce semantic types for entity groups
4. Build type hierarchy (subtypes/supertypes)
5. Create relation type constraints (domain → range)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from collections import defaultdict

from src.kg_construction.triple_extractor import Triple
from src.utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────

@dataclass
class EntityType:
    """An induced entity or event type."""
    name: str                          # e.g. "Drug", "Disease"
    description: str = ""
    entities: list[str] = field(default_factory=list)
    properties: dict = field(default_factory=dict)
    is_event_type: bool = False
    parent_type: str | None = None     # for hierarchy


@dataclass
class RelationType:
    """An induced relation type with domain/range constraints."""
    name: str                          # e.g. "TREATS"
    description: str = ""
    source_types: list[str] = field(default_factory=list)
    target_types: list[str] = field(default_factory=list)
    count: int = 0


@dataclass
class InducedSchema:
    """Complete schema induced from triples."""
    entity_types: dict[str, EntityType] = field(default_factory=dict)
    relation_types: dict[str, RelationType] = field(default_factory=dict)
    type_hierarchy: dict[str, list[str]] = field(default_factory=dict)  # parent → children


# ──────────────────────────────────────────────────────────────
# Schema induction prompts
# ──────────────────────────────────────────────────────────────

_INDUCTION_SYSTEM = """\
You are a knowledge graph schema expert.
Given a set of entity names and their relationships, induce semantic types.

Rules:
- Group similar entities under a common type.
- Use singular nouns as type names (Drug, Disease, not Drugs).
- Types should be meaningful and domain-appropriate.
- Create a hierarchy when logical (e.g., Medication → Drug).
"""

_TYPE_INDUCTION_PROMPT = """\
Given these entities and their relationships, induce the entity types.

ENTITIES:
{entities}

RELATIONS (head → relation → tail):
{relations}

Return a JSON object:
{{
  "types": [
    {{
      "name": "TypeName",
      "description": "Brief description",
      "entities": ["entity1", "entity2", ...],
      "is_event_type": false,
      "parent_type": null or "ParentTypeName"
    }}
  ],
  "relation_types": [
    {{
      "name": "RELATION_NAME",
      "description": "Brief description",
      "source_types": ["SourceType"],
      "target_types": ["TargetType"]
    }}
  ]
}}

Return ONLY the JSON, no other text.
"""


# ──────────────────────────────────────────────────────────────
# Main inducer
# ──────────────────────────────────────────────────────────────

class SchemaInducer:
    """
    Induces a knowledge graph schema from extracted triples.

    Implements AutoSchemaKG's schema induction with PG-HIVE-style
    similarity-based merging.

    Parameters
    ----------
    llm : LLMClient
        LLM for semantic type induction.
    batch_size : int
        Number of triples to batch for induction (Bs from AutoSchemaKG).
    merge_threshold : float
        Jaccard similarity threshold for merging types (θ from PG-HIVE).
    """

    def __init__(
        self,
        llm: LLMClient,
        batch_size: int = 50,
        merge_threshold: float = 0.7,
    ):
        self.llm = llm
        self.batch_size = batch_size
        self.merge_threshold = merge_threshold

    def induce(self, triples: list[Triple]) -> InducedSchema:
        """
        Induce schema from a list of extracted triples.

        Steps (AutoSchemaKG + PG-HIVE):
        1. Collect unique entities and relations
        2. Build co-occurrence matrix
        3. Use LLM for semantic type induction
        4. Merge similar types (Jaccard threshold)
        5. Build relation constraints
        """
        if not triples:
            return InducedSchema()

        # Step 1: Collect unique entities and relations
        entities, relations = self._collect_unique(triples)
        logger.info(
            "Collected %d unique entities, %d unique relations",
            len(entities), len(relations),
        )

        # Step 2: Build co-occurrence profiles
        profiles = self._build_profiles(triples)

        # Step 3: LLM-based type induction
        schema = self._llm_induce(entities, triples)

        # Step 4: Merge similar types (PG-HIVE Jaccard merging)
        schema = self._merge_similar_types(schema)

        # Step 5: Rebuild relation constraints from schema
        schema = self._build_relation_constraints(schema, triples)

        logger.info(
            "Schema induced: %d entity types, %d relation types",
            len(schema.entity_types), len(schema.relation_types),
        )
        return schema

    # ------------------------------------------------------------------
    # Step 1: Collect unique entities and relations
    # ------------------------------------------------------------------

    def _collect_unique(self, triples: list[Triple]) -> tuple[set[str], set[str]]:
        entities = set()
        relations = set()
        for t in triples:
            entities.add(t.head)
            entities.add(t.tail)
            relations.add(t.relation)
        return entities, relations

    # ------------------------------------------------------------------
    # Step 2: Build co-occurrence profiles
    # ------------------------------------------------------------------

    def _build_profiles(self, triples: list[Triple]) -> dict[str, dict]:
        """
        Build a profile for each entity: what relations it participates in
        (as head/tail) and what types it co-occurs with.

        From PG-HIVE: entities sharing similar relational profiles
        belong to the same type.
        """
        profiles: dict[str, dict] = defaultdict(
            lambda: {"as_head": set(), "as_tail": set(), "neighbors": set()}
        )
        for t in triples:
            profiles[t.head]["as_head"].add(t.relation)
            profiles[t.head]["neighbors"].add(t.tail)
            profiles[t.tail]["as_tail"].add(t.relation)
            profiles[t.tail]["neighbors"].add(t.head)
        return dict(profiles)

    # ------------------------------------------------------------------
    # Step 3: LLM-based type induction
    # ------------------------------------------------------------------

    def _llm_induce(
        self, entities: set[str], triples: list[Triple]
    ) -> InducedSchema:
        """Use the LLM to induce types from entities and relations."""
        # Format entities
        entity_list = sorted(entities)
        entities_str = ", ".join(entity_list[:200])  # Limit for prompt size

        # Format relations
        relation_samples = []
        for t in triples[:100]:  # Sample for prompt size
            relation_samples.append(f"{t.head} → {t.relation} → {t.tail}")
        relations_str = "\n".join(relation_samples)

        prompt = _TYPE_INDUCTION_PROMPT.format(
            entities=entities_str,
            relations=relations_str,
        )

        try:
            raw = self.llm.generate_json(prompt, system=_INDUCTION_SYSTEM)
        except Exception as e:
            logger.warning("LLM induction failed: %s — falling back to heuristic", e)
            return self._heuristic_induction(triples)

        return self._parse_induction_result(raw)

    def _parse_induction_result(self, raw: dict | list) -> InducedSchema:
        """Parse the LLM induction result into an InducedSchema."""
        schema = InducedSchema()

        if isinstance(raw, list):
            raw = {"types": raw, "relation_types": []}

        # Parse entity types
        for t in raw.get("types", []):
            if not isinstance(t, dict):
                continue
            name = t.get("name", "Unknown")
            et = EntityType(
                name=name,
                description=t.get("description", ""),
                entities=t.get("entities", []),
                is_event_type=t.get("is_event_type", False),
                parent_type=t.get("parent_type"),
            )
            schema.entity_types[name] = et

            # Build hierarchy
            if et.parent_type:
                if et.parent_type not in schema.type_hierarchy:
                    schema.type_hierarchy[et.parent_type] = []
                schema.type_hierarchy[et.parent_type].append(name)

        # Parse relation types
        for r in raw.get("relation_types", []):
            if not isinstance(r, dict):
                continue
            name = r.get("name", "UNKNOWN")
            rt = RelationType(
                name=name,
                description=r.get("description", ""),
                source_types=r.get("source_types", []),
                target_types=r.get("target_types", []),
            )
            schema.relation_types[name] = rt

        return schema

    # ------------------------------------------------------------------
    # Step 4: Merge similar types (PG-HIVE Algorithm 2)
    # ------------------------------------------------------------------

    def _merge_similar_types(self, schema: InducedSchema) -> InducedSchema:
        """
        Merge types with Jaccard similarity above threshold.

        From PG-HIVE (Algorithm 2):
        - For each pair of types, compute Jaccard similarity of entity sets
        - If similarity > θ, merge the smaller into the larger
        """
        type_names = list(schema.entity_types.keys())
        merged = set()

        for i in range(len(type_names)):
            if type_names[i] in merged:
                continue
            for j in range(i + 1, len(type_names)):
                if type_names[j] in merged:
                    continue
                t1 = schema.entity_types[type_names[i]]
                t2 = schema.entity_types[type_names[j]]

                # Compute Jaccard similarity of entity sets
                set1 = set(t1.entities)
                set2 = set(t2.entities)
                if not set1 and not set2:
                    continue
                intersection = set1 & set2
                union = set1 | set2
                jaccard = len(intersection) / len(union) if union else 0

                if jaccard >= self.merge_threshold:
                    # Merge t2 into t1
                    t1.entities = list(set1 | set2)
                    merged.add(type_names[j])
                    logger.info(
                        "Merged type '%s' into '%s' (Jaccard=%.2f)",
                        type_names[j], type_names[i], jaccard,
                    )

        # Remove merged types
        for name in merged:
            del schema.entity_types[name]

        return schema

    # ------------------------------------------------------------------
    # Step 5: Build relation constraints
    # ------------------------------------------------------------------

    def _build_relation_constraints(
        self, schema: InducedSchema, triples: list[Triple]
    ) -> InducedSchema:
        """
        For each relation, determine valid source/target types
        based on which types the head/tail entities belong to.
        """
        # Build entity → type mapping
        entity_to_type: dict[str, str] = {}
        for type_name, et in schema.entity_types.items():
            for entity in et.entities:
                entity_to_type[entity] = type_name

        # Count relation type constraints
        rel_counts: dict[str, dict] = defaultdict(
            lambda: {"sources": set(), "targets": set(), "count": 0}
        )
        for t in triples:
            src_type = entity_to_type.get(t.head, "Unknown")
            tgt_type = entity_to_type.get(t.tail, "Unknown")
            rel_counts[t.relation]["sources"].add(src_type)
            rel_counts[t.relation]["targets"].add(tgt_type)
            rel_counts[t.relation]["count"] += 1

        # Update or create relation types
        for rel_name, info in rel_counts.items():
            if rel_name in schema.relation_types:
                rt = schema.relation_types[rel_name]
                rt.source_types = sorted(info["sources"])
                rt.target_types = sorted(info["targets"])
                rt.count = info["count"]
            else:
                schema.relation_types[rel_name] = RelationType(
                    name=rel_name,
                    source_types=sorted(info["sources"]),
                    target_types=sorted(info["targets"]),
                    count=info["count"],
                )

        return schema

    # ------------------------------------------------------------------
    # Fallback heuristic induction (no LLM)
    # ------------------------------------------------------------------

    def _heuristic_induction(self, triples: list[Triple]) -> InducedSchema:
        """
        Fallback: induce types purely from triple structure.
        Groups entities by their relational profile.
        """
        schema = InducedSchema()

        # Group by relation patterns
        head_rels: dict[str, set[str]] = defaultdict(set)
        tail_rels: dict[str, set[str]] = defaultdict(set)
        for t in triples:
            head_rels[t.head].add(t.relation)
            tail_rels[t.tail].add(t.relation)

        # Group entities by profile
        profile_groups: dict[tuple, list[str]] = defaultdict(list)
        all_entities = set()
        for t in triples:
            all_entities.add(t.head)
            all_entities.add(t.tail)

        for entity in all_entities:
            profile = tuple(sorted(head_rels.get(entity, set()))) + ("|",) + tuple(sorted(tail_rels.get(entity, set())))
            profile_groups[profile].append(entity)

        # Create types from groups
        type_idx = 0
        for profile, entities in profile_groups.items():
            type_name = f"Type_{type_idx}"
            schema.entity_types[type_name] = EntityType(
                name=type_name,
                entities=entities,
            )
            type_idx += 1

        return self._build_relation_constraints(schema, triples)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def print_schema(self, schema: InducedSchema) -> None:
        """Pretty-print the induced schema."""
        print("\n" + "=" * 60)
        print("  INDUCED SCHEMA")
        print("=" * 60)

        print(f"\n  Entity Types ({len(schema.entity_types)}):")
        for name, et in schema.entity_types.items():
            prefix = "  [EVENT] " if et.is_event_type else "  "
            parent = f" (subtype of {et.parent_type})" if et.parent_type else ""
            print(f"  {prefix}{name}{parent}")
            print(f"       Entities: {', '.join(et.entities[:10])}")
            if et.description:
                print(f"       Desc: {et.description}")

        print(f"\n  Relation Types ({len(schema.relation_types)}):")
        for name, rt in schema.relation_types.items():
            print(f"    {rt.source_types} --[{name}]--> {rt.target_types}  (n={rt.count})")

        if schema.type_hierarchy:
            print(f"\n  Type Hierarchy:")
            for parent, children in schema.type_hierarchy.items():
                print(f"    {parent} → {children}")
