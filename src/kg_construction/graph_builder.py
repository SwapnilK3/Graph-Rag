"""
src/kg_construction/graph_builder.py
------------------------------------
Populates a Neo4j property graph from resolved triples and induced schema.

**Paper basis**: StructuGraphRAG (Zhu et al., 2024)
  - Entity-Relationship diagram from extracted entities
  - PostgreSQL-style schema → we adapt to Neo4j property graph
  - Populate with both extracted entities and survey response data

**Paper basis**: AutoSchemaKG (Bai et al., 2025)
  - Graph stored in NetworkX/Neo4j for retrieval
  - Subgraphs fed into LLM for answer generation

Algorithm
---------
1. Create unique constraints for each entity type
2. Create nodes for each entity (with label = entity type)
3. Create relationships from triples
4. Attach provenance metadata (source chunk, confidence)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.kg_construction.triple_extractor import Triple
from src.kg_construction.schema_inducer import InducedSchema
from src.kg_construction.entity_resolver import ResolutionResult
from src.utils.connector import Neo4jConnector

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────

@dataclass
class BuildResult:
    """Result of graph construction."""
    nodes_created: int = 0
    relationships_created: int = 0
    labels_used: list[str] = field(default_factory=list)
    relationship_types_used: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────
# Main builder
# ──────────────────────────────────────────────────────────────

class GraphBuilder:
    """
    Builds a Neo4j knowledge graph from resolved triples and induced schema.

    Parameters
    ----------
    connector : Neo4jConnector
        Active Neo4j connection.
    clear_existing : bool
        If True, wipe the database before building (default False).
    batch_size : int
        Number of operations per transaction (default 100).
    """

    def __init__(
        self,
        connector: Neo4jConnector,
        clear_existing: bool = False,
        batch_size: int = 100,
    ):
        self.connector = connector
        self.clear_existing = clear_existing
        self.batch_size = batch_size

    def build(
        self,
        triples: list[Triple],
        schema: InducedSchema | None = None,
    ) -> BuildResult:
        """
        Build the knowledge graph from triples.

        Parameters
        ----------
        triples : list[Triple]
            Resolved (deduplicated) triples.
        schema : InducedSchema, optional
            Induced schema. If provided, nodes get typed labels.
            If None, a generic "Entity" label is used.

        Returns
        -------
        BuildResult
        """
        result = BuildResult()

        if self.clear_existing:
            logger.info("Clearing existing database...")
            self.connector.clear_database()

        # Build entity → type mapping from schema
        entity_to_type = self._build_entity_type_map(schema)

        # Step 1: Create constraints (optional, best-effort)
        labels = set(entity_to_type.values()) or {"Entity"}
        self._create_constraints(labels)
        result.labels_used = sorted(labels)

        # Step 2: Create nodes
        nodes_created = self._create_nodes(triples, entity_to_type)
        result.nodes_created = nodes_created

        # Step 3: Create relationships
        rels_created, rel_types = self._create_relationships(triples, entity_to_type)
        result.relationships_created = rels_created
        result.relationship_types_used = sorted(rel_types)

        logger.info(
            "Graph built: %d nodes, %d relationships, labels=%s",
            result.nodes_created, result.relationships_created,
            result.labels_used,
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_entity_type_map(
        self, schema: InducedSchema | None
    ) -> dict[str, str]:
        """Map each entity to its induced type label."""
        if not schema:
            return {}
        mapping: dict[str, str] = {}
        for type_name, et in schema.entity_types.items():
            for entity in et.entities:
                mapping[entity] = type_name
        return mapping

    def _create_constraints(self, labels: set[str]) -> None:
        """Create uniqueness constraints for entity names (best-effort)."""
        for label in labels:
            try:
                cypher = (
                    f"CREATE CONSTRAINT IF NOT EXISTS "
                    f"FOR (n:`{label}`) REQUIRE n.name IS UNIQUE"
                )
                self.connector.run(cypher)
            except Exception as e:
                logger.debug("Constraint creation skipped for %s: %s", label, e)

    def _create_nodes(
        self,
        triples: list[Triple],
        entity_to_type: dict[str, str],
    ) -> int:
        """Create nodes for all entities in the triples."""
        # Collect unique entities
        entities: dict[str, dict] = {}
        for t in triples:
            for name, etype in [(t.head, t.head_type), (t.tail, t.tail_type)]:
                if name not in entities:
                    entities[name] = {
                        "name": name,
                        "label": entity_to_type.get(name, "Entity"),
                        "is_event": etype == "event",
                    }

        # Create nodes in batches
        created = 0
        for name, info in entities.items():
            label = info["label"]
            props = {"name": name}
            if info["is_event"]:
                props["node_kind"] = "event"

            cypher = f"MERGE (n:`{label}` {{name: $name}}) SET n += $props"
            try:
                self.connector.run(cypher, {"name": name, "props": props})
                created += 1
            except Exception as e:
                logger.warning("Failed to create node %s: %s", name, e)

        return created

    def _create_relationships(
        self,
        triples: list[Triple],
        entity_to_type: dict[str, str],
    ) -> tuple[int, set[str]]:
        """Create relationships from triples."""
        created = 0
        rel_types: set[str] = set()

        for t in triples:
            head_label = entity_to_type.get(t.head, "Entity")
            tail_label = entity_to_type.get(t.tail, "Entity")
            rel_type = t.relation

            cypher = (
                f"MATCH (a:`{head_label}` {{name: $head}}) "
                f"MATCH (b:`{tail_label}` {{name: $tail}}) "
                f"MERGE (a)-[r:`{rel_type}`]->(b) "
                f"SET r.confidence = $confidence"
            )
            try:
                self.connector.run(cypher, {
                    "head": t.head,
                    "tail": t.tail,
                    "confidence": t.confidence,
                })
                created += 1
                rel_types.add(rel_type)
            except Exception as e:
                logger.warning(
                    "Failed to create rel %s-[%s]->%s: %s",
                    t.head, rel_type, t.tail, e,
                )

        return created, rel_types
