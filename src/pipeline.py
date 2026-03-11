"""
src/pipeline.py
----------------
End-to-end methodology pipeline orchestrator.

Wires together all components from the research papers into a single
pipeline that can:
  1. Ingest raw text
  2. Extract triples (AutoSchemaKG)
  3. Induce schema (AutoSchemaKG + PG-HIVE)
  4. Resolve entities
  5. Build knowledge graph in Neo4j (StructuGraphRAG)
  6. Discover schema from graph (PG-HIVE + Schema Inference)
  7. Evolve schema incrementally (AdaKGC)

The Graph-RAG query layer (graph_rag/) is the LAST step and is NOT
part of this pipeline — it consumes the output of this pipeline.

Usage
-----
    from src.pipeline import MethodologyPipeline
    pipeline = MethodologyPipeline()
    result = pipeline.run("Your document text here...")
    pipeline.print_report(result)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from src.utils.connector import Neo4jConnector
from src.utils.llm_client import LLMClient
from src.utils.text_processor import chunk_text

from src.kg_construction.triple_extractor import TripleExtractor, ExtractionResult
from src.kg_construction.schema_inducer import SchemaInducer, InducedSchema
from src.kg_construction.entity_resolver import EntityResolver, ResolutionResult
from src.kg_construction.graph_builder import GraphBuilder, BuildResult

from src.schema_discovery.discoverer import SchemaDiscoverer, DiscoveredSchema
from src.schema_evolution.adapter import SchemaAdapter, EvolutionResult

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# Pipeline result
# ══════════════════════════════════════════════════════════════

@dataclass
class PipelineResult:
    """Result of running the full methodology pipeline."""
    # Stage 1: Extraction
    extraction: ExtractionResult | None = None
    # Stage 2: Schema Induction
    induced_schema: InducedSchema | None = None
    # Stage 3: Entity Resolution
    resolution: ResolutionResult | None = None
    # Stage 4: Graph Construction
    build: BuildResult | None = None
    # Stage 5: Schema Discovery
    discovered_schema: DiscoveredSchema | None = None
    # Stage 6: Schema Evolution (only on incremental runs)
    evolution: EvolutionResult | None = None
    # Metadata
    stages_completed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════
# Pipeline
# ══════════════════════════════════════════════════════════════

class MethodologyPipeline:
    """
    Full methodology pipeline from text to discovered schema.

    Parameters
    ----------
    connector : Neo4jConnector, optional
        Neo4j connection. If None, creates from .env variables.
    llm : LLMClient, optional
        LLM client. If None, creates from .env variables.
    chunk_size : int
        Text chunk size for extraction (default 1500).
    confidence_threshold : float
        Minimum triple confidence (default 0.5).
    similarity_threshold : float
        Entity resolution similarity threshold (default 0.85).
    merge_threshold : float
        Schema type merging Jaccard threshold (default 0.7).
    clear_graph : bool
        Whether to clear the graph before building (default False).
    """

    def __init__(
        self,
        connector: Neo4jConnector | None = None,
        llm: LLMClient | None = None,
        chunk_size: int = 1500,
        confidence_threshold: float = 0.5,
        similarity_threshold: float = 0.85,
        merge_threshold: float = 0.7,
        clear_graph: bool = False,
    ):
        self.connector = connector or Neo4jConnector()
        self.llm = llm or LLMClient()

        # Build component instances
        self.extractor = TripleExtractor(
            llm=self.llm,
            chunk_size=chunk_size,
            confidence_threshold=confidence_threshold,
        )
        self.inducer = SchemaInducer(
            llm=self.llm,
            merge_threshold=merge_threshold,
        )
        self.resolver = EntityResolver(
            similarity_threshold=similarity_threshold,
        )
        self.builder = GraphBuilder(
            connector=self.connector,
            clear_existing=clear_graph,
        )
        self.discoverer = SchemaDiscoverer(connector=self.connector)
        self.adapter = SchemaAdapter(merge_threshold=merge_threshold)

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run(self, text: str) -> PipelineResult:
        """
        Run the complete methodology pipeline on a document.

        Stages:
        1. Triple Extraction (AutoSchemaKG)
        2. Schema Induction (AutoSchemaKG + PG-HIVE)
        3. Entity Resolution
        4. Graph Construction (StructuGraphRAG)
        5. Schema Discovery (PG-HIVE)

        Parameters
        ----------
        text : str
            Full document text to process.

        Returns
        -------
        PipelineResult
        """
        result = PipelineResult()

        # ── Stage 1: Triple Extraction ──────────────────────
        logger.info("=== Stage 1: Triple Extraction (AutoSchemaKG) ===")
        try:
            result.extraction = self.extractor.extract(text)
            result.stages_completed.append("extraction")
            logger.info(
                "Extracted %d triples (%d entity, %d event)",
                len(result.extraction.triples),
                len(result.extraction.entity_triples),
                len(result.extraction.event_triples),
            )
        except Exception as e:
            result.errors.append(f"Extraction failed: {e}")
            logger.error("Extraction failed: %s", e)
            return result

        if not result.extraction.triples:
            result.errors.append("No triples extracted")
            return result

        # ── Stage 2: Schema Induction ───────────────────────
        logger.info("=== Stage 2: Schema Induction (AutoSchemaKG + PG-HIVE) ===")
        try:
            result.induced_schema = self.inducer.induce(result.extraction.triples)
            result.stages_completed.append("induction")
            logger.info(
                "Induced %d entity types, %d relation types",
                len(result.induced_schema.entity_types),
                len(result.induced_schema.relation_types),
            )
        except Exception as e:
            result.errors.append(f"Schema induction failed: {e}")
            logger.error("Schema induction failed: %s", e)

        # ── Stage 3: Entity Resolution ──────────────────────
        logger.info("=== Stage 3: Entity Resolution ===")
        try:
            result.resolution = self.resolver.resolve(result.extraction.triples)
            result.stages_completed.append("resolution")
            logger.info(
                "Resolved: %d→%d triples, %d→%d entities",
                result.resolution.original_count,
                result.resolution.resolved_count,
                result.resolution.entities_before,
                result.resolution.entities_after,
            )
        except Exception as e:
            result.errors.append(f"Entity resolution failed: {e}")
            logger.error("Entity resolution failed: %s", e)
            return result

        # ── Stage 4: Graph Construction ─────────────────────
        logger.info("=== Stage 4: Graph Construction (StructuGraphRAG) ===")
        try:
            triples_to_build = result.resolution.resolved_triples
            result.build = self.builder.build(
                triples_to_build,
                schema=result.induced_schema,
            )
            result.stages_completed.append("build")
            logger.info(
                "Built: %d nodes, %d relationships",
                result.build.nodes_created,
                result.build.relationships_created,
            )
        except Exception as e:
            result.errors.append(f"Graph construction failed: {e}")
            logger.error("Graph construction failed: %s", e)

        # ── Stage 5: Schema Discovery ───────────────────────
        logger.info("=== Stage 5: Schema Discovery (PG-HIVE) ===")
        try:
            result.discovered_schema = self.discoverer.discover()
            result.stages_completed.append("discovery")
            logger.info(
                "Discovered: %d node types, %d edge types",
                len(result.discovered_schema.node_types),
                len(result.discovered_schema.edge_types),
            )
        except Exception as e:
            result.errors.append(f"Schema discovery failed: {e}")
            logger.error("Schema discovery failed: %s", e)

        return result

    # ------------------------------------------------------------------
    # Incremental update
    # ------------------------------------------------------------------

    def run_incremental(
        self, text: str, existing_schema: DiscoveredSchema
    ) -> PipelineResult:
        """
        Incrementally process new text and evolve the schema.

        Stages 1-3 same as run(), then:
        4. Build graph (append, don't clear)
        5. Schema Evolution (AdaKGC) — compare against existing schema
        6. Re-discover updated schema
        """
        result = PipelineResult()

        # Stages 1-3
        try:
            result.extraction = self.extractor.extract(text)
            result.stages_completed.append("extraction")
        except Exception as e:
            result.errors.append(f"Extraction: {e}")
            return result

        if not result.extraction.triples:
            return result

        try:
            result.induced_schema = self.inducer.induce(result.extraction.triples)
            result.stages_completed.append("induction")
        except Exception as e:
            result.errors.append(f"Induction: {e}")

        try:
            result.resolution = self.resolver.resolve(result.extraction.triples)
            result.stages_completed.append("resolution")
        except Exception as e:
            result.errors.append(f"Resolution: {e}")
            return result

        # Stage 4: Append to graph
        try:
            result.build = self.builder.build(
                result.resolution.resolved_triples,
                schema=result.induced_schema,
            )
            result.stages_completed.append("build")
        except Exception as e:
            result.errors.append(f"Build: {e}")

        # Stage 5: Schema Evolution
        try:
            result.evolution = self.adapter.evolve(
                existing_schema,
                result.extraction.triples,
                induced=result.induced_schema,
            )
            result.stages_completed.append("evolution")
            logger.info(result.evolution.summary())
        except Exception as e:
            result.errors.append(f"Evolution: {e}")

        # Stage 6: Re-discover
        try:
            result.discovered_schema = self.discoverer.discover()
            result.stages_completed.append("discovery")
        except Exception as e:
            result.errors.append(f"Discovery: {e}")

        return result

    # ------------------------------------------------------------------
    # Individual stages (for flexibility)
    # ------------------------------------------------------------------

    def extract_only(self, text: str) -> ExtractionResult:
        """Run only the triple extraction stage."""
        return self.extractor.extract(text)

    def induce_only(self, triples) -> InducedSchema:
        """Run only the schema induction stage."""
        return self.inducer.induce(triples)

    def discover_only(self) -> DiscoveredSchema:
        """Run only the schema discovery stage (from existing graph)."""
        return self.discoverer.discover()

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_report(self, result: PipelineResult) -> None:
        """Print a human-readable summary of the pipeline run."""
        print("\n" + "=" * 60)
        print("  METHODOLOGY PIPELINE REPORT")
        print("=" * 60)

        print(f"\n  Stages completed: {', '.join(result.stages_completed)}")

        if result.errors:
            print(f"\n  Errors ({len(result.errors)}):")
            for e in result.errors:
                print(f"    - {e}")

        if result.extraction:
            ex = result.extraction
            print(f"\n  --- Extraction ---")
            print(f"    Chunks: {ex.chunks_processed}/{ex.total_chunks}")
            print(f"    Triples: {len(ex.triples)} (entity={len(ex.entity_triples)}, event={len(ex.event_triples)})")

        if result.induced_schema:
            s = result.induced_schema
            print(f"\n  --- Schema Induction ---")
            print(f"    Entity types: {len(s.entity_types)}")
            print(f"    Relation types: {len(s.relation_types)}")
            for name, et in s.entity_types.items():
                print(f"      [{name}]: {len(et.entities)} entities")

        if result.resolution:
            r = result.resolution
            print(f"\n  --- Entity Resolution ---")
            print(f"    Triples: {r.original_count} -> {r.resolved_count}")
            print(f"    Entities: {r.entities_before} -> {r.entities_after}")
            if r.merge_groups:
                print(f"    Merge groups: {len(r.merge_groups)}")

        if result.build:
            b = result.build
            print(f"\n  --- Graph Construction ---")
            print(f"    Nodes created: {b.nodes_created}")
            print(f"    Relationships: {b.relationships_created}")
            print(f"    Labels: {', '.join(b.labels_used)}")

        if result.discovered_schema:
            d = result.discovered_schema
            print(f"\n  --- Schema Discovery ---")
            print(f"    Node types: {len(d.node_types)}")
            print(f"    Edge types: {len(d.edge_types)}")
            for label, ns in d.node_types.items():
                sp = ", ".join(ns.searchable_properties) if ns.searchable_properties else "none"
                print(f"      [{label}] {ns.instance_count} instances, searchable: {sp}")

        if result.evolution:
            ev = result.evolution
            print(f"\n  --- Schema Evolution ---")
            print(f"    {ev.summary()}")

        print("\n" + "=" * 60 + "\n")

    def save_schema(self, schema: DiscoveredSchema, path: str) -> None:
        """Save discovered schema to JSON file."""
        with open(path, "w") as f:
            f.write(schema.to_json())
        logger.info("Schema saved to %s", path)
