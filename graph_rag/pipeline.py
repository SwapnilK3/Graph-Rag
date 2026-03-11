"""
pipeline.py
-----------
Unified Graph-RAG Pipeline — fully automatic, zero configuration.

This is the main orchestrator. Point it at any Neo4j graph:
  1. Auto-discovers the schema
  2. Auto-generates the config (intents, templates, searchable props)
  3. Initializes all 5 components
  4. Processes any natural-language query end-to-end

Usage
-----
    from pipeline import GraphRAGPipeline

    pipeline = GraphRAGPipeline()         # connects, discovers, configures
    result   = pipeline.query("What are the side effects of aspirin?")
    print(result["answer"])

Or with explicit connector:
    from connector import GraphDBConnector
    pipeline = GraphRAGPipeline(connector=GraphDBConnector())
"""

from __future__ import annotations

import os
import time
import logging
from typing import Optional

from connector import GraphDBConnector
from schema_discovery import SchemaDiscovery
from auto_config import AutoConfigGenerator
from entity_extractor import QueryTimeEntityExtractor
from intent_classifier import IntentClassifier
from traversal_engine import SmartTraversalEngine
from context_generator import ContextGenerator
from llm_interface import LLMInterface

logger = logging.getLogger(__name__)


class GraphRAGPipeline:
    """
    Fully automatic Graph-RAG pipeline.

    Parameters
    ----------
    connector:
        Optional pre-built GraphDBConnector. If None, creates one from .env.
    config:
        Optional pre-built config dict. If None, auto-discovers and generates.
    enable_llm:
        If True (default), initializes Gemini LLM for answering.
        Set to False for testing without an API key.
    verbose:
        If True, prints schema and config reports during initialization.
    """

    def __init__(
        self,
        connector: Optional[GraphDBConnector] = None,
        config: Optional[dict] = None,
        enable_llm: bool = True,
        verbose: bool = True,
    ):
        # ── Step 1: Connect ───────────────────────────────────────────
        self.connector = connector or GraphDBConnector()
        self.connector.check_connection()

        # ── Step 2: Discover schema ───────────────────────────────────
        self._discovery = SchemaDiscovery(self.connector)
        self.schema = self._discovery.discover()
        if verbose:
            self._discovery.print_schema(self.schema)

        # ── Step 3: Generate config ───────────────────────────────────
        if config:
            self.config = config
        else:
            generator = AutoConfigGenerator(self.schema)
            self.config = generator.generate()
            if verbose:
                generator.print_config(self.config)

        # ── Step 4: Initialize components ─────────────────────────────

        # Collect ALL searchable properties across all labels
        all_search_props = set()
        for props in self.schema["searchable_properties"].values():
            all_search_props.update(props)
        search_props = sorted(all_search_props) or ["name"]

        self.extractor = QueryTimeEntityExtractor(
            self.connector,
            search_properties=search_props,
        )
        self.classifier = IntentClassifier(self.config)
        self.engine     = SmartTraversalEngine(self.connector, self.config)
        self.generator  = ContextGenerator(self.config)

        # ── Step 5: LLM (optional) ────────────────────────────────────
        self.llm = None
        if enable_llm:
            key = os.getenv("GEMINI_API_KEY", "")
            if key and key != "your_gemini_api_key_here":
                try:
                    self.llm = LLMInterface(api_key=key)
                    logger.info("Gemini LLM initialized")
                except Exception as e:
                    logger.warning("LLM initialization failed: %s", e)
            else:
                logger.info("GEMINI_API_KEY not set — LLM disabled")

        if verbose:
            status = "enabled" if self.llm else "disabled"
            print(f"\n  Pipeline ready — LLM: {status}")
            print(f"  Searchable properties: {search_props}")
            print(f"  Intents: {list(self.config['intent_patterns'].keys())}\n")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(self, user_query: str) -> dict:
        """
        Process a natural-language query end-to-end.

        Returns
        -------
        dict with keys:
            query          — original query
            entry_nodes    — list of matched node dicts
            intent         — classified intent name
            strategy       — traversal strategy used
            hop_depth      — max hops traversed
            subgraph       — raw subgraph dict (nodes + relationships)
            context        — formatted text context
            answer         — LLM answer (or None if LLM disabled)
            timing         — dict of per-step timings in ms
        """
        timing = {}

        # Step 1: Entity extraction
        t0 = time.perf_counter()
        entry_nodes = self.extractor.extract_entry_nodes(user_query)
        timing["extract_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        if not entry_nodes:
            return {
                "query":       user_query,
                "entry_nodes": [],
                "intent":      "none",
                "strategy":    "none",
                "hop_depth":   0,
                "subgraph":    {"nodes": [], "relationships": []},
                "context":     "No matching entities found in the knowledge graph.",
                "answer":      None,
                "timing":      timing,
            }

        # Step 2: Intent classification
        t0 = time.perf_counter()
        intent = self.classifier.classify(user_query)
        timing["classify_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        # Step 3: Graph traversal
        t0 = time.perf_counter()
        subgraph = self.engine.traverse(entry_nodes, intent)
        timing["traverse_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        # Step 4: Context generation
        t0 = time.perf_counter()
        context = self.generator.generate(subgraph)
        timing["context_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        # Step 5: LLM answer (optional)
        answer = None
        if self.llm:
            t0 = time.perf_counter()
            try:
                answer = self.llm.answer(user_query, context)
            except Exception as e:
                logger.warning("LLM call failed: %s", e)
                answer = f"(LLM error: {e})"
            timing["llm_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        return {
            "query":       user_query,
            "entry_nodes": [n["name"] for n in entry_nodes],
            "intent":      intent,
            "strategy":    subgraph.get("strategy", "unknown"),
            "hop_depth":   subgraph.get("hop_depth", 0),
            "subgraph":    subgraph,
            "context":     context,
            "answer":      answer,
            "timing":      timing,
        }

    def query_interactive(self, user_query: str) -> str:
        """
        High-level query that prints a nicely formatted result.
        Returns the answer string.
        """
        result = self.query(user_query)

        print(f"\n{'─' * 60}")
        print(f"  Query   : {result['query']}")
        print(f"  Nodes   : {result['entry_nodes']}")
        print(f"  Intent  : {result['intent']}")
        print(f"  Strategy: {result['strategy']} (depth={result['hop_depth']})")

        nodes_count = len(result["subgraph"].get("nodes", []))
        rels_count  = len(result["subgraph"].get("relationships", []))
        print(f"  Graph   : {nodes_count} nodes, {rels_count} edges")

        timing = result["timing"]
        timing_str = ", ".join(f"{k}={v}" for k, v in timing.items())
        print(f"  Timing  : {timing_str}")

        if result["answer"]:
            print(f"\n  Answer  : {result['answer']}")
        else:
            print(f"\n  Context :\n{result['context']}")

        print(f"{'─' * 60}\n")
        return result.get("answer", result["context"])

    def close(self):
        """Close the Neo4j connection."""
        self.connector.close()

    # Context manager support
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
