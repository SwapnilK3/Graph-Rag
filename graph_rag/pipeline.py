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

from .connector import GraphDBConnector
from .schema_discovery import SchemaDiscovery
from .auto_config import AutoConfigGenerator
from .entity_extractor import QueryTimeEntityExtractor
from .intent_classifier import IntentClassifier
from .traversal_engine import SmartTraversalEngine
from .context_generator import ContextGenerator
from .llm_interface import LLMInterface

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
        domain: Optional[str] = None,
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

        # ── Step 3: Handle Config ─────────────────────────────────────
        if config:
            self.config = config
        elif domain:
            from .config import load_domain_config
            self.config = load_domain_config(domain)
            if verbose:
                print(f"  Loaded config for domain: {domain}")
        else:
            generator = AutoConfigGenerator(self.schema)
            self.config = generator.generate()
            if verbose:
                generator.print_config(self.config)


        # ── Step 4: LLM (now required for V2 intelligence) ─────────────
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

        # ── Step 5: Initialize other components ───────────────────────

        # Collect ALL searchable properties across all labels
        all_search_props = set()
        for props in self.schema["searchable_properties"].values():
            all_search_props.update(props)
        search_props = sorted(all_search_props) or ["name"]

        self.extractor = QueryTimeEntityExtractor(
            self.connector,
            llm=self.llm,  # Passed for V2 semantic search
            search_properties=search_props,
        )
        self.classifier = IntentClassifier(self.config, llm=self.llm)
        self.engine     = SmartTraversalEngine(self.connector, self.config)
        self.generator  = ContextGenerator(self.config, llm=self.llm)

        if verbose:
            status = "enabled" if self.llm else "disabled"
            print(f"\n  Pipeline ready — LLM: {status}")
            print(f"  Searchable properties: {search_props}")
            print(f"  Intents: {list(self.config['intent_patterns'].keys())}\n")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(self, user_query: str, max_retries: int = 2) -> dict:
        """
        V2 Agentic Query: Extends the standard pipeline with a reflection loop.
        If the initial search is insufficient, it attempts to re-plan.
        """
        trace = []
        timing = {}
        
        # Initial search state
        current_query = user_query
        best_subgraph = {"nodes": [], "relationships": []}
        best_context = ""
        best_intent = "none"
        best_quality = 0  # Fix 7: Use fact count as quality metric
        
        for attempt in range(max_retries + 1):
            step_start = time.perf_counter()
            logger.info("Starting reasoning step %d", attempt + 1)
            
            # Step 1: Entity extraction
            t0 = time.perf_counter()
            entry_nodes = self.extractor.extract_entry_nodes(current_query)
            timing[f"step{attempt}_extract_ms"] = round((time.perf_counter() - t0) * 1000, 1)

            if entry_nodes:
                # Step 2: Intent classification (Fix 8: re-classify on each attempt)
                t0 = time.perf_counter()
                intent = self.classifier.classify(current_query)
                timing[f"step{attempt}_classify_ms"] = round((time.perf_counter() - t0) * 1000, 1)

                # Step 3: Graph traversal
                t0 = time.perf_counter()
                subgraph = self.engine.traverse(entry_nodes, intent)
                timing[f"step{attempt}_traverse_ms"] = round((time.perf_counter() - t0) * 1000, 1)

                # Step 4: Context generation
                t0 = time.perf_counter()
                context = self.generator.generate(subgraph)
                timing[f"step{attempt}_context_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                
                # Fix 7: Quality metric — entity coverage ratio instead of string length
                rel_count = len(subgraph.get("relationships", []))
                node_count = len(subgraph.get("nodes", []))
                quality_score = rel_count + node_count  # facts found
                
                if quality_score > best_quality:
                    best_subgraph = subgraph
                    best_context = context
                    best_intent = intent
                    best_quality = quality_score
                
                trace.append({
                    "step": attempt + 1,
                    "nodes": [n.get("name") for n in entry_nodes],
                    "intent": intent,
                    "found_facts": rel_count,
                    "quality_score": quality_score,
                })

            # Fix 7: Reflection based on fact count, not string length
            if best_quality >= 5 or attempt == max_retries:
                # Sufficient data found (>=5 facts) or max retries reached
                break
            
            # Re-planning: Ask LLM for a better search term
            logger.info("Context insufficient (quality=%d). Re-planning...", best_quality)
            new_guidance = self._replan(user_query, best_context, trace)
            if new_guidance:
                current_query = new_guidance
            else:
                break

        # Final Step: LLM answer
        answer = None
        if self.llm and best_context:
            t0 = time.perf_counter()
            try:
                answer = self.llm.answer(user_query, best_context)
            except Exception as e:
                logger.warning("LLM call failed: %s", e)
                answer = f"(LLM error: {e})"
            timing["llm_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        return {
            "query":       user_query,
            "entry_nodes": [n.get("name", "Unknown") for n in best_subgraph.get("nodes", [])],
            "intent":      best_intent,
            "strategy":    best_subgraph.get("strategy", "unknown"),
            "hop_depth":   best_subgraph.get("hop_depth", 0),
            "subgraph":    best_subgraph,
            "context":     best_context or "No matching information found.",
            "answer":      answer,
            "thought_process": trace,
            "timing":      timing,
        }

    def _replan(self, query: str, context: str, trace: list) -> Optional[str]:
        """Ask the LLM to provide a better search focus based on what we've missed."""
        if not self.llm: return None
        
        attempt_logs = "\n".join([f"Step {t['step']}: Searched for {t['nodes']}, found {t['found_facts']} facts." for t in trace])
        
        prompt = f"""You are a Graph-RAG planner. 
We are searching for the answer to: "{query}"
So far we have:
{attempt_logs}

The current graph context is too thin to answer fully.
Suggest a NEW, SPECIFIC search term (entity name or relationship concept) that I should look for in the graph to fill the gap.
Return ONLY the search term. If you can't improve it, return empty.
Focus: """

        try:
            response = self.llm.generate_text(prompt)
            return response if response and len(response) < 50 else None
        except Exception:
            return None

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
