"""
agent.py
--------
V3 Component: Agentic Controller

Orchestrates a Plan → Retrieve → Evaluate → Refine loop on top of
the V2.5 pipeline. Unlike V2.5 (which retries the same pipeline with
different search terms), V3 decomposes queries into sub-questions,
retrieves each independently, evaluates coverage, and refines.

Architecture
------------
                    ┌─────────┐
                    │  Query  │
                    └────┬────┘
                         │
                    ┌────▼────┐
                    │ Memory  │──── cache hit ──→ Return
                    │ Recall  │
                    └────┬────┘
                         │ miss
                    ┌────▼────┐
    ┌──────────────►│ Planner │
    │               └────┬────┘
    │                    │ sub-questions
    │               ┌────▼────┐
    │               │Retrieve │ (uses V2.5 pipeline)
    │               └────┬────┘
    │               ┌────▼────┐
    │      gaps ◄───│Evaluate │
    │               └────┬────┘
    │                    │ sufficient
    └── refine      ┌────▼────┐
                    │Synthesize│
                    └────┬────┘
                         │
                    ┌────▼────┐
                    │  Store  │
                    └─────────┘
"""

from __future__ import annotations

import json
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

from .pipeline import GraphRAGPipeline
from .memory import AgentMemory
from .llm_interface import LLMInterface

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Result from the agentic query execution."""
    query: str
    answer: str
    sub_results: list[dict] = field(default_factory=list)
    plan_history: list[dict] = field(default_factory=list)
    memory_hit: bool = False
    iterations: int = 0
    timing: dict = field(default_factory=dict)


@dataclass
class QueryPlan:
    """A decomposed query plan."""
    original_query: str
    sub_questions: list[str]
    reasoning: str = ""


class GraphRAGAgent:
    """
    V3 Agentic Controller.

    Uses the V2.5 pipeline as its retrieval engine and adds:
    - Query decomposition (plan)
    - Multi-step retrieval
    - Quality evaluation
    - Memory recall and storage

    Parameters
    ----------
    pipeline : GraphRAGPipeline
        The underlying V2.5 pipeline (used for retrieval).
    memory : AgentMemory
        Cross-session memory layer.
    llm : LLMInterface
        For planning, evaluation, synthesis.
    max_iterations : int
        Max plan-retrieve-evaluate cycles.
    """

    def __init__(
        self,
        pipeline: GraphRAGPipeline,
        memory: AgentMemory,
        llm: LLMInterface,
        max_iterations: int = 3,
    ):
        self.pipeline = pipeline
        self.memory = memory
        self.llm = llm
        self.max_iterations = max_iterations

    def execute(self, query: str, domain: str = "default") -> AgentResult:
        """
        Execute an agentic query: Plan → Retrieve → Evaluate → Refine.

        Parameters
        ----------
        query : str
            The user's natural language query.
        domain : str
            Domain for memory scoping.

        Returns
        -------
        AgentResult
            The final answer with full trace.
        """
        timing = {}
        plan_history = []

        # ── Step 1: Memory Recall ─────────────────────────────────────
        t0 = time.perf_counter()
        memories = self.memory.recall(query, domain=domain, threshold=0.90)
        timing["memory_recall_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        if memories and memories[0].similarity >= 0.95:
            best = memories[0]
            logger.info(
                "Memory hit (sim=%.3f, age=%.0fs): '%s'",
                best.similarity, best.age_seconds, best.query[:50],
            )
            return AgentResult(
                query=query,
                answer=best.answer,
                memory_hit=True,
                timing=timing,
            )

        # ── Step 2: Plan ──────────────────────────────────────────────
        t0 = time.perf_counter()
        plan = self._plan(query)
        timing["plan_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        plan_history.append({
            "iteration": 0,
            "sub_questions": plan.sub_questions,
            "reasoning": plan.reasoning,
        })

        # ── Step 3: Retrieve-Evaluate Loop ────────────────────────────
        all_results = []
        iteration = 0

        for iteration in range(self.max_iterations):
            t0 = time.perf_counter()

            # Retrieve for each sub-question
            step_results = []
            for sub_q in plan.sub_questions:
                result = self.pipeline.query(sub_q, max_retries=1)
                step_results.append(result)

            all_results.extend(step_results)
            timing[f"retrieve_{iteration}_ms"] = round(
                (time.perf_counter() - t0) * 1000, 1
            )

            # Evaluate
            t0 = time.perf_counter()
            eval_result = self._evaluate(query, all_results)
            timing[f"evaluate_{iteration}_ms"] = round(
                (time.perf_counter() - t0) * 1000, 1
            )

            if eval_result["sufficient"]:
                logger.info(
                    "Retrieval sufficient after %d iteration(s)", iteration + 1
                )
                break

            # Refine if not sufficient and not last iteration
            if iteration < self.max_iterations - 1:
                t0 = time.perf_counter()
                plan = self._refine(query, plan, eval_result.get("gaps", []))
                timing[f"refine_{iteration}_ms"] = round(
                    (time.perf_counter() - t0) * 1000, 1
                )
                plan_history.append({
                    "iteration": iteration + 1,
                    "sub_questions": plan.sub_questions,
                    "gaps": eval_result.get("gaps", []),
                })

        # ── Step 4: Synthesize ────────────────────────────────────────
        t0 = time.perf_counter()
        answer = self._synthesize(query, all_results)
        timing["synthesize_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        # ── Step 5: Store in Memory ───────────────────────────────────
        merged_context = "\n---\n".join(
            r.get("context", "") for r in all_results if r.get("context")
        )
        total_facts = sum(
            len(r.get("subgraph", {}).get("relationships", []))
            for r in all_results
        )
        try:
            self.memory.store(
                query=query,
                answer=answer,
                context=merged_context[:2000],  # Cap context storage
                intent="agent",
                strategy="multi_step",
                quality_score=total_facts,
                domain=domain,
            )
        except Exception as e:
            logger.warning("Failed to store memory: %s", e)

        return AgentResult(
            query=query,
            answer=answer,
            sub_results=all_results,
            plan_history=plan_history,
            iterations=iteration + 1,
            timing=timing,
        )

    def _plan(self, query: str) -> QueryPlan:
        """
        Decompose a query into sub-questions.

        For simple queries, returns the original query as-is.
        For complex/multi-hop queries, decomposes into steps.
        """
        prompt = f"""Decompose this question into 1-3 simple sub-questions that can each be answered by searching a knowledge graph.
If the question is already simple, return just the original question.

Question: "{query}"

Return ONLY a JSON array of strings. Example: ["What drugs treat diabetes?", "What side effects do those drugs have?"]
Keep sub-questions short and specific."""

        try:
            response = self.llm.generate_text(prompt)
            # Try to parse JSON from response
            clean = response.strip()
            if clean.startswith("["):
                sub_qs = json.loads(clean)
                if isinstance(sub_qs, list) and all(isinstance(q, str) for q in sub_qs):
                    return QueryPlan(
                        original_query=query,
                        sub_questions=sub_qs[:3],  # Cap at 3
                        reasoning="LLM decomposition",
                    )
        except Exception as e:
            logger.debug("Plan decomposition failed: %s", e)

        # Fallback: use original query
        return QueryPlan(
            original_query=query,
            sub_questions=[query],
            reasoning="Simple query — no decomposition needed",
        )

    def _evaluate(self, query: str, results: list[dict]) -> dict:
        """
        Evaluate whether the retrieved results are sufficient to answer the query.

        Returns:
            {"sufficient": bool, "gaps": list[str], "coverage": float}
        """
        # Count total facts and unique intents
        total_facts = sum(
            len(r.get("subgraph", {}).get("relationships", []))
            for r in results
        )
        out_of_scope = sum(
            1 for r in results if r.get("intent") == "out_of_scope"
        )
        has_context = any(
            r.get("context", "").strip() for r in results
        )

        # If ALL sub-queries are out_of_scope, we're done (nothing to find)
        if out_of_scope == len(results):
            return {"sufficient": True, "gaps": [], "coverage": 0.0}

        # Sufficient if we have ≥5 facts and at least some context
        sufficient = total_facts >= 5 and has_context
        coverage = min(1.0, total_facts / 10)  # Normalize to 0-1

        gaps = []
        if not has_context:
            gaps.append("No context retrieved from any sub-question")
        if total_facts < 5:
            gaps.append(f"Only {total_facts} facts found (need ≥5)")

        return {"sufficient": sufficient, "gaps": gaps, "coverage": coverage}

    def _refine(self, query: str, plan: QueryPlan, gaps: list[str]) -> QueryPlan:
        """Refine the plan based on evaluation gaps."""
        gap_text = "\n".join(f"- {g}" for g in gaps)
        previous = "\n".join(f"- {q}" for q in plan.sub_questions)

        prompt = f"""Original question: "{query}"
Previous sub-questions that were searched:
{previous}

Gaps found:
{gap_text}

Generate 1-2 NEW, DIFFERENT sub-questions to fill these gaps.
Return ONLY a JSON array of strings."""

        try:
            response = self.llm.generate_text(prompt)
            clean = response.strip()
            if clean.startswith("["):
                new_qs = json.loads(clean)
                if isinstance(new_qs, list):
                    return QueryPlan(
                        original_query=query,
                        sub_questions=[q for q in new_qs if isinstance(q, str)][:2],
                        reasoning=f"Refined to fill gaps: {gaps}",
                    )
        except Exception as e:
            logger.debug("Refinement failed: %s", e)

        # Fallback: re-try original
        return QueryPlan(
            original_query=query,
            sub_questions=[query],
            reasoning="Refinement failed — retrying original",
        )

    def _synthesize(self, query: str, results: list[dict]) -> str:
        """
        Merge sub-results into a single grounded answer.

        Strictly uses context from results — no external knowledge.
        """
        # Collect all contexts
        contexts = []
        for r in results:
            ctx = r.get("context", "")
            if ctx and ctx.strip():
                contexts.append(ctx)

        if not contexts:
            return "No relevant information was found in the knowledge graph to answer this question."

        merged_context = "\n---\n".join(contexts)

        # Use LLM to synthesize
        try:
            answer = self.llm.answer(query, merged_context)
            return answer
        except Exception as e:
            logger.warning("Synthesis LLM call failed: %s", e)
            # Fallback: return best individual answer
            for r in results:
                if r.get("answer"):
                    return r["answer"]
            return merged_context
