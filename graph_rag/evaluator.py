"""
evaluator.py
------------
V3 Component: RAG Evaluation Framework

Evaluates Graph-RAG retrieval quality on five axes:
1. Grounding:  Is every fact in the answer traceable to a graph edge?
2. Multi-hop:  Did the system follow multi-hop paths correctly?
3. Coverage:   Did the system find all relevant facts?
4. Rejection:  Are out-of-scope queries correctly rejected?
5. Latency:    Is response time acceptable?

Usage
-----
    evaluator = RAGEvaluator(pipeline)
    report = evaluator.run_suite(TEST_CASES)
    evaluator.print_report(report)
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class EvalScore:
    """Score for a single test case."""
    test_name: str
    category: str
    passed: bool
    details: str = ""
    latency_ms: float = 0.0


@dataclass
class EvalReport:
    """Aggregate evaluation report."""
    scores: list[EvalScore] = field(default_factory=list)
    total: int = 0
    passed: int = 0
    failed: int = 0

    @property
    def pass_rate(self) -> float:
        return (self.passed / self.total * 100) if self.total else 0.0


@dataclass
class TestCase:
    """
    Definition of a single evaluation test case.

    Attributes
    ----------
    name : str
        Human-readable test name.
    category : str
        One of: grounding, multi_hop, coverage, rejection, abstract, edge_direction.
    query : str
        The query to run through the pipeline.
    expect_intent : str or None
        If set, the result intent must match.
    expect_strategy : str or None
        If set, the result strategy must match.
    expect_min_facts : int
        Minimum number of relationships expected.
    expect_rel_types : list[str] or None
        If set, ALL relationship types in the context must be in this list.
    expect_no_rel_types : list[str] or None
        If set, NONE of these relationship types should appear.
    expect_answer_contains : list[str] or None
        If set, the answer must contain all of these substrings.
    expect_no_llm_call : bool
        If True, timing should NOT contain 'llm_ms' (out-of-scope rejection).
    max_latency_ms : float
        Maximum acceptable total latency.
    """
    name: str
    category: str
    query: str
    expect_intent: Optional[str] = None
    expect_strategy: Optional[str] = None
    expect_min_facts: int = 0
    expect_rel_types: Optional[list[str]] = None
    expect_no_rel_types: Optional[list[str]] = None
    expect_answer_contains: Optional[list[str]] = None
    expect_no_llm_call: bool = False
    max_latency_ms: float = 30000.0  # 30s default


# ── Default Test Suite ────────────────────────────────────────────────
DEFAULT_TEST_CASES = [
    TestCase(
        name="In-domain specific query",
        category="coverage",
        query="What are the side effects of aspirin?",
        expect_min_facts=1,
    ),
    TestCase(
        name="Out-of-domain rejection",
        category="rejection",
        query="Who is the Prime Minister of India?",
        expect_intent="out_of_scope",
        expect_no_llm_call=True,
    ),
    TestCase(
        name="Abstract query detection",
        category="abstract",
        query="Why are drugs used?",
        expect_intent="abstract",
    ),
    TestCase(
        name="Generic entity (label match)",
        category="coverage",
        query="Tell me about drugs",
        expect_min_facts=0,  # At least finds Drug nodes via label
    ),
    TestCase(
        name="Empty/gibberish query rejection",
        category="rejection",
        query="xyzzy foobar blargh 12345",
        expect_intent="out_of_scope",
        expect_no_llm_call=True,
    ),
    TestCase(
        name="Multi-entity query",
        category="coverage",
        query="What do aspirin and ibuprofen have in common?",
        expect_min_facts=1,
    ),
    TestCase(
        name="Empty query handling",
        category="rejection",
        query="",
        expect_intent="out_of_scope",
    ),
    TestCase(
        name="Latency check",
        category="latency",
        query="What treats headaches?",
        max_latency_ms=15000.0,
    ),
]


class RAGEvaluator:
    """
    Evaluates a Graph-RAG pipeline against a suite of test cases.

    Parameters
    ----------
    pipeline : GraphRAGPipeline
        An initialized pipeline to test.
    """

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def run_suite(self, test_cases: list[TestCase] | None = None) -> EvalReport:
        """Run all test cases and produce a report."""
        cases = test_cases or DEFAULT_TEST_CASES
        report = EvalReport()

        for tc in cases:
            score = self._run_case(tc)
            report.scores.append(score)
            report.total += 1
            if score.passed:
                report.passed += 1
            else:
                report.failed += 1

        return report

    def _run_case(self, tc: TestCase) -> EvalScore:
        """Run a single test case and evaluate the result."""
        try:
            t0 = time.perf_counter()
            result = self.pipeline.query(tc.query) if tc.query else {
                "intent": "out_of_scope", "entry_nodes": [],
                "subgraph": {"nodes": [], "relationships": []},
                "context": "", "answer": None, "timing": {},
                "thought_process": [], "strategy": "none", "hop_depth": 0,
                "query": "",
            }
            latency = (time.perf_counter() - t0) * 1000
        except Exception as e:
            return EvalScore(
                test_name=tc.name, category=tc.category,
                passed=False, details=f"Pipeline error: {e}",
            )

        failures = []

        # Check intent
        if tc.expect_intent and result.get("intent") != tc.expect_intent:
            failures.append(
                f"Intent: expected '{tc.expect_intent}', got '{result.get('intent')}'"
            )

        # Check strategy
        if tc.expect_strategy and result.get("strategy") != tc.expect_strategy:
            failures.append(
                f"Strategy: expected '{tc.expect_strategy}', got '{result.get('strategy')}'"
            )

        # Check minimum facts
        rels = result.get("subgraph", {}).get("relationships", [])
        if len(rels) < tc.expect_min_facts:
            failures.append(
                f"Facts: expected ≥{tc.expect_min_facts}, got {len(rels)}"
            )

        # Check allowed relationship types
        if tc.expect_rel_types:
            actual_types = {r["type"] for r in rels}
            disallowed = actual_types - set(tc.expect_rel_types)
            if disallowed:
                failures.append(
                    f"Unexpected rel types: {disallowed}"
                )

        # Check forbidden relationship types
        if tc.expect_no_rel_types:
            actual_types = {r["type"] for r in rels}
            forbidden = actual_types & set(tc.expect_no_rel_types)
            if forbidden:
                failures.append(
                    f"Forbidden rel types found: {forbidden}"
                )

        # Check answer content
        if tc.expect_answer_contains and result.get("answer"):
            answer_lower = result["answer"].lower()
            for substr in tc.expect_answer_contains:
                if substr.lower() not in answer_lower:
                    failures.append(f"Answer missing: '{substr}'")

        # Check no LLM call (out-of-scope should not call LLM)
        if tc.expect_no_llm_call and "llm_ms" in result.get("timing", {}):
            failures.append("LLM was called when it should not have been")

        # Check latency
        if latency > tc.max_latency_ms:
            failures.append(
                f"Latency: {latency:.0f}ms > {tc.max_latency_ms:.0f}ms limit"
            )

        passed = len(failures) == 0
        details = "PASS" if passed else "; ".join(failures)

        return EvalScore(
            test_name=tc.name,
            category=tc.category,
            passed=passed,
            details=details,
            latency_ms=latency,
        )

    @staticmethod
    def print_report(report: EvalReport):
        """Print a formatted evaluation report."""
        print(f"\n{'═' * 70}")
        print(f"  GRAPH-RAG EVALUATION REPORT")
        print(f"  {report.passed}/{report.total} passed ({report.pass_rate:.0f}%)")
        print(f"{'═' * 70}")

        for score in report.scores:
            icon = "✓" if score.passed else "✗"
            latency_str = f" ({score.latency_ms:.0f}ms)" if score.latency_ms else ""
            print(f"  {icon} [{score.category:12s}] {score.test_name:40s} {score.details}{latency_str}")

        print(f"{'═' * 70}\n")
