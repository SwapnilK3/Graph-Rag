"""
entity_extractor.py
-------------------
Component 1: Query-Time Entity Extractor

Responsibility
--------------
Given a raw user query string, identify the graph nodes that serve as
*entry points* for traversal.

Algorithm
---------
1. Tokenize the query into candidate terms.
2. Remove common stop-words that carry no entity information.
3. Build a ranked list of candidate strings by combining:
   - individual keywords (unigrams)
   - adjacent pairs   (bigrams)
   - adjacent triples (trigrams)
   Longer n-grams are tried first because multi-word entity names
   (e.g. "stomach bleeding", "blood pressure") should win over their
   component words.
4. For each candidate, first attempt an EXACT, case-insensitive match
   against configurable node properties (default: `name`).
5. If exact match yields nothing, attempt a PARTIAL (CONTAINS) match.
6. Optionally fall back to Levenshtein fuzzy matching when the
   python-Levenshtein package is available.
7. De-duplicate results and return.

Design goals
------------
* No heavy NLP dependency required (spaCy, NLTK, etc.)
* Target latency < 100 ms
* Validates entity existence in the graph (avoids hallucinated nodes)
* Configurable: which properties to search is driven by config/…json
"""

from __future__ import annotations

import re
import logging
from itertools import combinations
from typing import Optional

from connector import GraphDBConnector

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default stop-word list (English)
# Deliberately conservative: we remove only words that are almost never
# entity names.  Domain-specific stop-words can be added via the constructor.
# ---------------------------------------------------------------------------
_DEFAULT_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "to", "of", "in", "on", "at", "by", "for", "with", "about",
        "against", "between", "into", "through", "during", "before", "after",
        "above", "below", "from", "up", "down", "out", "off", "over", "under",
        "again", "further", "then", "once", "and", "but", "or", "nor", "so",
        "yet", "both", "either", "neither", "not", "only", "own", "same",
        "than", "too", "very", "just", "because", "as", "until", "while",
        "if", "when", "where", "how", "what", "which", "who", "whom", "that",
        "this", "these", "those", "i", "me", "my", "myself", "we", "our",
        "you", "your", "he", "him", "his", "she", "her", "it", "its", "they",
        "them", "their", "tell", "give", "show", "find", "get", "let",
        "make", "know", "see", "take", "come", "go", "say", "ask",
    }
)


class QueryTimeEntityExtractor:
    """
    Extracts graph entry-nodes from a natural-language user query.

    Parameters
    ----------
    connector:
        An active GraphConnector instance.
    search_properties:
        List of node properties to search against (e.g. ["name", "brand_name"]).
        Defaults to ["name"].
    node_labels:
        Optional list of Neo4j labels to restrict search to.
        When None, all node types are searched.
    extra_stop_words:
        Additional domain-specific words to ignore (e.g. ["drug", "disease"]).
    fuzzy_threshold:
        Levenshtein similarity threshold (0–1) for fuzzy matching.
        Only used when python-Levenshtein is installed.
    """

    def __init__(
        self,
        connector: GraphDBConnector,
        search_properties: list[str] | None = None,
        node_labels: list[str] | None = None,
        extra_stop_words: list[str] | None = None,
        fuzzy_threshold: float = 0.85,
    ):
        self.connector = connector
        self.search_properties = search_properties or ["name"]
        self.node_labels = node_labels  # None = any label
        self.fuzzy_threshold = fuzzy_threshold
        self.stop_words = _DEFAULT_STOP_WORDS | frozenset(
            w.lower() for w in (extra_stop_words or [])
        )

        # Check fuzzy-matching availability once at construction time.
        try:
            import Levenshtein  # noqa: F401
            self._fuzzy_available = True
        except ImportError:
            self._fuzzy_available = False
            logger.debug(
                "python-Levenshtein not installed; fuzzy matching disabled."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_entry_nodes(self, query: str) -> list[dict]:
        """
        Main entry point.

        Args:
            query: Raw user query string.

        Returns:
            Ordered list of unique node dicts.  Each dict contains at least:
                - id       (Neo4j internal element ID as string)
                - label    (first label of the node, e.g. "Drug")
                - name     (value of the `name` property, if present)
                - properties (full property dict)
        """
        if not query or not query.strip():
            return []

        keywords = self._extract_keywords(query)
        logger.debug("Extracted keyword candidates: %s", keywords)

        seen_ids: set[str] = set()
        entry_nodes: list[dict] = []

        for keyword in keywords:
            nodes = self._find_in_graph(keyword)
            for node in nodes:
                node_id = node["id"]
                if node_id not in seen_ids:
                    seen_ids.add(node_id)
                    entry_nodes.append(node)

        if not entry_nodes:
            logger.info("No entry nodes found for query: %r", query)
        else:
            logger.info(
                "Found %d entry node(s): %s",
                len(entry_nodes),
                [n.get("name", n["id"]) for n in entry_nodes],
            )

        return entry_nodes

    # ------------------------------------------------------------------
    # Keyword extraction
    # ------------------------------------------------------------------

    def _extract_keywords(self, query: str) -> list[str]:
        """
        Tokenize the query and return candidate entity strings, ordered
        from longest (trigram) to shortest (unigram) so that multi-word
        entities are matched before their component words.
        """
        # Normalise whitespace; keep hyphens inside words (e.g. "co-codamol").
        cleaned = re.sub(r"[^a-zA-Z0-9\s\-]", " ", query)
        tokens = cleaned.lower().split()

        # Filter stop-words; keep numeric tokens (dosages, years, etc.)
        content_tokens = [t for t in tokens if t not in self.stop_words]

        if not content_tokens:
            # If stop-word removal wiped everything, fall back to all tokens.
            content_tokens = tokens

        # Build n-grams (n = 3 → 2 → 1), longest first so graph lookup
        # favours multi-word entity names.
        candidates: list[str] = []
        seen: set[str] = set()

        for n in (3, 2, 1):
            for i in range(len(content_tokens) - n + 1):
                phrase = " ".join(content_tokens[i : i + n])
                if phrase not in seen:
                    seen.add(phrase)
                    candidates.append(phrase)

        return candidates

    # ------------------------------------------------------------------
    # Graph lookup
    # ------------------------------------------------------------------

    def _find_in_graph(self, keyword: str) -> list[dict]:
        """
        Try exact → partial → fuzzy match for a single keyword string.

        Returns a (possibly empty) list of node dicts.
        """
        # 1. Exact match
        nodes = self._exact_match(keyword)
        if nodes:
            logger.debug("Exact match for %r: %d result(s)", keyword, len(nodes))
            return nodes

        # 2. Partial (CONTAINS) match
        nodes = self._partial_match(keyword)
        if nodes:
            logger.debug("Partial match for %r: %d result(s)", keyword, len(nodes))
            return nodes

        # 3. Fuzzy match (if available and keyword is long enough to be meaningful)
        if self._fuzzy_available and len(keyword) > 3:
            nodes = self._fuzzy_match(keyword)
            if nodes:
                logger.debug("Fuzzy match for %r: %d result(s)", keyword, len(nodes))
                return nodes

        return []

    def _build_label_filter(self) -> str:
        """Return an optional Cypher label restriction clause segment."""
        if not self.node_labels:
            return ""
        labels = "|".join(self.node_labels)
        return f":{labels}"

    def _exact_match(self, keyword: str) -> list[dict]:
        """
        MATCH nodes where ANY of the search_properties equals the keyword
        (case-insensitive).
        """
        label_filter = self._build_label_filter()
        conditions = " OR ".join(
            f"toLower(n.{prop}) = $keyword"
            for prop in self.search_properties
        )
        cypher = f"""
            MATCH (n{label_filter})
            WHERE {conditions}
            RETURN
                elementId(n)            AS id,
                labels(n)[0]            AS label,
                properties(n)           AS properties
            LIMIT 10
        """
        rows = self.connector.execute_query(cypher, {"keyword": keyword.lower()})
        return [self._row_to_node(r) for r in rows]

    def _partial_match(self, keyword: str) -> list[dict]:
        """
        MATCH nodes where ANY of the search_properties *contains* the keyword.
        """
        label_filter = self._build_label_filter()
        conditions = " OR ".join(
            f"toLower(n.{prop}) CONTAINS $keyword"
            for prop in self.search_properties
        )
        cypher = f"""
            MATCH (n{label_filter})
            WHERE {conditions}
            RETURN
                elementId(n)            AS id,
                labels(n)[0]            AS label,
                properties(n)           AS properties
            LIMIT 10
        """
        rows = self.connector.execute_query(cypher, {"keyword": keyword.lower()})
        return [self._row_to_node(r) for r in rows]

    def _fuzzy_match(self, keyword: str) -> list[dict]:
        """
        Pull candidate nodes whose primary search property has a similar
        length to the keyword (cheap pre-filter), then score them locally
        with Levenshtein similarity.
        """
        import Levenshtein  # type: ignore

        primary_prop = self.search_properties[0]
        label_filter = self._build_label_filter()

        # Pre-filter: only fetch nodes whose property length is within ±50% of
        # the keyword length to limit the candidate pool.
        kw_len = len(keyword)
        min_len = max(1, kw_len - kw_len // 2)
        max_len = kw_len + kw_len // 2

        cypher = f"""
            MATCH (n{label_filter})
            WHERE size(toLower(n.{primary_prop})) >= $min_len
              AND size(toLower(n.{primary_prop})) <= $max_len
            RETURN
                elementId(n)            AS id,
                labels(n)[0]            AS label,
                properties(n)           AS properties,
                toLower(n.{primary_prop}) AS candidate_name
            LIMIT 200
        """
        rows = self.connector.execute_query(
            cypher, {"min_len": min_len, "max_len": max_len}
        )

        scored = []
        kw_lower = keyword.lower()
        for row in rows:
            candidate_name = row.get("candidate_name", "")
            if not candidate_name:
                continue
            ratio = Levenshtein.ratio(kw_lower, candidate_name)
            if ratio >= self.fuzzy_threshold:
                scored.append((ratio, row))

        # Sort best matches first; return top 5.
        scored.sort(key=lambda x: x[0], reverse=True)
        return [self._row_to_node(r) for _, r in scored[:5]]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_node(row: dict) -> dict:
        """Convert a raw Cypher result row into a clean node dict."""
        props = row.get("properties") or {}
        return {
            "id": row["id"],
            "label": row.get("label", "Unknown"),
            "name": props.get("name", ""),
            "properties": props,
        }
