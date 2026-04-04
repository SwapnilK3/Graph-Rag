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
import math
from itertools import combinations
from typing import Optional

from .connector import GraphDBConnector
from .llm_interface import LLMInterface

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
        llm: Optional[LLMInterface] = None,
        search_properties: list[str] | None = None,
        node_labels: list[str] | None = None,
        extra_stop_words: list[str] | None = None,
        fuzzy_threshold: float = 0.85,
        semantic_threshold: float = 0.70,
    ):
        self.connector = connector
        self.llm = llm  # New in V2: LLM for semantic extraction/matching
        self.search_properties = search_properties or ["name"]
        self.node_labels = node_labels
        self.fuzzy_threshold = fuzzy_threshold
        self.semantic_threshold = semantic_threshold
        self.stop_words = _DEFAULT_STOP_WORDS | frozenset(
            w.lower() for w in (extra_stop_words or [])
        )

        try:
            import Levenshtein
            self._fuzzy_available = True
        except ImportError:
            self._fuzzy_available = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_entry_nodes(self, query: str) -> list[dict]:
        """
        V3 Extractor: N-gram first (free) → LLM extraction (if needed) → Label → Semantic.
        
        Optimization: Avoids LLM call when n-gram extraction finds matches.
        This reduces API calls by ~50% for common queries.
        """
        if not query or not query.strip():
            return []

        seen_ids: set[str] = set()
        entry_nodes: list[dict] = []

        # 1. Try n-gram extraction FIRST (free, no LLM call)
        ngram_keywords = self._extract_keywords(query)
        for candidate in ngram_keywords:
            nodes = self._find_in_graph_v2(candidate)
            for node in nodes:
                if node["id"] not in seen_ids:
                    seen_ids.add(node["id"])
                    entry_nodes.append(node)

        # 2. If n-grams found matches, skip LLM call (saves API quota)
        if entry_nodes:
            logger.info("N-gram extraction found %d nodes — skipping LLM call", len(entry_nodes))
            return entry_nodes

        # 3. LLM-aided extraction (only when n-grams fail)
        if self.llm:
            llm_entities = self._extract_entities_with_llm(query)
            for candidate in llm_entities:
                nodes = self._find_in_graph_v2(candidate)
                for node in nodes:
                    if node["id"] not in seen_ids:
                        seen_ids.add(node["id"])
                        entry_nodes.append(node)

        if entry_nodes:
            return entry_nodes

        # 4. Label match: check if query mentions a node TYPE (e.g., "drugs", "diseases")
        logger.info("No specific nodes found; trying label match.")
        entry_nodes = self._label_match(query)
        if entry_nodes:
            return entry_nodes

        # 5. Final: broad semantic search of the whole query
        if self.llm:
            logger.info("Label match failed; trying broad semantic search.")
            entry_nodes = self._semantic_search(query, limit=5)

        logger.info("Found %d entry node(s)", len(entry_nodes))
        return entry_nodes

    def _label_match(self, query: str) -> list[dict]:
        """
        Check if query mentions a graph label (node type) like 'drugs', 'diseases'.
        If so, return sample nodes of that type to enable pattern-based reasoning.
        """
        q_lower = query.lower()
        for label in (self.node_labels or []):
            label_lower = label.lower()
            # Match singular, plural, or substring
            if label_lower in q_lower or (label_lower + "s") in q_lower:
                cypher = f"""
                MATCH (n:{label})-[r]->()
                WITH n, count(r) AS rels
                ORDER BY rels DESC
                RETURN elementId(n) as id, labels(n)[0] as label, 
                       properties(n) as properties
                LIMIT 5
                """
                rows = self.connector.execute_query(cypher)
                if rows:
                    logger.info("Label match: found %d sample %s nodes", len(rows), label)
                    return [self._row_to_node(r) for r in rows]
        return []

    # ------------------------------------------------------------------
    # V2 Search logic
    # ------------------------------------------------------------------

    def _find_in_graph_v2(self, keyword: str) -> list[dict]:
        """Hybrid search: Exact → Partial → Fuzzy → Semantic"""
        # 1. Exact match
        nodes = self._exact_match(keyword)
        if nodes: return nodes

        # 2. Partial match
        nodes = self._partial_match(keyword)
        if nodes: return nodes

        # 3. Fuzzy match
        if self._fuzzy_available and len(keyword) > 3:
            nodes = self._fuzzy_match(keyword)
            if nodes: return nodes
            
        # 4. Semantic match (New in V2)
        if self.llm and len(keyword) > 2:
            return self._semantic_search(keyword)

        return []

    def _extract_entities_with_llm(self, query: str) -> list[str]:
        """Ask the LLM to pull out likely entity names to search for."""
        prompt = f"""Extract the core entity names (proper nouns, concepts, specific terms) from this search query.
Return ONLY a comma-separated list of names. If none, return empty.
Query: "{query}" """
        
        try:
            response = self.llm.generate_text(prompt)
            entities = [e.strip() for e in response.split(",") if e.strip()]
            logger.debug("LLM extracted entities: %s", entities)
            return entities
        except Exception as e:
            logger.warning("LLM entity extraction failed: %s", e)
            return []

    def _semantic_search(self, text: str, limit: int = 3) -> list[dict]:
        """
        V2 Semantic Search: Finds nodes whose 'name' is semantically close to 'text'.
        Note: For production, this should use a Neo4j Vector Index.
        For production, this should use a Neo4j Vector Index.
        Falls back to full-label scan with embedding cache for small graphs.
        """
        # Step 1: Get query embedding
        try:
            query_vec = self.llm.embed_text(text)
        except Exception as e:
            logger.error("Embedding failed: %s", e)
            return []

        # Step 2a: Try Neo4j native vector index first (requires index setup)
        try:
            vector_cypher = """
            CALL db.index.vector.queryNodes('entity_embeddings', $limit, $embedding)
            YIELD node, score
            WHERE score >= $threshold
            RETURN elementId(node) as id, labels(node)[0] as label, 
                   properties(node) as properties, score
            ORDER BY score DESC
            """
            rows = self.connector.execute_query(vector_cypher, {
                "limit": limit,
                "embedding": query_vec,
                "threshold": self.semantic_threshold,
            })
            if rows:
                logger.debug("Neo4j vector index returned %d results", len(rows))
                return [self._row_to_node(r) for r in rows]
        except Exception:
            logger.debug("No Neo4j vector index available; falling back to in-memory search")

        # Step 2b: Fallback — full-label scan with embedding comparison
        # Capped at 500 nodes to remain feasible for research/small graphs
        label_filter = self._build_label_filter()
        prop = self.search_properties[0]
        
        cypher = f"""
        MATCH (n{label_filter})
        WHERE n.{prop} IS NOT NULL
        RETURN elementId(n) as id, labels(n)[0] as label, properties(n) as properties, n.{prop} as name
        LIMIT 500
        """
        rows = self.connector.execute_query(cypher)
        if len(rows) >= 500:
            logger.warning("Semantic search scanned 500 nodes (cap reached). Consider creating a Neo4j vector index.")
        
        scored = []
        names = [r["name"] for r in rows if isinstance(r.get("name"), str)]
        if not names:
            return []
        
        try:
            candidate_vecs = self.llm.embed_batch(names)
            for i, row in enumerate(rows):
                if i < len(candidate_vecs):
                    score = self._cosine_similarity(query_vec, candidate_vecs[i])
                    if score >= self.semantic_threshold:
                        scored.append((score, row))
        except Exception as e:
            logger.warning("Batch embedding failed: %s", e)
            return []

        scored.sort(key=lambda x: x[0], reverse=True)
        return [self._row_to_node(r) for _, r in scored[:limit]]


    @staticmethod
    def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
        dot_product = sum(a * b for a, b in zip(v1, v2))
        magnitude1 = math.sqrt(sum(a * a for a in v1))
        magnitude2 = math.sqrt(sum(a * a for a in v2))
        if not magnitude1 or not magnitude2: return 0.0
        return dot_product / (magnitude1 * magnitude2)

    # 175: Keyword extraction (V1 logic preserved)
    def _extract_keywords(self, query: str) -> list[str]:
        cleaned = re.sub(r"[^a-zA-Z0-9\s\-]", " ", query)
        tokens = cleaned.lower().split()
        content_tokens = [t for t in tokens if t not in self.stop_words]
        if not content_tokens: content_tokens = tokens
        candidates: list[str] = []
        seen: set[str] = set()
        for n in (3, 2, 1):
            for i in range(len(content_tokens) - n + 1):
                phrase = " ".join(content_tokens[i : i + n])
                if phrase not in seen:
                    seen.add(phrase)
                    candidates.append(phrase)
        return candidates

    def _build_label_filter(self) -> str:
        if not self.node_labels: return ""
        return ":" + "|".join(self.node_labels)

    def _exact_match(self, keyword: str) -> list[dict]:
        label_filter = self._build_label_filter()
        conditions = " OR ".join(f"toLower(n.{prop}) = $keyword" for prop in self.search_properties)
        cypher = f"MATCH (n{label_filter}) WHERE {conditions} RETURN elementId(n) AS id, labels(n)[0] AS label, properties(n) AS properties LIMIT 10"
        rows = self.connector.execute_query(cypher, {"keyword": keyword.lower()})
        return [self._row_to_node(r) for r in rows]

    def _partial_match(self, keyword: str) -> list[dict]:
        label_filter = self._build_label_filter()
        conditions = " OR ".join(f"toLower(n.{prop}) CONTAINS $keyword" for prop in self.search_properties)
        cypher = f"MATCH (n{label_filter}) WHERE {conditions} RETURN elementId(n) AS id, labels(n)[0] AS label, properties(n) AS properties LIMIT 10"
        rows = self.connector.execute_query(cypher, {"keyword": keyword.lower()})
        return [self._row_to_node(r) for r in rows]

    def _fuzzy_match(self, keyword: str) -> list[dict]:
        import Levenshtein
        primary_prop = self.search_properties[0]
        label_filter = self._build_label_filter()
        kw_len = len(keyword)
        min_len, max_len = max(1, kw_len - kw_len // 2), kw_len + kw_len // 2
        cypher = f"MATCH (n{label_filter}) WHERE size(toLower(n.{primary_prop})) >= $min_len AND size(toLower(n.{primary_prop})) <= $max_len RETURN elementId(n) AS id, labels(n)[0] AS label, properties(n) AS properties, toLower(n.{primary_prop}) AS candidate_name LIMIT 200"
        rows = self.connector.execute_query(cypher, {"min_len": min_len, "max_len": max_len})
        scored = []
        kw_lower = keyword.lower()
        for row in rows:
            name = row.get("candidate_name")
            if name:
                ratio = Levenshtein.ratio(kw_lower, name)
                if ratio >= self.fuzzy_threshold: scored.append((ratio, row))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [self._row_to_node(r) for _, r in scored[:5]]

    @staticmethod
    def _row_to_node(row: dict) -> dict:
        props = row.get("properties") or {}
        return {"id": row["id"], "label": row.get("label", "Unknown"), "name": props.get("name", ""), "properties": props}
