"""
auto_config.py
--------------
Component 0.5: Automatic Config Generator

Takes the schema dict from SchemaDiscovery and generates the complete
config that IntentClassifier, TraversalEngine, and ContextGenerator need.

This replaces manually-written config/medical_graph.json with a
programmatically-generated equivalent — making the entire pipeline
schema-agnostic and zero-configuration.

Intent Generation Strategy:
---------------------------
For each discovered relationship (source_label)-[:REL_TYPE]->(target_label):
  1. Create a FORWARD intent:   "source_rel_type"  (e.g. "drug_causes")
  2. Create a REVERSE intent:   "target_rel_type"  (e.g. "sideeffect_caused_by")
  3. Generate keywords from label and relationship names
  4. Use "targeted" strategy for single-hop intents

Additionally:
  5. Create "neighborhood" intent (variable_hop, 2-hop exploration)
  6. Create "connection" intent (shortest_path between nodes)
  7. Create "shared" intent (shared_neighbor between entries)
  8. Detect multi-hop chains and create chained intents
"""

from __future__ import annotations

import re
import logging
from itertools import combinations

logger = logging.getLogger(__name__)


def _humanize(name: str) -> str:
    """
    Convert SCREAMING_SNAKE or CamelCase to lowercase words.
    TREATS -> "treats", INTERACTS_WITH -> "interacts with",
    SideEffect -> "side effect"
    """
    # First handle CamelCase
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    # Then handle SCREAMING_SNAKE
    return name.replace("_", " ").lower().strip()


def _make_intent_name(prefix: str, rel_type: str) -> str:
    """Build a clean intent key.  e.g. ("drug", "CAUSES") -> "drug_causes" """
    return f"{prefix.lower()}_{rel_type.lower()}"


class AutoConfigGenerator:
    """
    Generates a complete pipeline config dict from a discovered schema.

    Usage
    -----
        from schema_discovery import SchemaDiscovery
        from auto_config import AutoConfigGenerator

        schema = SchemaDiscovery(connector).discover()
        config = AutoConfigGenerator(schema).generate()
    """

    def __init__(self, schema: dict):
        self.schema = schema

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> dict:
        """
        Build the full config dict that all downstream components consume.
        """
        intents    = {}
        templates  = {}
        rels       = self.schema["relationships"]

        # ── 1. Targeted intents from each relationship ────────────────

        for rel in rels:
            rel_type = rel["type"]
            src      = rel["source_label"]
            tgt      = rel["target_label"]
            human    = _humanize(rel_type)

            # Forward intent: Drug causes SideEffect → "drug_causes"
            fwd_name = _make_intent_name(src, rel_type)
            fwd_keywords = self._forward_keywords(src, tgt, rel_type, human)
            if fwd_name not in intents:
                intents[fwd_name] = {
                    "strategy":      "targeted",
                    "keywords":      fwd_keywords,
                    "source_label":  src,
                    "relationship":  rel_type,
                    "target_label":  tgt,
                    "entry_anchor":  "source",
                }

            # Reverse intent: what Drug treats Disease → "disease_treated_by"
            rev_name = _make_intent_name(tgt, f"{rel_type}_reverse")
            rev_keywords = self._reverse_keywords(src, tgt, rel_type, human)
            if rev_name not in intents:
                intents[rev_name] = {
                    "strategy":      "targeted",
                    "keywords":      rev_keywords,
                    "source_label":  src,
                    "relationship":  rel_type,
                    "target_label":  tgt,
                    "entry_anchor":  "target",
                }

            # Relationship template
            if rel_type not in templates:
                templates[rel_type] = self._make_template(rel_type, human)

        # ── 2. Detect and create chained (multi-hop) intents ──────────

        chains = self._detect_chains(rels)
        for chain in chains:
            chain_name = "_to_".join(
                _humanize(h["relationship"]).replace(" ", "_")
                for h in chain["hops"]
            )
            intent_name = f"chain_{chain_name}"
            intents[intent_name] = {
                "strategy":    "chained",
                "keywords":    chain["keywords"],
                "entry_label": chain["entry_label"],
                "hops":        chain["hops"],
            }

        # ── 3. Structural intents (strategy-based) ────────────────────

        all_labels_lower = [_humanize(l) for l in self.schema["node_labels"]]

        intents["neighborhood"] = {
            "strategy":  "variable_hop",
            "keywords":  [
                "everything about", "overview of", "related to",
                "all about", "full profile", "tell me about",
                "information about", "details of",
            ],
            "min_hops": 1,
            "max_hops": 2,
        }

        intents["connection"] = {
            "strategy":  "shortest_path",
            "keywords":  [
                "connected", "how is", "what connects", "path between",
                "link between", "relationship between", "how are",
                "related to each other",
            ],
            "max_hops": 6,
        }

        intents["shared"] = {
            "strategy":       "shared_neighbor",
            "keywords":       [
                "in common", "share", "both", "same",
                "common", "mutual", "overlap",
            ],
            "min_connections": 2,
        }

        # ── 4. Order intents: multi-hop first, then specific, then general ──

        ordered_intents = self._order_intents(intents)

        config = {
            "domain":                 "auto-discovered",
            "description":            self._make_description(),
            "auto_generated":         True,
            "intent_patterns":        ordered_intents,
            "relationship_templates": templates,
            "general_traversal":      {"node_limit": 50},
        }

        logger.info(
            "Auto-generated config: %d intents, %d templates",
            len(ordered_intents), len(templates),
        )
        return config

    # ------------------------------------------------------------------
    # Keyword generation
    # ------------------------------------------------------------------

    def _forward_keywords(
        self, src: str, tgt: str, rel_type: str, human_rel: str,
    ) -> list[str]:
        """
        Generate keywords for forward traversal.
        e.g. Drug-[CAUSES]->SideEffect:
          ["side effect of", "causes", "what does * cause", ...]
        """
        src_h = _humanize(src)
        tgt_h = _humanize(tgt)
        kws = set()

        # Relationship-based
        kws.add(human_rel)
        for word in human_rel.split():
            if len(word) > 3:
                kws.add(word)

        # Target-based
        kws.add(f"{tgt_h} of")
        kws.add(f"what {tgt_h}")

        # Combined
        kws.add(f"what does it {human_rel}")
        kws.add(f"{human_rel} {tgt_h}")

        return sorted(kws)

    def _reverse_keywords(
        self, src: str, tgt: str, rel_type: str, human_rel: str,
    ) -> list[str]:
        """
        Generate keywords for reverse traversal.
        e.g. Drug-[TREATS]->Disease (reverse = Disease → what Drug treats it):
          ["what treats", "treatment for", "treated by", ...]
        """
        src_h = _humanize(src)
        tgt_h = _humanize(tgt)
        kws = set()

        # Passive voice
        past = self._past_participle(human_rel)
        kws.add(f"{past} by")
        kws.add(f"what {human_rel}")

        # Source-based
        kws.add(f"which {src_h}")
        kws.add(f"what {src_h}")

        return sorted(kws)

    @staticmethod
    def _past_participle(verb: str) -> str:
        """Naive past participle: treats→treated, causes→caused, etc."""
        verb = verb.strip()
        if verb.endswith("es"):
            return verb[:-1] + "d"   # causes → caused
        if verb.endswith("s"):
            return verb[:-1] + "ed"  # treats → treated
        if verb.endswith("e"):
            return verb + "d"
        return verb + "ed"

    # ------------------------------------------------------------------
    # Chain detection (multi-hop)
    # ------------------------------------------------------------------

    def _detect_chains(self, rels: list[dict]) -> list[dict]:
        """
        Find 2-hop chains: A-[R1]->B  and  B-[R2]->C where B is the
        same label.  These become chained traversal intents.

        Example:  Drug-[CAUSES]->SideEffect  +  SideEffect-[INCREASES_RISK_OF]->Disease
                  → 2-hop chain from Drug through SideEffect to Disease
        """
        chains = []
        # Index: source_label → list of rels
        by_source: dict[str, list[dict]] = {}
        for r in rels:
            by_source.setdefault(r["source_label"], []).append(r)

        for r1 in rels:
            intermediate = r1["target_label"]
            for r2 in by_source.get(intermediate, []):
                if r2["type"] == r1["type"]:
                    continue  # skip self-loops in type
                if r2["target_label"] == r1["source_label"]:
                    continue  # skip circular

                h1 = _humanize(r1["type"])
                h2 = _humanize(r2["type"])
                final_h = _humanize(r2["target_label"])

                chains.append({
                    "entry_label": r1["source_label"],
                    "hops": [
                        {"relationship": r1["type"], "target_label": intermediate},
                        {"relationship": r2["type"], "target_label": r2["target_label"]},
                    ],
                    "keywords": [
                        f"indirectly {h2}",
                        f"{h1} then {h2}",
                        f"through {_humanize(intermediate)}",
                        f"leads to {final_h}",
                    ],
                })

        logger.debug("Detected %d multi-hop chains", len(chains))
        return chains

    # ------------------------------------------------------------------
    # Template generation
    # ------------------------------------------------------------------

    @staticmethod
    def _make_template(rel_type: str, human_rel: str) -> str:
        """Build a natural-language template string."""
        # Common relationship patterns
        templates_map = {
            "treats":            "{source} treats {target}",
            "causes":            "{source} may cause {target}",
            "interacts with":    "{source} interacts with {target}",
            "has symptom":       "{source} has symptom: {target}",
            "increases risk of": "{source} increases risk of {target}",
            "metabolized by":    "{source} is metabolized by {target}",
            "belongs to":        "{source} belongs to {target}",
            "acted in":          "{source} acted in {target}",
            "directed":          "{source} directed {target}",
            "works at":          "{source} works at {target}",
            "located in":        "{source} is located in {target}",
        }

        if human_rel in templates_map:
            return templates_map[human_rel]

        # Fallback: use the humanized relationship name directly
        return "{source} " + human_rel + " {target}"

    # ------------------------------------------------------------------
    # Intent ordering
    # ------------------------------------------------------------------

    @staticmethod
    def _order_intents(intents: dict) -> dict:
        """
        Order intents so more-specific ones come first:
          1. Chained (multi-hop) — most specific keywords
          2. Structural (shared, connection, neighborhood)
          3. Targeted forward / reverse — broadest keywords
        """
        priority = {"chained": 0, "shared_neighbor": 1, "shortest_path": 2,
                     "variable_hop": 3, "targeted": 4}
        sorted_items = sorted(
            intents.items(),
            key=lambda kv: (priority.get(kv[1].get("strategy", "targeted"), 5), kv[0]),
        )
        return dict(sorted_items)

    # ------------------------------------------------------------------
    # Description
    # ------------------------------------------------------------------

    def _make_description(self) -> str:
        labels = ", ".join(self.schema["node_labels"])
        return f"Auto-discovered graph with labels: {labels}"

    # ------------------------------------------------------------------
    # Pretty print
    # ------------------------------------------------------------------

    def print_config(self, config: dict) -> None:
        """Human-readable config summary."""
        print(f"\n{'─' * 60}")
        print(f"  AUTO-GENERATED CONFIG")
        print(f"  Domain: {config['domain']}")
        print(f"{'─' * 60}")

        print(f"\n  INTENTS ({len(config['intent_patterns'])}):")
        for name, pattern in config["intent_patterns"].items():
            strategy = pattern.get("strategy", "targeted")
            keywords = pattern.get("keywords", [])[:3]
            print(f"    {name:30s} [{strategy:15s}]  keywords: {keywords}")

        print(f"\n  RELATIONSHIP TEMPLATES ({len(config['relationship_templates'])}):")
        for rel_type, template in config["relationship_templates"].items():
            print(f"    {rel_type:25s} → {template}")
        print(f"{'─' * 60}\n")
