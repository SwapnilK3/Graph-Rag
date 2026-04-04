"""
context_generator.py
--------------------
Component 4: Context Generator

Converts a subgraph dict into structured, LLM-readable plain text.
Uses relationship templates from the domain config JSON so the output
language matches the domain (e.g. "Aspirin treats Headache" vs a generic
arrow notation).

Output structure
----------------
ENTITIES:
  - [Drug] Aspirin (type: NSAID)
  - [Disease] Headache
  - [SideEffect] Nausea

RELATIONSHIPS:
  - Aspirin treats Headache
  - Aspirin may cause Nausea
"""

import json
import logging
from typing import Optional
from .llm_interface import LLMInterface

logger = logging.getLogger(__name__)

_DEFAULT_REL_TEMPLATE = "{source} --{type}--> {target}"


def _load_config(config_path_or_dict) -> dict:
    if isinstance(config_path_or_dict, dict):
        return config_path_or_dict
    with open(config_path_or_dict) as f:
        return json.load(f)


class ContextGenerator:
    """
    Converts a subgraph dict into structured text.
    V2 features: LLM-based narrative synthesis for more coherent prompts.
    """

    def __init__(self, config: str | dict, llm: Optional[LLMInterface] = None):
        loaded = _load_config(config)
        self.llm = llm
        self._rel_templates: dict[str, str] = loaded.get("relationship_templates", {})

    def generate(self, subgraph: dict) -> str:
        """
        Main entry point. Always uses template-based factual generation
        to preserve zero-hallucination grounding.
        """
        return self.generate_template(subgraph)

    def generate_template(self, subgraph: dict) -> str:
        """Standard V1 template-based generation with deduplication and compression."""
        nodes = subgraph.get("nodes", [])
        relationships = subgraph.get("relationships", [])
        
        if not nodes:
            return ""  # Domain guard handles the "out of scope" messaging

        id_to_name = {n["id"]: n["name"] or n["label"] for n in nodes}
        lines = ["ENTITIES:"]
        for node in nodes:
            label, name = node["label"], node["name"] or "(unnamed)"
            lines.append(f"  - [{label}] {name}")

        if relationships:
            # Deduplicate: same source→target→type = one fact
            seen_rels: set[tuple] = set()
            unique_rels: list[dict] = []
            for rel in relationships:
                key = (rel["source_id"], rel["target_id"], rel["type"])
                if key not in seen_rels:
                    seen_rels.add(key)
                    unique_rels.append(rel)
            
            # Cap at 20 to limit token usage
            if len(unique_rels) > 20:
                unique_rels = unique_rels[:20]

            lines.append("\nRELATIONSHIPS:")
            for rel in unique_rels:
                src = id_to_name.get(rel["source_id"], "Unknown")
                tgt = id_to_name.get(rel["target_id"], "Unknown")
                tmpl = self._rel_templates.get(rel["type"], _DEFAULT_REL_TEMPLATE)
                lines.append(f"  - {tmpl.format(source=src, target=tgt, type=rel['type'])}")

        return "\n".join(lines)

