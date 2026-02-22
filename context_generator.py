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


_DEFAULT_REL_TEMPLATE = "{source} --{type}--> {target}"


class ContextGenerator:
    """
    Converts a subgraph dict produced by SmartTraversalEngine into
    structured text suitable for an LLM prompt.

    Parameters
    ----------
    config_path:
        Path to the domain config JSON file.
        Reads "relationship_templates" to produce natural-language
        relationship sentences.
    """

    def __init__(self, config_path: str):
        with open(config_path) as f:
            config = json.load(f)
        self._rel_templates: dict[str, str] = config.get("relationship_templates", {})

    def generate(self, subgraph: dict) -> str:
        """
        Convert subgraph to plain text context.

        Args:
            subgraph: Dict with "nodes" and "relationships" lists.

        Returns:
            Formatted string ready to inject into an LLM prompt.
        """
        nodes = subgraph.get("nodes", [])
        relationships = subgraph.get("relationships", [])
        strategy  = subgraph.get("strategy", "")
        hop_depth = subgraph.get("hop_depth", 1)

        if not nodes:
            return "No relevant information found in the knowledge graph."

        # id → display name for relationship sentences
        id_to_name: dict[str, str] = {
            n["id"]: n["name"] or n["label"] for n in nodes
        }

        lines: list[str] = []
        if strategy:
            lines.append(
                f"[Traversal: {strategy}, depth={hop_depth}, "
                f"{len(nodes)} nodes, {len(relationships)} edges]"
            )
            lines.append("")

        lines.append("ENTITIES:")
        for node in nodes:
            label = node["label"]
            name  = node["name"] or "(unnamed)"
            # Show extra properties (skip 'name' itself, skip empty values)
            extras = [
                f"{k}: {v}"
                for k, v in node.get("properties", {}).items()
                if k != "name" and v not in (None, "")
            ]
            detail = f" ({', '.join(extras)})" if extras else ""
            lines.append(f"  - [{label}] {name}{detail}")

        if relationships:
            lines.append("\nRELATIONSHIPS:")
            for rel in relationships:
                src_name = id_to_name.get(rel["source_id"], rel["source_id"])
                tgt_name = id_to_name.get(rel["target_id"], rel["target_id"])
                rel_type = rel["type"]
                template = self._rel_templates.get(rel_type, _DEFAULT_REL_TEMPLATE)
                sentence = template.format(
                    source=src_name,
                    target=tgt_name,
                    type=rel_type,
                )
                lines.append(f"  - {sentence}")

        return "\n".join(lines)
