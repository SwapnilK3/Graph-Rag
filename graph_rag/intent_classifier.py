import json
import logging
from typing import Optional
from .llm_interface import LLMInterface

logger = logging.getLogger(__name__)

def _load_config(config_path_or_dict) -> dict:
    if isinstance(config_path_or_dict, dict):
        return config_path_or_dict
    with open(config_path_or_dict) as f:
        return json.load(f)


class IntentClassifier:
    """
    Classifies a user query into a named intent.
    V2.5: Abstract query detection + LLM inference + keyword fallback.
    """

    # Markers that indicate an abstract/meta query (pattern reasoning, not triple retrieval)
    ABSTRACT_MARKERS = frozenset({
        "why", "purpose", "reason", "explain", "how come",
        "what is the role", "what are the roles", "what is the use",
        "what are the uses", "overview", "summary", "describe",
    })

    def __init__(self, config: str | dict, llm: Optional[LLMInterface] = None):
        loaded = _load_config(config)
        self.llm = llm
        
        # Keep V1 patterns as a robust fallback
        self._patterns: dict[str, list[str]] = {
            name: [kw.lower() for kw in pattern["keywords"]]
            for name, pattern in loaded["intent_patterns"].items()
        }
        
        # Meta-information for LLM classification
        self._intent_descriptions = {
            name: pattern.get("description", f"Search for {name}")
            for name, pattern in loaded["intent_patterns"].items()
        }

    def classify(self, query: str) -> str:
        """
        Hybrid Classification: Abstract Detection → LLM Inference → Keyword Match → General.
        """
        q_lower = query.lower()

        # Priority 1: Abstract/meta query detection
        if any(marker in q_lower for marker in self.ABSTRACT_MARKERS):
            # Only use abstract if no specific intent keyword also matches
            has_specific = any(
                any(kw in q_lower for kw in keywords)
                for keywords in self._patterns.values()
            )
            if not has_specific:
                logger.info("Abstract query detected: '%s'", query)
                return "abstract"

        # Priority 2: LLM inference
        if self.llm:
            intent = self._classify_with_llm(query)
            if intent and intent in self._patterns:
                logger.info("LLM classified intent: %s", intent)
                return intent

        # Priority 3: Keyword Matching
        for intent, keywords in self._patterns.items():
            if any(kw in q_lower for kw in keywords):
                logger.debug("Keyword match for intent: %s", intent)
                return intent
                
        return "general"

    def _classify_with_llm(self, query: str) -> Optional[str]:
        """Ask the LLM to pick the best intent from the available list."""
        intent_list = "\n".join([f"- {name}: {desc}" for name, desc in self._intent_descriptions.items()])
        
        prompt = f"""Given the user query, pick the BEST intent from the list below that matches the search goal.
Available Intents:
{intent_list}
- general: Use this if nothing else matches.

Return ONLY the single name of the intent (e.g., 'side_effects').
Query: "{query}" """

        try:
            response = self.llm.generate_text(prompt).lower()
            
            # Clean up response (some models might add punctuation)
            clean_intent = "".join(c for c in response if c.isalnum() or c == "_")
            return clean_intent if clean_intent in self._patterns else None
        except Exception as e:
            logger.warning("LLM intent classification failed: %s", e)
            return None

    def all_intents(self) -> list[str]:
        return list(self._patterns.keys())
