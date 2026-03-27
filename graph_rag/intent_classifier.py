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
    V2 features: LLM-based 'Inference' classification with keyword fallback.
    """

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
        Hybrid Classification: LLM Inference → Keyword Match → General.
        """
        if self.llm:
            intent = self._classify_with_llm(query)
            if intent and intent in self._patterns:
                logger.info("LLM classified intent: %s", intent)
                return intent

        # Fallback to V1 Keyword Matching
        q = query.lower()
        for intent, keywords in self._patterns.items():
            if any(kw in q for kw in keywords):
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
