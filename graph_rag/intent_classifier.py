"""
intent_classifier.py
--------------------
Component 2: Query Intent Classifier

Reads intent keyword patterns from a domain config JSON.
Returns the first matching intent name, or "general" as fallback.

Simple, deterministic: no ML, no ambiguity.
"""

import json


def _load_config(config_path_or_dict) -> dict:
    """Accept a file path (str) or a config dict directly."""
    if isinstance(config_path_or_dict, dict):
        return config_path_or_dict
    with open(config_path_or_dict) as f:
        return json.load(f)


class IntentClassifier:
    """
    Classifies a user query into a named intent by checking
    whether the query contains any of the intent's keywords.

    Parameters
    ----------
    config:
        Path to a domain config JSON file (str), OR a config dict.
        Must contain an "intent_patterns" dict where each
        key is an intent name and the value has a "keywords" list.
    """

    def __init__(self, config: str | dict):
        loaded = _load_config(config)

        # Build {intent_name: [keyword, ...]} — lowercase for fast matching
        self._patterns: dict[str, list[str]] = {
            name: [kw.lower() for kw in pattern["keywords"]]
            for name, pattern in loaded["intent_patterns"].items()
        }

    def classify(self, query: str) -> str:
        """
        Return the intent name whose keywords appear first in the query,
        or "general" if no keyword matches.

        Args:
            query: Raw user query string.

        Returns:
            Intent name string, e.g. "side_effects", "treatment", "general".
        """
        q = query.lower()
        for intent, keywords in self._patterns.items():
            if any(kw in q for kw in keywords):
                return intent
        return "general"

    def all_intents(self) -> list[str]:
        """Return all known intent names (excluding 'general')."""
        return list(self._patterns.keys())
