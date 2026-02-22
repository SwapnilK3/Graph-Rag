"""
intent_classifier.py
--------------------
Component 2: Query Intent Classifier

Reads intent keyword patterns from a domain config JSON.
Returns the first matching intent name, or "general" as fallback.

Simple, deterministic: no ML, no ambiguity.
"""

import json


class IntentClassifier:
    """
    Classifies a user query into a named intent by checking
    whether the query contains any of the intent's keywords.

    Parameters
    ----------
    config_path:
        Path to the domain config JSON file.
        The file must contain an "intent_patterns" dict where each
        key is an intent name and the value has a "keywords" list.
    """

    def __init__(self, config_path: str):
        with open(config_path) as f:
            config = json.load(f)

        # Build {intent_name: [keyword, ...]} — lowercase for fast matching
        self._patterns: dict[str, list[str]] = {
            name: [kw.lower() for kw in pattern["keywords"]]
            for name, pattern in config["intent_patterns"].items()
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
