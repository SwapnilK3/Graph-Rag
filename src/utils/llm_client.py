"""
src/utils/llm_client.py
-----------------------
Shared LLM client wrapping Google Gemini for the methodology pipeline.

Used by:
  - TripleExtractor  (KG construction)
  - SchemaInducer    (type induction)
  - SchemaAdapter    (evolution)
  - Graph-RAG layer  (answer generation)
"""

import os
import json
import re
from google import genai
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    """
    Thin wrapper around Google Gemini for structured & freeform generation.

    Parameters
    ----------
    api_key : str, optional
        Gemini API key. Falls back to GEMINI_API_KEY env var.
    model : str
        Model name (default: gemini-2.5-flash).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.5-flash",
    ):
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "Gemini API key not found. "
                "Set GEMINI_API_KEY in .env or pass api_key=."
            )
        self._client = genai.Client(api_key=key)
        self._model = model

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------
    def generate(self, prompt: str, system: str | None = None) -> str:
        """Generate text with an optional system instruction."""
        config = None
        if system:
            from google.genai import types
            config = types.GenerateContentConfig(
                system_instruction=system,
            )
        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=config,
        )
        return response.text.strip()

    def generate_json(self, prompt: str, system: str | None = None) -> dict | list:
        """
        Generate text and parse as JSON.
        Strips markdown code fences if present.
        """
        text = self.generate(prompt, system)
        # Strip ```json ... ``` fences
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        return json.loads(text)
