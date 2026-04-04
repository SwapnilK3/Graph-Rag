"""
llm_interface.py
----------------
Component 5: LLM Interface (Google Gemini)

Sends the knowledge graph context together with the user query to Gemini.
The model is strictly instructed to answer ONLY from the provided context —
it must not use external knowledge. This keeps answers grounded and
traceable to the graph.

Usage
-----
    from llm_interface import LLMInterface

    llm = LLMInterface()           # reads GEMINI_API_KEY from .env
    answer = llm.answer(query, context)
"""

import os
from google import genai
from google.genai import types


_SYSTEM_PROMPT = """\
You are a knowledge graph assistant.
Answer the question using ONLY the information provided in the KNOWLEDGE GRAPH CONTEXT below.
Do not use any external knowledge or make assumptions beyond what is stated.
If the context does not contain enough information to answer, say so clearly.\
"""

_USER_PROMPT_TEMPLATE = """\
KNOWLEDGE GRAPH CONTEXT:
{context}

QUESTION: {query}

ANSWER:\
"""


class LLMInterface:
    """
    Wraps the Gemini API for single-turn, context-grounded Q&A.

    Parameters
    ----------
    api_key:
        Gemini API key. Falls back to the GEMINI_API_KEY environment variable.
    model:
        Gemini model name. Default is "gemini-2.5-flash" (fast, free tier).

    Raises
    ------
    ValueError
        If no API key is found in the parameter or environment.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.5-flash",
        max_retries: int = 2,
    ):
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key or key == "your_gemini_api_key_here":
            raise ValueError(
                "Gemini API key not found.\n"
                "Set GEMINI_API_KEY in your .env file.\n"
                "Get a free key at: https://aistudio.google.com/apikey"
            )
        self._client = genai.Client(api_key=key)
        self._model = model
        self._max_retries = max_retries

    def _call_with_retry(self, fn, *args, **kwargs):
        """Call a function with automatic retry on rate limit errors."""
        import time as _time
        import logging as _log
        logger = _log.getLogger(__name__)

        last_error = None
        for attempt in range(self._max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    wait = 2 ** (attempt + 1)  # 2s, 4s, 8s
                    logger.warning(
                        "Rate limited (attempt %d/%d). Retrying in %ds...",
                        attempt + 1, self._max_retries + 1, wait
                    )
                    last_error = e
                    _time.sleep(wait)
                else:
                    raise  # Non-rate-limit errors propagate immediately
        raise last_error  # All retries exhausted

    def answer(self, query: str, context: str) -> str:
        """
        Generate a grounded answer from the LLM with retry on rate limit.
        """
        prompt = _USER_PROMPT_TEMPLATE.format(context=context, query=query)

        def _call():
            response = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=types.GenerateContentConfig(system_instruction=_SYSTEM_PROMPT),
            )
            return response.text.strip()

        return self._call_with_retry(_call)

    def generate_text(self, prompt: str) -> str:
        """
        Generate a text response for a prompt with retry on rate limit.
        """
        def _call():
            response = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
            )
            return response.text.strip()

        return self._call_with_retry(_call)

    def embed_text(self, text: str, task_type: str = "RETRIEVAL_QUERY") -> list[float]:
        """
        Generate a vector embedding for a single string.
        
        Args:
            text: The string to embed.
            task_type: The purpose of the embedding (RETRIEVAL_QUERY, RETRIEVAL_DOCUMENT, etc.)
            
        Returns:
            A list of floats (embedding vector).
        """
        response = self._client.models.embed_content(
            model="text-embedding-004",
            contents=text,
            config=types.EmbedContentConfig(task_type=task_type)
        )
        return response.embeddings[0].values

    def embed_batch(self, texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
        """
        Generate embeddings for a list of strings in one call.
        """
        if not texts:
            return []
            
        response = self._client.models.embed_content(
            model="text-embedding-004",
            contents=texts,
            config=types.EmbedContentConfig(task_type=task_type)
        )
        return [e.values for e in response.embeddings]
