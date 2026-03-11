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

    def answer(self, query: str, context: str) -> str:
        """
        Generate a grounded answer from the LLM.

        Args:
            query:   The original user question.
            context: The knowledge graph context string from ContextGenerator.

        Returns:
            The LLM's answer as a plain string.
        """
        prompt = _USER_PROMPT_TEMPLATE.format(context=context, query=query)
        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(system_instruction=_SYSTEM_PROMPT),
        )
        return response.text.strip()
