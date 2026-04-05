# llms/factory.py

from typing import Any, cast
from .base import LLMClient
from .openai_client import OpenAIChatClient


def make_llm_client(**kwargs: Any) -> LLMClient:
    """
    Construct the default LLM backend for Democritus v1.5.

    kwargs are forwarded to OpenAIChatClient, e.g.:

        make_llm_client(max_tokens=128, max_batch_size=16)

    Env-based defaults:
      OPENAI_API_KEY
      DEMOC_LLM_BASE_URL
      DEMOC_LLM_MODEL
      DEMOC_LLM_MAX_TOKENS
      DEMOC_LLM_TEMPERATURE
      DEMOC_LLM_BATCH_SIZE
    """
    return cast(LLMClient, OpenAIChatClient(**kwargs))
