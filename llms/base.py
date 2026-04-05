# llms/base.py
from typing import List, Protocol

class LLMClient(Protocol):
    """Minimal interface for any Democritus LLM backend."""

    def ask(self, prompt: str) -> str:
        ...

    def ask_batch(self, prompts: List[str]) -> List[str]:
        ...
