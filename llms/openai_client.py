# llms/openai_client.py

import json
import os
from typing import List, Optional
from urllib.error import HTTPError
from urllib.request import Request, urlopen

try:
    import requests
except ModuleNotFoundError:  # pragma: no cover - exercised in CLIFF integration envs
    requests = None


class OpenAIChatClient:
    """
    Generic client for OpenAI-style /v1/chat/completions endpoint.

    Env vars (recommended):

      OPENAI_API_KEY        – your API key
      DEMOC_LLM_BASE_URL    – default: https://api.openai.com
      DEMOC_LLM_MODEL       – default: gpt-4.1-mini
      DEMOC_LLM_MAX_TOKENS  – default: 256
      DEMOC_LLM_TEMPERATURE – default: 0.7
      DEMOC_LLM_BATCH_SIZE  – default: 4 (controls how many we loop over at a time)
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        max_batch_size: Optional[int] = None,
        timeout: int = 120,
    ) -> None:
        self.base_url = (
            base_url
            or os.getenv("DEMOC_LLM_BASE_URL")
            or "https://api.openai.com"
        ).rstrip("/")

        self.model = (
            model
            or os.getenv("DEMOC_LLM_MODEL")
            or "gpt-4.1-mini"
        )

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Please export it before running Democritus."
            )

        self.max_tokens = max_tokens or int(os.getenv("DEMOC_LLM_MAX_TOKENS", "256"))
        self.temperature = temperature or float(os.getenv("DEMOC_LLM_TEMPERATURE", "0.7"))
        self.max_batch_size = max_batch_size or int(os.getenv("DEMOC_LLM_BATCH_SIZE", "4"))
        self.timeout = timeout

    # ---------------- internal helpers ----------------

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _single_chat(self, prompt: str) -> str:
        """One call to /v1/chat/completions."""
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a precise assistant that follows instructions exactly.",
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        if requests is not None:
            resp = requests.post(
                url,
                json=payload,
                headers=self._headers(),
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        else:
            request = Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=self._headers(),
                method="POST",
            )
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    charset = response.headers.get_content_charset() or "utf-8"
                    data = json.loads(response.read().decode(charset, errors="replace"))
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"OpenAI chat completion request failed with status {exc.code}: {detail}"
                ) from exc
        choices = data.get("choices", [])
        if not choices:
            return ""
        msg = choices[0].get("message", {})
        return (msg.get("content") or "").strip()

    # ---------------- public API ----------------

    def ask(self, prompt: str) -> str:
        return self._single_chat(prompt)

    def ask_batch(self, prompts: List[str]) -> List[str]:
        """
        Batch interface for the pipeline.

        For OpenAI, we just loop and respect max_batch_size to avoid
        an explosion of parallel HTTP calls.
        """
        outputs: List[str] = []
        for i in range(0, len(prompts), self.max_batch_size):
            batch = prompts[i : i + self.max_batch_size]
            for p in batch:
                outputs.append(self._single_chat(p))
        return outputs
