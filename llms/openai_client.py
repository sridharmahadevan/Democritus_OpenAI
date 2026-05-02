# llms/openai_client.py

import json
import os
from pathlib import Path
from typing import List, Optional, Union
from urllib.error import HTTPError
from urllib.request import Request, urlopen

try:
    import requests
except ModuleNotFoundError:  # pragma: no cover - exercised in CLIFF integration envs
    requests = None

try:
    from CLIFF_CatAgi.functorflow_v3.llm_usage import (
        append_llm_usage_row,
        enforce_llm_token_budget,
        extract_openai_usage,
        llm_usage_metadata_from_env,
        llm_usage_path_from_env,
        raise_if_over_llm_token_budget,
    )
except ImportError:  # pragma: no cover - standalone Democritus_OpenAI usage without CLIFF workspace
    append_llm_usage_row = None
    enforce_llm_token_budget = None
    extract_openai_usage = None
    raise_if_over_llm_token_budget = None

    def llm_usage_metadata_from_env() -> dict[str, object]:
        return {}

    def llm_usage_path_from_env() -> Optional[Path]:
        return None


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
        usage_log_path: Optional[Union[Path, str]] = None,
        usage_metadata: Optional[dict[str, object]] = None,
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
        self.usage_log_path = (
            Path(usage_log_path).expanduser().resolve()
            if usage_log_path is not None
            else llm_usage_path_from_env()
        )
        self.usage_metadata = {**llm_usage_metadata_from_env(), **dict(usage_metadata or {})}

    # ---------------- internal helpers ----------------

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _extract_error_detail(response) -> str:
        try:
            payload = response.json()
        except Exception:
            payload = None
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                message = error.get("message")
                if message:
                    return str(message)
            message = payload.get("message")
            if message:
                return str(message)
        detail = getattr(response, "text", "")
        if detail:
            return str(detail)
        return "No response body returned."

    def _record_usage(self, *, prompt: str, response_text: str, payload: dict[str, object]) -> None:
        if append_llm_usage_row is None or extract_openai_usage is None:
            return
        usage = extract_openai_usage(payload)
        append_llm_usage_row(
            self.usage_log_path,
            usage=usage,
            metadata={
                **self.usage_metadata,
                "client": "Democritus_OpenAI.OpenAIChatClient",
                "provider": "openai_compatible_chat",
                "request_kind": "chat.completions",
                "model": str(usage.get("model") or self.model),
                "prompt_chars": len(prompt),
                "response_chars": len(response_text),
            },
        )

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
        if enforce_llm_token_budget is not None:
            budget_status = enforce_llm_token_budget(
                self.usage_log_path,
                requested_completion_tokens=self.max_tokens,
                prompt_chars=len(prompt),
            )
            allowed_completion_tokens = int(budget_status.get("allowed_completion_tokens") or 0)
            if allowed_completion_tokens > 0:
                payload["max_tokens"] = min(int(payload["max_tokens"]), allowed_completion_tokens)

        if requests is not None:
            resp = requests.post(
                url,
                json=payload,
                headers=self._headers(),
                timeout=self.timeout,
            )
            status_code = getattr(resp, "status_code", 200)
            if status_code >= 400:
                detail = self._extract_error_detail(resp)
                raise RuntimeError(
                    "OpenAI chat completion request failed with "
                    f"status {status_code} for model {self.model!r} "
                    f"(prompt chars={len(prompt)}): {detail}"
                )
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
        response_text = (msg.get("content") or "").strip()
        self._record_usage(prompt=prompt, response_text=response_text, payload=data)
        if raise_if_over_llm_token_budget is not None:
            raise_if_over_llm_token_budget(self.usage_log_path)
        return response_text

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
