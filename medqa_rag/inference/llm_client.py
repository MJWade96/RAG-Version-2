from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI


@dataclass(slots=True)
class LLMResponse:
    text: str
    raw: dict[str, Any] = field(default_factory=dict)


class TianyiOpenAILLMClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        enable_thinking: bool = True,
        timeout: int = 60,
    ) -> None:
        if not base_url:
            raise ValueError("base_url must be configured.")
        if not api_key:
            raise ValueError("api_key must be configured.")
        if not model:
            raise ValueError("model must be configured.")

        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.enable_thinking = enable_thinking
        self.timeout = timeout
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout)

    def generate(self, prompt: str, temperature: float = 0.0) -> LLMResponse:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            stream=False,
            extra_body={"enable_thinking": self.enable_thinking},
        )
        content = response.choices[0].message.content or ""
        return LLMResponse(text=content, raw=response.model_dump())
