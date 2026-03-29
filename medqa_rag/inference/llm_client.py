from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI


class RateLimiter:
    """Thread-safe rate limiter using token bucket algorithm."""

    def __init__(self, rate: float = 10.0, burst: int = 10):
        self.rate = rate  # requests per second
        self.burst = burst
        self.tokens = burst
        self.last_update = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self):
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
            self.last_update = now

            if self.tokens >= 1:
                self.tokens -= 1
                return

            wait_time = (1 - self.tokens) / self.rate
            time.sleep(wait_time)
            self.tokens = 0
            self.last_update = time.monotonic()


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
        rate_limit: float = 10.0,
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
        self.rate_limiter = RateLimiter(rate=rate_limit, burst=int(rate_limit))

    def generate(self, prompt: str, temperature: float = 0.0) -> LLMResponse:
        self.rate_limiter.acquire()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            stream=False,
            extra_body={"enable_thinking": self.enable_thinking},
        )
        content = response.choices[0].message.content or ""
        return LLMResponse(text=content, raw=response.model_dump())
