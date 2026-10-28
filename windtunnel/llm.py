"""LLM clients: Nebius Token Factory for real runs, and an offline mock for plumbing tests."""

from __future__ import annotations

import hashlib
import os
import random
import re
import threading
import time
from dataclasses import dataclass
from typing import Protocol

from .models import CATALOG, get_model

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


@dataclass(frozen=True)
class Completion:
    text: str
    model: str  # catalog key, e.g. "nano"
    input_tokens: int
    output_tokens: int
    latency_s: float

    @property
    def cost_usd(self) -> float:
        return get_model(self.model).cost(self.input_tokens, self.output_tokens)


class LLMClient(Protocol):
    def complete(
        self, model: str, messages: list[dict], *, temperature: float = 0.0, max_tokens: int = 1024
    ) -> Completion: ...


class TokenFactoryClient:
    """OpenAI-compatible client for Nebius Token Factory, with backoff on rate limits."""

    BASE_URL = "https://api.tokenfactory.nebius.com/v1/"

    def __init__(self, api_key: str | None = None, *, thinking: bool = False, max_retries: int = 6, timeout: float = 120.0):
        from openai import OpenAI

        key = api_key or os.environ.get("NEBIUS_API_KEY")
        if not key:
            raise RuntimeError("Set NEBIUS_API_KEY, or run with --mock.")
        self._client = OpenAI(base_url=self.BASE_URL, api_key=key, timeout=timeout, max_retries=0)
        self.thinking = thinking
        self.max_retries = max_retries

    def complete(
        self, model: str, messages: list[dict], *, temperature: float = 0.0, max_tokens: int = 1024
    ) -> Completion:
        import openai

        retryable = (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError)
        for attempt in range(self.max_retries + 1):
            start = time.perf_counter()
            try:
                resp = self._client.chat.completions.create(
                    model=get_model(model).id,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body={"chat_template_kwargs": {"enable_thinking": self.thinking}},
                )
            except retryable:
                if attempt == self.max_retries:
                    raise
                time.sleep(min(60.0, 2**attempt) * (0.5 + random.random()))
                continue
            text = _THINK_RE.sub("", resp.choices[0].message.content or "").strip()
            usage = resp.usage
            return Completion(
                text=text,
                model=model,
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
                latency_s=time.perf_counter() - start,
            )
        raise AssertionError("unreachable")

    def list_model_ids(self) -> list[str]:
        return sorted(m.id for m in self._client.models.list())


class MockClient:
    """Offline stand-in for Token Factory.

    Answers are simulated from the task's own labels: each model tier is right with a
    fixed probability. Mock races test the plumbing, never the models, so mock numbers
    must not be quoted as results.
    """

    DEFAULT_SKILL = {"nano": 0.70, "lightning": 0.72, "super": 0.82, "ultra": 0.90}
    DEFAULT_LATENCY = {"nano": 0.4, "lightning": 0.3, "super": 0.9, "ultra": 2.5}

    def __init__(self, oracle: dict[str, str], wrong_answers: list[str], *, skill: dict[str, float] | None = None, seed: int = 0):
        self.oracle = oracle
        self.wrong_answers = wrong_answers or ["unknown"]
        self.skill = {**self.DEFAULT_SKILL, **(skill or {})}
        self.seed = seed
        self._seen: dict[tuple, int] = {}
        self._lock = threading.Lock()

    @classmethod
    def for_task(cls, task, **kw) -> MockClient:
        oracle = {e.input: e.expected for e in task.examples}
        wrong = task.labels or sorted({e.expected for e in task.examples})
        return cls(oracle, wrong, **kw)

    def _draw(self, *parts) -> float:
        digest = hashlib.sha256("|".join(map(str, (self.seed, *parts))).encode()).digest()
        return int.from_bytes(digest[:8], "big") / 2**64

    def complete(
        self, model: str, messages: list[dict], *, temperature: float = 0.0, max_tokens: int = 1024
    ) -> Completion:
        from .designs import PLAN_MARKER, SPLIT_MARKER, WORKER_MARKER
        from .scoring import normalize

        if model not in CATALOG:
            raise KeyError(model)
        prompt = "\n".join(m["content"] for m in messages)
        # Same prompt at temperature 0 gives the same answer; sampled calls vary per occurrence.
        occurrence = 0
        if temperature > 0:
            with self._lock:
                key = (model, prompt, temperature)
                occurrence = self._seen.get(key, 0)
                self._seen[key] = occurrence + 1
        u = self._draw(model, temperature, occurrence, prompt)

        if "PICK:" in prompt:
            ids = re.findall(r"^- ([\w.-]+):", prompt, re.MULTILINE)
            text = f"(mock advisor)\nPICK: {ids[0] if ids else 'none'}"
        elif PLAN_MARKER in prompt:
            text = "1. Read the input.\n2. Match it to the closest option.\n3. Answer."
        elif SPLIT_MARKER in prompt:
            text = "What is being asked?\nWhich option fits best?"
        elif WORKER_MARKER in prompt:
            text = "(mock finding)"
        else:
            matches = [inp for inp in self.oracle if inp in prompt]
            expected = self.oracle[max(matches, key=len)] if matches else None
            if expected is not None and u < self.skill[model]:
                answer = expected
            else:
                wrong = [w for w in self.wrong_answers if expected is None or normalize(w) != normalize(expected)]
                answer = wrong[int(self._draw("wrong", u) * len(wrong))] if wrong else "unknown"
            text = f"(mock reasoning)\nANSWER: {answer}"

        latency = self.DEFAULT_LATENCY.get(model, 1.0) * (0.8 + 0.4 * self._draw("latency", u))
        return Completion(text, model, max(1, len(prompt) // 4), max(1, len(text) // 4), latency)
