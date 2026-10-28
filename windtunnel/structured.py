"""Getting JSON out of model replies."""

from __future__ import annotations

import json
import re
from typing import Any

from .llm import Completion, LLMClient

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Any:
    """Parse the first JSON object or array in a reply, tolerating code fences and chatter."""
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    for candidate in candidates:
        for opener, closer in (("{", "}"), ("[", "]")):
            start, end = candidate.find(opener), candidate.rfind(closer)
            if start != -1 and end > start:
                try:
                    return json.loads(candidate[start : end + 1])
                except json.JSONDecodeError:
                    continue
    raise ValueError("no JSON found in model output")


def ask_json(
    client: LLMClient, model: str, system: str, user: str, *, max_tokens: int = 2048
) -> tuple[Any, list[Completion]]:
    """Ask for JSON; retry once with a stricter reminder if the first reply doesn't parse."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    calls = []
    for attempt in range(2):
        completion = client.complete(model, messages, max_tokens=max_tokens)
        calls.append(completion)
        try:
            return extract_json(completion.text), calls
        except ValueError:
            if attempt == 1:
                raise
            messages = [
                *messages,
                {"role": "assistant", "content": completion.text},
                {"role": "user", "content": "That was not valid JSON. Reply with only the JSON, nothing else."},
            ]
    raise AssertionError("unreachable")
