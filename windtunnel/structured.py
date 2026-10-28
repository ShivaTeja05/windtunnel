"""Getting JSON out of model replies."""

from __future__ import annotations

import json
import re
from typing import Any

from .llm import Completion, LLMClient

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_DECODER = json.JSONDecoder()


class JSONReplyError(ValueError):
    """No JSON in the reply even after a retry. `calls` holds the paid completions, so callers can count them."""

    def __init__(self, message: str, calls: list[Completion]):
        super().__init__(message)
        self.calls = calls


def extract_json(text: str, expect: type | tuple[type, ...] | None = None) -> Any:
    """Parse the first JSON object or array in a reply, tolerating code fences and chatter.

    Each '{' or '[' is tried in order with a streaming decode, so prose after the JSON
    (even prose containing braces) cannot break it. With `expect`, values of other types
    are skipped, so a citation marker such as "[1]" is never taken for the reply.
    """
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    for candidate in candidates:
        for i, ch in enumerate(candidate):
            if ch not in "{[":
                continue
            try:
                value, _ = _DECODER.raw_decode(candidate, i)
            except json.JSONDecodeError:
                continue
            if expect is None or isinstance(value, expect):
                return value
    raise ValueError("no JSON found in model output")


def ask_json(
    client: LLMClient,
    model: str,
    system: str,
    user: str,
    *,
    max_tokens: int = 2048,
    expect: type | tuple[type, ...] = dict,
) -> tuple[Any, list[Completion]]:
    """Ask for JSON; retry once with a stricter reminder if the first reply doesn't parse.

    Raises JSONReplyError (a ValueError) carrying both paid calls when neither reply parses.
    """
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    calls = []
    for attempt in range(2):
        completion = client.complete(model, messages, max_tokens=max_tokens)
        calls.append(completion)
        try:
            return extract_json(completion.text, expect), calls
        except ValueError as exc:
            if attempt == 1:
                raise JSONReplyError(str(exc), calls) from None
            messages = [
                *messages,
                {"role": "assistant", "content": completion.text},
                {"role": "user", "content": "That was not valid JSON. Reply with only the JSON, nothing else."},
            ]
    raise AssertionError("unreachable")
