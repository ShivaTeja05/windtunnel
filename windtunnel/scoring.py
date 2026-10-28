"""Scorers compare a design's final answer with the expected answer and return 0.0-1.0."""

import math
import re
from collections.abc import Callable

_PUNCT = re.compile(r"[^\w\s.]")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def normalize(text: str) -> str:
    text = text.strip().strip("`'\"").casefold().replace("_", " ").replace("-", " ")
    text = _PUNCT.sub("", text)
    return " ".join(text.split()).rstrip(".")


def exact(pred: str, expected: str) -> float:
    return float(normalize(pred) == normalize(expected))


def contains(pred: str, expected: str) -> float:
    return float(normalize(expected) in normalize(pred))


def numeric(pred: str, expected: str) -> float:
    match = _NUMBER.search(pred.replace(",", ""))
    if not match:
        return 0.0
    return float(math.isclose(float(match.group()), float(expected), rel_tol=1e-6, abs_tol=1e-9))


SCORERS: dict[str, Callable[[str, str], float]] = {"exact": exact, "contains": contains, "numeric": numeric}


def get_scorer(name: str) -> Callable[[str, str], float]:
    try:
        return SCORERS[name]
    except KeyError:
        raise KeyError(f"unknown scorer {name!r}; known: {', '.join(SCORERS)}") from None
