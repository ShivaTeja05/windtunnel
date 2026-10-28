"""Scorers compare a design's final answer with the expected answer and return 0.0-1.0."""

import math
import re
from collections.abc import Callable, Iterable

# Keep "+", "#" and "-" so labels such as "C++", "C#" and "-1" stay distinct.
_PUNCT = re.compile(r"[^\w\s.+#-]")
# A hyphen inside a word reads like a space ("account-access"); a dash not before a digit is punctuation.
_DASH = re.compile(r"(?<=\w)-(?=\w)|-(?!\d)")
_NUMBER = re.compile(r"-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
_NUMBER_ONLY = re.compile(r"^" + _NUMBER.pattern + r"$")
_NUMBER_NOISE = re.compile(r"[,\s$€£%]")


def normalize(text: str) -> str:
    text = text.strip().strip("`'\"").casefold().replace("_", " ")
    text = _PUNCT.sub("", _DASH.sub(" ", text))
    return " ".join(text.split()).rstrip(".")


def exact(pred: str, expected: str) -> float:
    target = normalize(expected)
    return float(bool(target) and normalize(pred) == target)


def contains(pred: str, expected: str) -> float:
    target = normalize(expected)
    return float(bool(target) and target in normalize(pred))  # an empty label would match everything


def parse_expected_number(expected: str) -> float:
    """The number a numeric label states ("1,234", "$12", "12%", ".5", "1e3"); ValueError if it states none."""
    cleaned = _NUMBER_NOISE.sub("", expected)
    if not _NUMBER_ONLY.match(cleaned):
        raise ValueError(f"expected value {expected!r} is not a number")
    return float(cleaned)


def numeric(pred: str, expected: str) -> float:
    match = _NUMBER.search(pred.replace(",", ""))
    if not match:
        return 0.0
    return float(math.isclose(float(match.group()), parse_expected_number(expected), rel_tol=1e-6, abs_tol=1e-9))


SCORERS: dict[str, Callable[[str, str], float]] = {"exact": exact, "contains": contains, "numeric": numeric}


def get_scorer(name: str) -> Callable[[str, str], float]:
    try:
        return SCORERS[name]
    except KeyError:
        raise KeyError(f"unknown scorer {name!r}; known: {', '.join(SCORERS)}") from None


def validate_expected(scorer: str, labels: Iterable[tuple[str, str]]) -> None:
    """Check (example id, expected) pairs before any money is spent; ValueError names the first bad one."""
    get_scorer(scorer)
    for example_id, expected in labels:
        if not normalize(str(expected)):
            raise ValueError(f"example {example_id!r} has an empty expected answer")
        if scorer == "numeric":
            try:
                parse_expected_number(str(expected))
            except ValueError as exc:
                raise ValueError(f"example {example_id!r}: {exc} (the numeric scorer needs a number)") from None
