"""L2 Scout: search papers and the web for methods, beyond what an LLM remembers.

Providers: arXiv (newest preprints), OpenAlex (all fields, used for cross-field
analogies), Semantic Scholar (best effort; the free pool is often rate limited) and
Tavily (the web, when TAVILY_API_KEY is set). Results are deduplicated by arXiv ID,
DOI or title, and interleaved across queries so no single approach family dominates.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import httpx

from .understand import ProblemSpec

ARXIV_CATS = "(cat:cs.CL OR cat:cs.AI OR cat:cs.LG OR cat:cs.MA OR cat:cs.IR OR cat:cs.SE)"
_ARXIV_ID = re.compile(r"arxiv(?:\.org/(?:abs|pdf)/|[./:])(\d{4}\.\d{4,5})", re.IGNORECASE)
_STOPWORDS = {"a", "an", "and", "the", "of", "for", "in", "on", "with", "to", "by", "vs", "versus", "using", "via"}


@dataclass
class Source:
    title: str
    url: str
    kind: str  # "paper" or "web"
    year: int | None = None
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    venue: str = ""
    arxiv_id: str | None = None
    doi: str | None = None
    citation_count: int | None = None
    found_via: list[str] = field(default_factory=list)  # providers
    queries: list[str] = field(default_factory=list)
    other_field: bool = False  # came from an analogy query

    @property
    def key(self) -> str:
        if self.arxiv_id:
            return f"arxiv:{self.arxiv_id}"
        if self.doi:
            return f"doi:{self.doi.lower()}"
        return "title:" + re.sub(r"\W+", " ", self.title.lower()).strip()

    @property
    def verified(self) -> bool:
        """True when the record came from a scholarly index and carries an arXiv ID or DOI (not resolved here)."""
        return self.kind == "paper" and bool(self.arxiv_id or self.doi)

    @property
    def recent(self) -> bool:
        return self.year is not None and self.year >= date.today().year - 1

    @property
    def long_tail(self) -> bool:
        return self.citation_count is not None and self.citation_count < 50

    def to_dict(self) -> dict:
        return {**asdict(self), "key": self.key, "verified": self.verified, "recent": self.recent, "long_tail": self.long_tail}

    def merge(self, other: Source) -> None:
        if self.kind == "web" and other.kind == "paper":  # an index record beats a web page for the same title
            self.kind, self.url = other.kind, other.url
            if other.abstract:  # the paper's own abstract, not the web page's snippet, now backs the "verified" card
                self.abstract = other.abstract
        for name in ("abstract", "venue", "arxiv_id", "doi", "year", "citation_count"):
            if not getattr(self, name) and getattr(other, name):
                setattr(self, name, getattr(other, name))
        if not self.authors:
            self.authors = other.authors
        self.found_via = list(dict.fromkeys(self.found_via + other.found_via))
        self.queries = list(dict.fromkeys(self.queries + other.queries))
        self.other_field = self.other_field or other.other_field


class Fetcher:
    """HTTP with a disk cache, per-host pacing and retries on 429/5xx.

    The cache keeps at most `max_cache_files` responses; past that the oldest are deleted, so a
    stream of new search queries cannot fill the disk.
    """

    MIN_INTERVAL = {"export.arxiv.org": 3.0, "api.semanticscholar.org": 1.1}
    PRUNE_EVERY = 100  # cache writes between size checks

    def __init__(
        self,
        cache_dir: str | Path | None = ".windtunnel/cache",
        timeout: float = 30.0,
        retries: int = 3,
        max_cache_files: int | None = None,
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.client = httpx.Client(timeout=timeout, follow_redirects=True, headers={"User-Agent": "windtunnel/0.1"})
        self.retries = retries
        self.max_cache_files = max_cache_files if max_cache_files is not None else int(os.environ.get("WINDTUNNEL_CACHE_MAX_FILES", "5000"))
        self._writes = 0
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _cache_key(method: str, url: str, params: dict | None, json_body: dict | None) -> str:
        return hashlib.sha256(json.dumps([method, url, params, json_body], sort_keys=True).encode()).hexdigest()

    def is_cached(self, method: str, url: str, *, params: dict | None = None, json_body: dict | None = None) -> bool:
        return bool(self.cache_dir and (self.cache_dir / f"{self._cache_key(method, url, params, json_body)}.txt").is_file())

    def prune_cache(self) -> int:
        """Delete the oldest cached responses past max_cache_files. Returns how many were deleted."""
        if not self.cache_dir or not self.cache_dir.is_dir() or self.max_cache_files <= 0:
            return 0
        files = list(self.cache_dir.glob("*.txt"))
        excess = len(files) - self.max_cache_files
        if excess <= 0:
            return 0
        files.sort(key=lambda f: f.stat().st_mtime)
        for f in files[:excess]:
            f.unlink(missing_ok=True)
        return excess

    def _pace(self, host: str) -> None:
        interval = self.MIN_INTERVAL.get(host, 0.0)
        with self._lock:
            wait = self._last.get(host, 0.0) + interval - time.monotonic()
            self._last[host] = time.monotonic() + max(wait, 0.0)
        if wait > 0:
            time.sleep(wait)

    def request(self, method: str, url: str, *, params: dict | None = None, json_body: dict | None = None, headers: dict | None = None) -> str:
        cache_key = self._cache_key(method, url, params, json_body)
        cache_file = self.cache_dir / f"{cache_key}.txt" if self.cache_dir else None
        if cache_file and cache_file.is_file():
            return cache_file.read_text()
        host = httpx.URL(url).host
        for attempt in range(self.retries + 1):
            self._pace(host)
            resp = self.client.request(method, url, params=params, json=json_body, headers=headers)
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt == self.retries:
                    resp.raise_for_status()
                time.sleep(2 ** (attempt + 1))
                continue
            resp.raise_for_status()
            if cache_file:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(resp.text)
                with self._lock:
                    self._writes += 1
                    prune = self._writes % self.PRUNE_EVERY == 0
                if prune:
                    self.prune_cache()
            return resp.text
        raise AssertionError("unreachable")


def _keywords(query: str, limit: int = 6) -> list[str]:
    words = [w for w in re.findall(r"[A-Za-z0-9-]+", query.lower()) if w not in _STOPWORDS and len(w) > 2]
    return words[:limit]


def parse_arxiv_atom(xml_text: str, query: str) -> list[Source]:
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml_text)
    sources = []
    for entry in root.findall("a:entry", ns):
        entry_id = entry.findtext("a:id", default="", namespaces=ns)
        match = _ARXIV_ID.search(entry_id)
        title = " ".join(entry.findtext("a:title", default="", namespaces=ns).split())
        if not match or not title:
            continue
        published = entry.findtext("a:published", default="", namespaces=ns)
        sources.append(
            Source(
                title=title,
                url=f"https://arxiv.org/abs/{match.group(1)}",
                kind="paper",
                year=int(published[:4]) if published[:4].isdigit() else None,
                authors=[a.findtext("a:name", default="", namespaces=ns) for a in entry.findall("a:author", ns)],
                abstract=" ".join(entry.findtext("a:summary", default="", namespaces=ns).split()),
                venue="arXiv",
                arxiv_id=match.group(1),
                found_via=["arxiv"],
                queries=[query],
            )
        )
    return sources


def search_arxiv(query: str, limit: int, fetcher: Fetcher) -> list[Source]:
    terms = " AND ".join(f"all:{w}" for w in _keywords(query))
    params = {"search_query": f"({terms}) AND {ARXIV_CATS}", "max_results": limit, "sortBy": "relevance"}
    return parse_arxiv_atom(fetcher.request("GET", "https://export.arxiv.org/api/query", params=params), query)


def openalex_abstract(inverted: dict | None) -> str:
    if not inverted:
        return ""
    positions = [(i, word) for word, idxs in inverted.items() for i in idxs]
    return " ".join(word for _, word in sorted(positions))


def search_openalex(query: str, limit: int, fetcher: Fetcher) -> list[Source]:
    params = {
        "search": query,
        "per-page": limit,
        "select": "id,display_name,publication_year,doi,cited_by_count,authorships,abstract_inverted_index,primary_location",
    }
    data = json.loads(fetcher.request("GET", "https://api.openalex.org/works", params=params))
    sources = []
    for w in data.get("results", []):
        loc = w.get("primary_location") or {}
        landing = loc.get("landing_page_url") or ""
        doi = (w.get("doi") or "").replace("https://doi.org/", "") or None
        arxiv = _ARXIV_ID.search(doi or "") or _ARXIV_ID.search(landing)
        if not w.get("display_name"):
            continue
        sources.append(
            Source(
                title=w["display_name"],
                url=f"https://arxiv.org/abs/{arxiv.group(1)}" if arxiv else (landing or w.get("doi") or w["id"]),
                kind="paper",
                year=w.get("publication_year"),
                authors=[a["author"]["display_name"] for a in w.get("authorships", [])[:8]],
                abstract=openalex_abstract(w.get("abstract_inverted_index")),
                venue=((loc.get("source") or {}).get("display_name")) or "",
                arxiv_id=arxiv.group(1) if arxiv else None,
                doi=doi,
                citation_count=w.get("cited_by_count"),
                found_via=["openalex"],
                queries=[query],
            )
        )
    return sources


def search_semantic_scholar(query: str, limit: int, fetcher: Fetcher) -> list[Source]:
    params = {"query": query, "limit": limit, "fields": "title,abstract,year,authors,externalIds,citationCount,venue,url"}
    data = json.loads(fetcher.request("GET", "https://api.semanticscholar.org/graph/v1/paper/search", params=params))
    sources = []
    for p in data.get("data", []):
        ids = p.get("externalIds") or {}
        arxiv = ids.get("ArXiv")
        sources.append(
            Source(
                title=p.get("title") or "",
                url=f"https://arxiv.org/abs/{arxiv}" if arxiv else (p.get("url") or ""),
                kind="paper",
                year=p.get("year"),
                authors=[a.get("name", "") for a in (p.get("authors") or [])[:8]],
                abstract=p.get("abstract") or "",
                venue=p.get("venue") or "",
                arxiv_id=arxiv,
                doi=ids.get("DOI"),
                citation_count=p.get("citationCount"),
                found_via=["semantic_scholar"],
                queries=[query],
            )
        )
    return [s for s in sources if s.title]


class TavilyCapReached(RuntimeError):
    pass


class _CallCap:
    """Sliding-window cap on paid calls in this process. Tavily is billed per search and sits outside BudgetGuard."""

    def __init__(self, env_var: str, default: int, window_s: float = 3600.0):
        self.env_var, self.default, self.window_s = env_var, default, window_s
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    @property
    def limit(self) -> int:
        return int(os.environ.get(self.env_var, str(self.default)))

    def take(self) -> bool:
        """Count one call. False (and nothing counted) when the window is already full."""
        now = time.monotonic()
        with self._lock:
            while self._calls and now - self._calls[0] > self.window_s:
                self._calls.popleft()
            if len(self._calls) >= self.limit:
                return False
            self._calls.append(now)
            return True


TAVILY_CAP = _CallCap("WINDTUNNEL_TAVILY_PER_HOUR", 60)


def search_tavily(query: str, limit: int, fetcher: Fetcher) -> list[Source]:
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        return []
    body = {"query": query, "max_results": limit, "search_depth": "basic", "include_answer": False}
    cached = getattr(fetcher, "is_cached", None)
    if not (cached and cached("POST", "https://api.tavily.com/search", json_body=body)) and not TAVILY_CAP.take():
        # Raised, so scout() records "tavily failed for ...: TavilyCapReached" in the run's warnings.
        raise TavilyCapReached(f"Tavily cap of {TAVILY_CAP.limit} paid searches per hour reached")
    data = json.loads(
        fetcher.request("POST", "https://api.tavily.com/search", json_body=body, headers={"Authorization": f"Bearer {key}"})
    )
    return [
        Source(title=r.get("title") or r["url"], url=r["url"], kind="web", abstract=r.get("content") or "", found_via=["tavily"], queries=[query])
        for r in data.get("results", [])
        if r.get("url")
    ]


Provider = Callable[[str, int, Fetcher], list[Source]]
PROVIDERS: dict[str, Provider] = {
    "arxiv": search_arxiv,
    "openalex": search_openalex,
    "semantic_scholar": search_semantic_scholar,
    "tavily": search_tavily,
}
# arXiv search is restricted to CS categories, so it can't serve cross-field analogy queries.
ANALOGY_PROVIDERS = {"openalex", "semantic_scholar", "tavily"}


@dataclass
class ScoutResult:
    sources: list[Source]
    queries: list[str]
    analogy_queries: list[str]
    warnings: list[str]

    def to_dict(self) -> dict:
        return {
            "queries": self.queries,
            "analogy_queries": self.analogy_queries,
            "warnings": self.warnings,
            "sources": [s.to_dict() for s in self.sources],
        }


def _dedup_keys(s: Source) -> list[str]:
    keys = [f"arxiv:{s.arxiv_id}"] if s.arxiv_id else []
    keys += [f"doi:{s.doi.lower()}"] if s.doi else []
    title = re.sub(r"\W+", " ", s.title.lower()).strip()
    return keys + ([f"title:{title}"] if title else [])


def scout(
    spec: ProblemSpec,
    *,
    providers: dict[str, Provider] | None = None,
    per_query: int = 5,
    max_sources: int = 15,
    fetcher: Fetcher | None = None,
) -> ScoutResult:
    providers = providers if providers is not None else PROVIDERS
    fetcher = fetcher or Fetcher()
    jobs = [(q, name, False) for q in spec.search_queries for name in providers]
    jobs += [(q, name, True) for q in spec.analogy_queries for name in providers if name in ANALOGY_PROVIDERS]
    warnings: list[str] = []

    def run(job):
        query, name, other_field = job
        try:
            found = providers[name](query, per_query, fetcher)
        except Exception as exc:  # one provider failing must not sink the search
            return job, [], f"{name} failed for {query!r}: {type(exc).__name__}"
        for s in found:
            s.other_field = other_field
        return job, found, None

    with ThreadPoolExecutor(max_workers=max(1, len(providers))) as pool:
        outcomes = list(pool.map(run, jobs))

    per_query_lists: dict[str, list[Source]] = {}
    for (query, _, _), found, warning in outcomes:
        if warning:
            warnings.append(warning)
        per_query_lists.setdefault(query, []).extend(found)

    # One record per paper: the arXiv preprint and the published DOI version share a title,
    # so any matching arXiv ID, DOI or normalised title merges into the existing entry.
    index: dict[str, Source] = {}
    order: list[Source] = []
    # Round-robin across queries so every approach family gets represented.
    lists = list(per_query_lists.values())
    for i in range(max((len(lst) for lst in lists), default=0)):
        for lst in lists:
            if i < len(lst):
                s = lst[i]
                existing = next((index[k] for k in _dedup_keys(s) if k in index), None)
                if existing is not None:
                    existing.merge(s)
                    s = existing
                else:
                    order.append(s)
                for k in _dedup_keys(s):
                    index.setdefault(k, s)
    sources = order[:max_sources]
    return ScoutResult(sources, list(spec.search_queries), list(spec.analogy_queries), warnings)
