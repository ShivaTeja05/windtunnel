import pytest

from windtunnel.extract import card_with_model, classify_family, extract_cards
from windtunnel.llm import Completion
from windtunnel.research import research
from windtunnel.scout import Source, openalex_abstract, parse_arxiv_atom, scout
from windtunnel.structured import extract_json
from windtunnel.understand import understand, understand_offline

ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2203.11171v4</id>
    <published>2022-03-21T17:59:59Z</published>
    <title>Self-Consistency Improves Chain of Thought
      Reasoning in Language Models</title>
    <summary>We propose self-consistency, a decoding strategy. It samples diverse paths.</summary>
    <author><name>Xuezhi Wang</name></author>
    <author><name>Jason Wei</name></author>
  </entry>
</feed>"""


class ScriptedClient:
    def __init__(self, replies):
        self.replies = list(replies)

    def complete(self, model, messages, *, temperature=0.0, max_tokens=1024):
        return Completion(self.replies.pop(0), model, 100, 50, 0.1)


def test_extract_json_handles_fences_and_chatter():
    assert extract_json('Sure!\n```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]}
    assert extract_json('noise [{"x": 1}, {"x": 2}] tail') == [{"x": 1}, {"x": 2}]


def test_offline_understanding_uses_whole_words():
    spec = understand_offline("Route support tickets into billing or bug queues. Must cost under $2 per 1000.")
    assert "classification" in spec.problem_types
    assert "code" not in spec.problem_types
    assert any("$2" in c for c in spec.constraints)


def test_model_understanding_parses_json():
    reply = '{"summary": "Route tickets", "problem_types": ["classification", "bogus"], "search_queries": ["q1", "q2"], "analogy_queries": ["a1"]}'
    spec = understand("route tickets", ScriptedClient([reply]))
    assert spec.problem_types == ["classification"]
    assert spec.search_queries == ["q1", "q2"] and spec.produced_by == "ultra"


def test_arxiv_atom_parsing():
    [s] = parse_arxiv_atom(ATOM, "q")
    assert s.arxiv_id == "2203.11171" and s.year == 2022
    assert s.title.startswith("Self-Consistency Improves Chain of Thought Reasoning")
    assert s.verified and s.authors == ["Xuezhi Wang", "Jason Wei"]


def test_openalex_abstract_reconstruction():
    assert openalex_abstract({"world": [1], "hello": [0]}) == "hello world"


def test_family_keywords_need_word_starts():
    assert classify_family("improves test coverage of programs") == "other"
    assert classify_family("a RAG pipeline with reranking") == "retrieval"
    assert classify_family("majority voting over sampled answers") == "self_consistency"


def fake_providers():
    def arxiv(query, limit, fetcher):
        return [Source("Self-Consistency", "https://arxiv.org/abs/2203.11171", "paper", 2022, arxiv_id="2203.11171",
                       abstract="Majority voting over sampled reasoning paths.", found_via=["arxiv"], queries=[query])]

    def openalex(query, limit, fetcher):
        return [
            Source("Self-Consistency (dup)", "https://doi.org/x", "paper", 2022, arxiv_id="2203.11171", citation_count=800,
                   found_via=["openalex"], queries=[query]),
            Source(f"Fault tolerance via voting for {query}", "https://doi.org/y", "paper", 1995, doi=f"10.1/{query}",
                   citation_count=12, found_via=["openalex"], queries=[query]),
        ]

    return {"arxiv": arxiv, "openalex": openalex}


def test_scout_dedupes_and_marks_other_field():
    spec = understand_offline("classify tickets")
    spec.search_queries, spec.analogy_queries = ["q1"], ["a1"]
    result = scout(spec, providers=fake_providers(), fetcher=object())
    keys = [s.key for s in result.sources]
    assert keys.count("arxiv:2203.11171") == 1
    merged = next(s for s in result.sources if s.key == "arxiv:2203.11171")
    assert merged.citation_count == 800 and set(merged.found_via) == {"arxiv", "openalex"}
    analogy = next(s for s in result.sources if "a1" in s.title)
    assert analogy.other_field


def test_card_citation_comes_from_source_not_model():
    src = Source("Real Paper", "https://arxiv.org/abs/2203.11171", "paper", 2022, arxiv_id="2203.11171", abstract="x")
    reply = '{"method_name": "SC", "family": "self_consistency", "mechanism": "vote", "relevance": 9}'
    card = card_with_model(src, understand_offline("classify"), ScriptedClient([reply]))
    assert card.citation["url"] == "https://arxiv.org/abs/2203.11171" and card.verified
    assert card.relevance == 3 and card.family == "self_consistency"


def test_failed_extraction_falls_back_offline():
    src = Source("Voting paper", "https://arxiv.org/abs/1", "paper", 2020, arxiv_id="1111.11111", abstract="majority voting")
    cards, warnings = extract_cards([src], understand_offline("classify"), ScriptedClient(["not json", "still not"]))
    assert cards[0].extracted_by == "offline-heuristic" and warnings


def test_research_offline_end_to_end():
    result = research("classify support tickets", providers=fake_providers(), fetcher=object())
    assert result.mode == "offline" and result.cards
    assert "self_consistency" in result.methods_map()


def test_extract_json_ignores_braces_in_trailing_prose():
    assert extract_json('{"summary": "x"}\nNote: I used {braces} here.') == {"summary": "x"}


def test_understand_rejects_or_cleans_wrong_shaped_plans():
    spec = understand("classify tickets", ScriptedClient(['{"summary": "x", "requirements": 3, "search_queries": {"a": 1}}']))
    assert spec.requirements == [] and spec.search_queries  # fell back to the offline queries
    with pytest.raises(ValueError):
        understand("classify tickets", ScriptedClient(['["not", "a", "plan"]']))


def test_scout_merges_arxiv_and_doi_records_of_one_paper():
    def arxiv(query, limit, fetcher):
        return [Source("Self-Consistency Improves Chain of Thought", "https://arxiv.org/abs/2203.11171", "paper", 2022,
                       arxiv_id="2203.11171", found_via=["arxiv"])]

    def openalex(query, limit, fetcher):
        return [Source("Self-consistency improves chain of thought.", "https://doi.org/10.18653/v1/x", "paper", 2023,
                       doi="10.18653/v1/x", citation_count=900, found_via=["openalex"])]

    spec = understand_offline("classify tickets")
    spec.search_queries, spec.analogy_queries = ["q1"], []
    result = scout(spec, providers={"arxiv": arxiv, "openalex": openalex}, fetcher=object())
    assert len(result.sources) == 1
    merged = result.sources[0]
    assert merged.doi == "10.18653/v1/x" and merged.citation_count == 900 and set(merged.found_via) == {"arxiv", "openalex"}


def test_card_without_abstract_is_not_written_from_memory():
    src = Source("Chain-of-Verification", "https://doi.org/10.1/x", "paper", 2023, doi="10.1/x", abstract="")
    client = ScriptedClient([])  # any call would fail: there must be none
    card = card_with_model(src, understand_offline("classify"), client)
    assert card.extracted_by == "offline-heuristic" and "no-abstract" in card.flags and card.reported_result == ""


def test_reported_result_must_appear_in_the_abstract():
    src = Source("Paper", "https://arxiv.org/abs/1", "paper", 2023, arxiv_id="2301.00001", abstract="Voting improves F1 by 4.5 points.")
    invented = '{"method_name": "V", "family": "self_consistency", "reported_result": "+23% F1 on MultiSpanQA"}'
    card = card_with_model(src, understand_offline("classify"), ScriptedClient([invented]))
    assert card.reported_result == "" and "unverified-result" in card.flags
    stated = '{"method_name": "V", "family": "self_consistency", "reported_result": "F1 up 4.5 points"}'
    card = card_with_model(src, understand_offline("classify"), ScriptedClient([stated]))
    assert card.reported_result == "F1 up 4.5 points" and "unverified-result" not in card.flags


def test_failed_extraction_still_counts_its_spend():
    src = Source("Voting paper", "https://arxiv.org/abs/1", "paper", 2020, arxiv_id="1111.11111", abstract="majority voting")
    cards, _ = extract_cards([src], understand_offline("classify"), ScriptedClient(["not json", "still not"]))
    assert cards[0].extracted_by == "offline-heuristic"
    assert cards[0].cost_usd == 2 * Completion("", "super", 100, 50, 0.1).cost_usd
