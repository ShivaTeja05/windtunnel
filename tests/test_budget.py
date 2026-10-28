import pytest

from windtunnel.budget import BudgetExceeded, BudgetGuard
from windtunnel.llm import Completion

MSG = [{"role": "user", "content": "x" * 3000}]  # ~1000 tokens, matching what the fake reports


class FixedCostClient:
    calls = 0

    def complete(self, model, messages, *, temperature=0.0, max_tokens=1024):
        self.calls += 1
        return Completion("ANSWER: x", model, 1000, 100, 0.1)  # ultra: $0.0013


def test_cap_blocks_calls_before_they_reach_the_api(tmp_path):
    inner = FixedCostClient()
    guard = BudgetGuard(inner, limit_usd=0.005, ledger=tmp_path / "spend.json")
    with pytest.raises(BudgetExceeded):
        for _ in range(100):
            guard.complete("ultra", MSG, max_tokens=100)
    assert guard.spent_usd <= 0.005
    assert inner.calls < 100


def test_spend_persists_across_runs(tmp_path):
    ledger = tmp_path / "spend.json"
    BudgetGuard(FixedCostClient(), limit_usd=1, ledger=ledger).complete("ultra", MSG, max_tokens=100)
    again = BudgetGuard(FixedCostClient(), limit_usd=1, ledger=ledger)
    assert again.spent_usd == pytest.approx(0.0013)


def test_concurrent_guards_share_one_ledger(tmp_path):
    # Two guards on the same ledger (say a web job and an MCP call) must see each other's spend.
    ledger = tmp_path / "spend.json"
    first = BudgetGuard(FixedCostClient(), limit_usd=0.004, ledger=ledger)
    second = BudgetGuard(FixedCostClient(), limit_usd=0.004, ledger=ledger)
    first.complete("ultra", MSG, max_tokens=100)
    second.complete("ultra", MSG, max_tokens=100)
    assert first.spent_usd == second.spent_usd == pytest.approx(0.0026)  # added, not overwritten
    with pytest.raises(BudgetExceeded):
        first.complete("ultra", MSG, max_tokens=100)


def test_in_flight_reservations_count_against_other_guards(tmp_path):
    ledger = tmp_path / "spend.json"
    other = BudgetGuard(FixedCostClient(), limit_usd=0.003, ledger=ledger)

    class CallsOtherGuard(FixedCostClient):
        def complete(self, model, messages, **kw):
            with pytest.raises(BudgetExceeded):  # the outer call's reservation is still held
                other.complete("ultra", MSG, max_tokens=100)
            return super().complete(model, messages, **kw)

    BudgetGuard(CallsOtherGuard(), limit_usd=0.003, ledger=ledger).complete("ultra", MSG, max_tokens=100)
    assert other.spent_usd == pytest.approx(0.0013)


def test_threads_never_push_spend_past_the_cap(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    guard = BudgetGuard(FixedCostClient(), limit_usd=0.02, ledger=tmp_path / "spend.json")

    def call(_):
        try:
            guard.complete("ultra", MSG, max_tokens=100)
        except BudgetExceeded:
            pass

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(call, range(40)))
    assert 0.015 < guard.spent_usd <= 0.02


def test_worst_case_covers_non_latin_text():
    from windtunnel.budget import worst_case_prompt_tokens

    chinese = "退款没有到账" * 2000  # 12,000 characters, about one token each
    assert worst_case_prompt_tokens([{"role": "user", "content": chinese}]) >= len(chinese)
