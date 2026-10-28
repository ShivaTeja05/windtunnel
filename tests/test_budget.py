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
