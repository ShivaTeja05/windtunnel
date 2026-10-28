"""Nemotron models on Nebius Token Factory, with list prices for cost accounting."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    key: str
    id: str
    usd_per_m_input: float
    usd_per_m_output: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.usd_per_m_input + output_tokens * self.usd_per_m_output) / 1_000_000


# Token Factory public catalog as fetched 2026-10-20. Run `windtunnel models --check` to verify the IDs.
CATALOG: dict[str, ModelInfo] = {
    m.key: m
    for m in [
        ModelInfo("nano", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B", 0.06, 0.24),
        ModelInfo("lightning", "nvidia/Nemotron-3_5-Lightning", 0.06, 0.24),
        ModelInfo("super", "nvidia/nemotron-3-super-120b-a12b", 0.30, 0.90),
        ModelInfo("ultra", "nvidia/Nemotron-3-Ultra-550b-a55b", 1.00, 3.00),
    ]
}


def get_model(key: str) -> ModelInfo:
    try:
        return CATALOG[key]
    except KeyError:
        raise KeyError(f"unknown model {key!r}; known: {', '.join(CATALOG)}") from None
