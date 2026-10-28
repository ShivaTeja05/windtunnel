"""Architecture graphs: the single source of truth for what a design does.

Each design compiles to a graph of steps (nodes) and data flow (edges). The same graph
drives the diagram users see, so a diagram can never drift from what was tested.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .designs import Design

# Node kinds and their Mermaid shapes.
SHAPES = {
    "input": ("([", "])"),
    "llm": ("[", "]"),
    "parallel": ("[[", "]]"),
    "vote": ("[/", "\\]"),
    "check": ("{{", "}}"),
    "output": ("([", "])"),
}

# Model tiers get one colour each so every diagram shows which model does what.
TIER_STYLES = {
    "nano": "fill:#E8F5E9,stroke:#2E7D32,color:#1B5E20",
    "lightning": "fill:#E0F7FA,stroke:#00838F,color:#006064",
    "super": "fill:#E3F2FD,stroke:#1565C0,color:#0D47A1",
    "ultra": "fill:#F3E5F5,stroke:#6A1B9A,color:#4A148C",
    "io": "fill:#F5F5F5,stroke:#616161,color:#212121",
}


@dataclass(frozen=True)
class Node:
    id: str
    kind: str
    label: str
    model: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in SHAPES:
            raise ValueError(f"node {self.id!r}: unknown kind {self.kind!r}")


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    label: str = ""
    dashed: bool = False  # a conditional branch that only some inputs take


@dataclass
class DesignGraph:
    design_id: str
    title: str
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "design_id": self.design_id,
            "title": self.title,
            "nodes": [n.__dict__ for n in self.nodes],
            "edges": [e.__dict__ for e in self.edges],
        }

    def models(self) -> list[str]:
        return sorted({n.model for n in self.nodes if n.model})


def _chain(design: Design, title: str, steps: list[Node]) -> DesignGraph:
    nodes = [Node("in", "input", "Input"), *steps, Node("out", "output", "Answer")]
    edges = [Edge(a.id, b.id) for a, b in zip(nodes, nodes[1:])]
    return DesignGraph(design.id, title, nodes, edges)


def graph_for(design: Design) -> DesignGraph:
    m = design.model
    p = design.params
    if design.family == "single_call":
        return _chain(design, "Single call", [Node("answer", "llm", f"Answer · {m}", m)])
    if design.family == "self_check":
        return _chain(
            design,
            "Answer + self-check",
            [Node("answer", "llm", f"Draft answer · {m}", m), Node("check", "check", f"Check & correct · {m}", m)],
        )
    if design.family == "plan_execute":
        return _chain(
            design,
            "Plan, then execute",
            [Node("plan", "llm", f"Write plan · {m}", m), Node("exec", "llm", f"Follow plan · {m}", m)],
        )
    if design.family == "self_consistency":
        n = int(p.get("n", 5))
        g = DesignGraph(design.id, f"Self-consistency ×{n}")
        g.nodes += [
            Node("in", "input", "Input"),
            Node("samples", "parallel", f"{n} independent answers · {m}", m),
            Node("vote", "vote", "Majority vote"),
            Node("out", "output", "Answer"),
        ]
        g.edges += [Edge("in", "samples"), Edge("samples", "vote", f"{n} answers"), Edge("vote", "out")]
        return g
    if design.family == "orchestrator_workers":
        k = int(p.get("max_subtasks", 3))
        wm = p.get("worker_model", m)
        g = DesignGraph(design.id, "Orchestrator + workers")
        g.nodes += [Node("in", "input", "Input"), Node("split", "llm", f"Split into ≤{k} sub-questions · {m}", m)]
        g.edges += [Edge("in", "split")]
        for i in range(1, k + 1):
            g.nodes.append(Node(f"w{i}", "llm", f"Worker {i} · {wm}", wm))
            g.edges += [Edge("split", f"w{i}"), Edge(f"w{i}", "synth")]
        g.nodes += [Node("synth", "llm", f"Combine findings · {m}", m), Node("out", "output", "Answer")]
        g.edges += [Edge("synth", "out")]
        return g
    if design.family == "cascade":
        cheap = p.get("cheap_model", "nano")
        strong = p.get("escalate_model", "super")
        n = int(p.get("samples", 2))
        g = DesignGraph(design.id, f"Cascade {cheap} → {strong}")
        g.nodes += [
            Node("in", "input", "Input"),
            Node("samples", "parallel", f"{n} cheap answers · {cheap}", cheap),
            Node("agree", "check", "Answers agree?"),
            Node("escalate", "llm", f"Escalate · {strong}", strong),
            Node("out", "output", "Answer"),
        ]
        g.edges += [
            Edge("in", "samples"),
            Edge("samples", "agree", f"{n} answers"),
            Edge("agree", "out", "agree"),
            Edge("agree", "escalate", "disagree", dashed=True),
            Edge("escalate", "out"),
        ]
        return g
    raise ValueError(f"no graph for family {design.family!r}")


def _label(text: str) -> str:
    return '"' + re.sub(r'["<>]', "", text) + '"'


def to_mermaid(graph: DesignGraph, direction: str = "LR") -> str:
    """Render a graph as a Mermaid flowchart, coloured by model tier."""
    lines = [f"flowchart {direction}"]
    for n in graph.nodes:
        open_, close = SHAPES[n.kind]
        lines.append(f"  {n.id}{open_}{_label(n.label)}{close}")
    for e in graph.edges:
        link = "-.->" if e.dashed else "-->"
        arrow = f" {link}|{_label(e.label)}| " if e.label else f" {link} "
        lines.append(f"  {e.src}{arrow}{e.dst}")
    for tier, style in TIER_STYLES.items():
        lines.append(f"  classDef {tier} {style}")
    for n in graph.nodes:
        lines.append(f"  class {n.id} {n.model or 'io'}")
    return "\n".join(lines)
