import pytest

from windtunnel.designs import Design, default_designs
from windtunnel.graph import graph_for, to_mermaid


@pytest.mark.parametrize("design", default_designs(), ids=lambda d: d.id)
def test_every_default_design_has_a_connected_graph(design):
    g = graph_for(design)
    ids = {n.id for n in g.nodes}
    assert {"in", "out"} <= ids
    assert all(e.src in ids and e.dst in ids for e in g.edges)
    reachable, frontier = {"in"}, ["in"]
    while frontier:
        cur = frontier.pop()
        for e in g.edges:
            if e.src == cur and e.dst not in reachable:
                reachable.add(e.dst)
                frontier.append(e.dst)
    assert reachable == ids


def test_graph_records_which_models_run():
    g = graph_for(Design("crew", "orchestrator_workers", "ultra", {"worker_model": "nano", "max_subtasks": 2}))
    assert g.models() == ["nano", "ultra"]
    assert sum(1 for n in g.nodes if n.model == "nano") == 2


def test_mermaid_output_styles_nodes_by_tier():
    text = to_mermaid(graph_for(Design("v", "self_consistency", "nano", {"n": 3})))
    assert text.startswith("flowchart LR")
    assert 'samples[["3 independent answers · nano"]]' in text
    assert "class samples nano" in text and "class vote io" in text
