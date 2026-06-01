import pytest

from pd_disaggregation.core.topology import (
    TopologyConfig,
    combine_stage_routes,
    default_topologies,
)


def test_default_topologies_fit_eight_rank_pd_pool() -> None:
    routes = default_topologies()

    assert routes
    assert all(route.required_gpu_count == 8 for route in routes)


def test_combine_stage_routes_uses_independent_stage_layouts() -> None:
    routes = {route.name: route for route in default_topologies()}

    combined = combine_stage_routes(routes["strong_prefill"], routes["aligned_lane"])

    assert combined.prefill_tp == 4
    assert combined.decode_dp == 4
    assert combined.name == "strong_prefill->aligned_lane"


def test_topology_rejects_invalid_parallelism() -> None:
    with pytest.raises(ValueError):
        TopologyConfig("invalid", 0, 1, 1, 1, 1.0, 1.0)
