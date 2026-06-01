from pd_disaggregation.core import (
    KVMetadata,
    Request,
    TransferEngine,
    TransferPlanner,
    combine_stage_routes,
    default_topologies,
)


def test_transfer_plan_maps_ranks_and_bytes() -> None:
    routes = {route.name: route for route in default_topologies()}
    route = combine_stage_routes(routes["strong_prefill"], routes["aligned_lane"])
    request = Request("handoff", 4096, 64, 0.0, 500.0, 2000.0)
    kv_meta = KVMetadata(32, 32, 128, request.prompt_len, 2, route.prefill_tp)

    plan = TransferPlanner().build_transfer_plan(request, route, kv_meta)

    assert plan.route_name == "strong_prefill->aligned_lane"
    assert plan.estimated_transfer_bytes == kv_meta.estimated_kv_bytes
    assert plan.source_ranks == (0, 1, 2, 3)
    assert plan.target_ranks == (4, 5, 6, 7)
    assert plan.estimated_transfer_time_ms > 0


def test_simulated_transfer_engine_reports_plan_estimate() -> None:
    route = default_topologies()[0]
    request = Request("run", 512, 16, 0.0, 100.0, 1000.0)
    plan = TransferPlanner().build_transfer_plan(
        request, route, KVMetadata(2, 8, 64, 512, 2, 2)
    )

    result = TransferEngine().run(plan)

    assert result.simulated is True
    assert result.elapsed_time_ms == plan.estimated_transfer_time_ms
