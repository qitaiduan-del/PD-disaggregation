from pd_disaggregation.core import (
    KVMetadata,
    Request,
    SLOCostModel,
    SystemState,
    default_topologies,
)


def make_request(prompt_len: int = 2048) -> Request:
    return Request("cost", prompt_len, 64, 0.0, 1000.0, 5000.0)


def make_state(prefill_queue: int = 0, decode_queue: int = 0) -> SystemState:
    return SystemState(prefill_queue, decode_queue, 2, 2, 8, 0.0)


def test_prefill_latency_grows_with_prompt_and_queue_pressure() -> None:
    model = SLOCostModel()
    route = default_topologies()[0]

    short = model.estimate(make_request(512), make_state(), route)
    long_queued = model.estimate(make_request(4096), make_state(8, 0), route)

    assert long_queued.prefill_time_ms > short.prefill_time_ms
    assert long_queued.prefill_queue_delay_ms > short.prefill_queue_delay_ms


def test_transfer_latency_grows_with_kv_payload() -> None:
    model = SLOCostModel()
    route = default_topologies()[0]
    small = KVMetadata(32, 32, 128, 512, 2, 2)
    large = KVMetadata(32, 32, 128, 4096, 2, 2)

    small_cost = model.estimate(make_request(), make_state(), route, small)
    large_cost = model.estimate(make_request(), make_state(), route, large)

    assert large.estimated_kv_bytes == small.estimated_kv_bytes * 8
    assert large_cost.transfer_time_ms > small_cost.transfer_time_ms


def test_tight_slo_is_reported_as_violation() -> None:
    request = Request("tight", 4096, 128, 0.0, 1.0, 1.0)
    estimate = SLOCostModel().estimate(request, make_state(), default_topologies()[0])

    assert estimate.slo_violation is True
