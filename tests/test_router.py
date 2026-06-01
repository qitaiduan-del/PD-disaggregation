import pytest

from pd_disaggregation.core import DynamicPDRouter, KVMetadata, Request, SystemState


def request(prompt_len: int, output_len: int = 64) -> Request:
    return Request("router", prompt_len, output_len, 0.0, 300.0, 2000.0)


def state(prefill_queue: int = 0, decode_queue: int = 0) -> SystemState:
    return SystemState(prefill_queue, decode_queue, 2, 2, 8, 0.0)


def kv(seq_len: int) -> KVMetadata:
    return KVMetadata(32, 32, 128, seq_len, 2, 2)


def test_light_request_keeps_balanced_route() -> None:
    router = DynamicPDRouter()

    assert router.select_prefill_route(request(512), state()).name == "balanced"
    assert (
        router.select_decode_route(request(512), state(), kv(512)).name
        == "balanced"
    )


def test_long_prompt_selects_stronger_prefill_parallelism() -> None:
    route = DynamicPDRouter().select_prefill_route(request(8192), state())

    assert route.name == "strong_prefill"
    assert route.prefill_tp == 4


def test_prefill_does_not_choose_a_decode_only_lane_variant() -> None:
    route = DynamicPDRouter().select_prefill_route(request(2048), state(prefill_queue=8))

    assert route.name in {"balanced", "strong_prefill"}


def test_decode_pressure_selects_parallel_lanes() -> None:
    route = DynamicPDRouter().select_decode_route(
        request(2048, output_len=32), state(decode_queue=24), kv(2048)
    )

    assert route.name == "aligned_lane"
    assert route.decode_dp == 4


def test_decode_selection_requires_kv_metadata() -> None:
    with pytest.raises(ValueError):
        DynamicPDRouter().select_route(request(1024), state(), stage="decode")
