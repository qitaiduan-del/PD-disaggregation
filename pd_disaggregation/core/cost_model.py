"""Explainable queue, compute, transfer, and SLO latency estimates."""

from __future__ import annotations

from dataclasses import dataclass

from .profiler import TransferProfiler
from .request import KVMetadata, Request
from .system_state import SystemState
from .topology import TopologyConfig


@dataclass(frozen=True)
class CostEstimate:
    """Latency estimate for one complete prefill-to-decode route."""

    prefill_time_ms: float
    decode_time_per_token_ms: float
    transfer_time_ms: float
    prefill_queue_delay_ms: float
    decode_queue_delay_ms: float
    queue_delay_ms: float
    total_ttft_ms: float
    total_decode_latency_ms: float
    total_e2e_latency_ms: float
    slo_violation: bool


class SLOCostModel:
    """Compute transparent analytical estimates for route comparison.

    TP accelerates per-request compute with less than linear scaling; DP and
    queue_parallelism_factor increase independent request throughput. Decode
    queue time assumes queued jobs have 48 tokens of remaining work, an
    explicit workload prior that may later be learned from scheduler metrics.
    """

    def __init__(self, profiler: TransferProfiler | None = None) -> None:
        self.profiler = profiler or TransferProfiler.default()

    @staticmethod
    def _tp_efficiency(tp_size: int) -> float:
        return 1.0 + 0.78 * (tp_size - 1)

    def prefill_time_ms(
        self, request: Request, state: SystemState, route: TopologyConfig
    ) -> float:
        """Estimate prompt computation including modest contention slowdown."""

        pressure = state.prefill_queue_len / max(
            state.active_prefill_workers * route.prefill_dp, 1
        )
        compute_ms = (
            2.5
            + 22.0
            * (request.prompt_len / 1024.0)
            / self._tp_efficiency(route.prefill_tp)
        )
        communication_ms = (
            0.75 * (route.prefill_tp - 1) * route.communication_factor
        )
        return (compute_ms + communication_ms) * (1.0 + 0.05 * pressure)

    def decode_time_per_token_ms(
        self, state: SystemState, route: TopologyConfig
    ) -> float:
        """Estimate steady-state per-token decode compute for this layout."""

        pressure = state.decode_queue_len / max(
            state.active_decode_workers * route.decode_dp, 1
        )
        compute_ms = 6.5 / self._tp_efficiency(route.decode_tp)
        communication_ms = (
            0.18 * (route.decode_tp - 1) * route.communication_factor
        )
        return (compute_ms + communication_ms) * (1.0 + 0.02 * min(pressure, 4.0))

    def queue_delays_ms(
        self, state: SystemState, route: TopologyConfig
    ) -> tuple[float, float]:
        """Estimate separate prefill and decode queue waits."""

        prefill_capacity = (
            state.active_prefill_workers
            * route.prefill_dp
            * route.queue_parallelism_factor
        )
        prefill_delay = 4.0 * state.prefill_queue_len / prefill_capacity

        decode_capacity = (
            state.active_decode_workers
            * route.decode_dp
            * route.queue_parallelism_factor
        )
        unpressured_decode_token_ms = (
            6.5 / self._tp_efficiency(route.decode_tp)
            + 0.18 * (route.decode_tp - 1) * route.communication_factor
        )
        decode_delay = (
            state.decode_queue_len * 48.0 * unpressured_decode_token_ms
            / decode_capacity
        )
        return prefill_delay, decode_delay

    def estimate(
        self,
        request: Request,
        state: SystemState,
        route: TopologyConfig,
        kv_meta: KVMetadata | None = None,
    ) -> CostEstimate:
        """Estimate end-to-end latency; omit KV metadata before handoff exists."""

        prefill_time = self.prefill_time_ms(request, state, route)
        decode_token_time = self.decode_time_per_token_ms(state, route)
        prefill_queue, decode_queue = self.queue_delays_ms(state, route)
        transfer_time = (
            self.profiler.estimate_transfer_time_ms(
                kv_meta.estimated_kv_bytes, route
            )
            if kv_meta is not None
            else 0.0
        )
        ttft = (
            prefill_queue
            + prefill_time
            + transfer_time
            + decode_queue
            + decode_token_time
        )
        decode_latency = (
            transfer_time + decode_queue + decode_token_time * request.output_len
        )
        e2e = prefill_queue + prefill_time + decode_latency
        violation = ttft > request.slo_ttft_ms or e2e > request.slo_e2e_ms
        return CostEstimate(
            prefill_time_ms=prefill_time,
            decode_time_per_token_ms=decode_token_time,
            transfer_time_ms=transfer_time,
            prefill_queue_delay_ms=prefill_queue,
            decode_queue_delay_ms=decode_queue,
            queue_delay_ms=prefill_queue + decode_queue,
            total_ttft_ms=ttft,
            total_decode_latency_ms=decode_latency,
            total_e2e_latency_ms=e2e,
            slo_violation=violation,
        )
