"""Two-stage dynamic PD route selection."""

from __future__ import annotations

from typing import Iterable, Tuple

from .cost_model import SLOCostModel
from .request import KVMetadata, Request
from .system_state import SystemState
from .topology import TopologyConfig, default_topologies


class DynamicPDRouter:
    """Select independent prefill and decode layouts using estimated latency."""

    def __init__(
        self,
        topologies: Iterable[TopologyConfig] | None = None,
        cost_model: SLOCostModel | None = None,
        baseline_route_name: str = "balanced",
    ) -> None:
        self.topologies: Tuple[TopologyConfig, ...] = tuple(
            topologies or default_topologies()
        )
        if not self.topologies:
            raise ValueError("At least one topology is required.")
        self.cost_model = cost_model or SLOCostModel()
        self.baseline_route_name = baseline_route_name

    def _available(self, state: SystemState) -> Tuple[TopologyConfig, ...]:
        available = tuple(
            route
            for route in self.topologies
            if route.required_gpu_count <= state.gpu_count
        )
        if not available:
            raise ValueError("No topology fits the available GPU count.")
        return available

    def _baseline(self, available: Tuple[TopologyConfig, ...]) -> TopologyConfig:
        return next(
            (
                route
                for route in available
                if route.name == self.baseline_route_name
            ),
            available[0],
        )

    def select_prefill_route(
        self, request: Request, system_state: SystemState
    ) -> TopologyConfig:
        """Pick a prompt execution layout before KV cache is produced."""

        available = self._available(system_state)
        baseline = self._baseline(available)
        # Decode alternatives may share an identical prefill layout.  They are
        # not meaningful alternatives until the post-prefill decision point.
        unique_prefill_routes: dict[tuple[int, int], TopologyConfig] = {
            (baseline.prefill_tp, baseline.prefill_dp): baseline
        }
        for route in available:
            unique_prefill_routes.setdefault(
                (route.prefill_tp, route.prefill_dp), route
            )
        prefill_candidates = tuple(unique_prefill_routes.values())
        baseline_cost = self.cost_model.estimate(request, system_state, baseline)
        lightly_loaded = (
            system_state.prefill_queue_len <= system_state.active_prefill_workers
            and request.prompt_len <= 2048
            and baseline_cost.total_ttft_ms <= request.slo_ttft_ms
        )
        if lightly_loaded:
            return baseline

        def score(route: TopologyConfig) -> tuple[float, int]:
            estimate = self.cost_model.estimate(request, system_state, route)
            slo_penalty = (
                max(0.0, estimate.total_ttft_ms - request.slo_ttft_ms) * 3.0
            )
            prefill_ready = (
                estimate.prefill_queue_delay_ms + estimate.prefill_time_ms
            )
            return prefill_ready + slo_penalty, -route.prefill_tp

        return min(prefill_candidates, key=score)

    def select_decode_route(
        self,
        request: Request,
        system_state: SystemState,
        kv_meta: KVMetadata,
    ) -> TopologyConfig:
        """Pick a post-prefill layout using KV transfer and decode pressure."""

        available = self._available(system_state)
        baseline = self._baseline(available)
        baseline_cost = self.cost_model.estimate(
            request, system_state, baseline, kv_meta
        )
        lightly_loaded = (
            system_state.decode_queue_len <= system_state.active_decode_workers
            and baseline_cost.total_ttft_ms <= request.slo_ttft_ms
        )
        if lightly_loaded:
            return baseline

        def score(route: TopologyConfig) -> tuple[float, int]:
            estimate = self.cost_model.estimate(
                request, system_state, route, kv_meta
            )
            violation_penalty = 0.0
            if estimate.total_ttft_ms > request.slo_ttft_ms:
                violation_penalty += (
                    estimate.total_ttft_ms - request.slo_ttft_ms
                ) * 4.0
            if estimate.total_e2e_latency_ms > request.slo_e2e_ms:
                violation_penalty += (
                    estimate.total_e2e_latency_ms - request.slo_e2e_ms
                ) * 0.5
            return estimate.total_decode_latency_ms + violation_penalty, -route.decode_dp

        return min(available, key=score)

    def select_route(
        self,
        request: Request,
        system_state: SystemState,
        stage: str = "prefill",
        kv_meta: KVMetadata | None = None,
    ) -> TopologyConfig:
        """Generic stage dispatcher matching the scheduler-facing API."""

        if stage == "prefill":
            return self.select_prefill_route(request, system_state)
        if stage == "decode":
            if kv_meta is None:
                raise ValueError("kv_meta is required when selecting a decode route.")
            return self.select_decode_route(request, system_state, kv_meta)
        raise ValueError(f"Unknown routing stage: {stage!r}.")
