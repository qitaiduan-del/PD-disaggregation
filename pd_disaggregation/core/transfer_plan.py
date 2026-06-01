"""KV-cache handoff plan generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .handles import RouteHandleBundle
from .profiler import TransferProfiler
from .request import KVMetadata, Request
from .topology import TopologyConfig


@dataclass(frozen=True)
class TransferPlan:
    """An executable description of one simulated prefill-to-decode handoff."""

    request_id: str
    route_name: str
    estimated_transfer_bytes: int
    estimated_transfer_time_ms: float
    source_ranks: Tuple[int, ...]
    target_ranks: Tuple[int, ...]


class TransferPlanner:
    """Build transfer plans from a selected resolved route and KV metadata."""

    def __init__(self, profiler: TransferProfiler | None = None) -> None:
        self.profiler = profiler or TransferProfiler.default()

    def build_transfer_plan(
        self, request: Request, route: TopologyConfig, kv_meta: KVMetadata
    ) -> TransferPlan:
        """Create rank mapping and profile-based estimated handoff duration."""

        handle = RouteHandleBundle.from_topologies((route,)).get(route.name)
        transfer_time = self.profiler.estimate_transfer_time_ms(
            kv_meta.estimated_kv_bytes, route
        )
        return TransferPlan(
            request_id=request.request_id,
            route_name=route.name,
            estimated_transfer_bytes=kv_meta.estimated_kv_bytes,
            estimated_transfer_time_ms=transfer_time,
            source_ranks=handle.source_ranks,
            target_ranks=handle.target_ranks,
        )
