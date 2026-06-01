"""Topology descriptions and built-in scheduling candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class TopologyConfig:
    """Parallel layout and relative communication/queue characteristics."""

    name: str
    prefill_tp: int
    prefill_dp: int
    decode_tp: int
    decode_dp: int
    communication_factor: float
    queue_parallelism_factor: float

    def __post_init__(self) -> None:
        parallelism = (self.prefill_tp, self.prefill_dp, self.decode_tp, self.decode_dp)
        if not self.name:
            raise ValueError("Topology name cannot be empty.")
        if any(value <= 0 for value in parallelism):
            raise ValueError("TP and DP values must be positive.")
        if self.communication_factor <= 0 or self.queue_parallelism_factor <= 0:
            raise ValueError("Topology factors must be positive.")

    @property
    def prefill_world_size(self) -> int:
        """Number of ranks assigned to the prefill pool."""

        return self.prefill_tp * self.prefill_dp

    @property
    def decode_world_size(self) -> int:
        """Number of ranks assigned to the decode pool."""

        return self.decode_tp * self.decode_dp

    @property
    def required_gpu_count(self) -> int:
        """Number of ranks needed when prefill and decode pools coexist."""

        return self.prefill_world_size + self.decode_world_size


def combine_stage_routes(
    prefill_route: TopologyConfig, decode_route: TopologyConfig
) -> TopologyConfig:
    """Resolve independently selected stages into one handoff topology."""

    return TopologyConfig(
        name=f"{prefill_route.name}->{decode_route.name}",
        prefill_tp=prefill_route.prefill_tp,
        prefill_dp=prefill_route.prefill_dp,
        decode_tp=decode_route.decode_tp,
        decode_dp=decode_route.decode_dp,
        communication_factor=max(
            prefill_route.communication_factor, decode_route.communication_factor
        ),
        queue_parallelism_factor=decode_route.queue_parallelism_factor,
    )


def default_topologies() -> Tuple[TopologyConfig, ...]:
    """Return small, explainable candidates for an eight-rank PD deployment."""

    return (
        TopologyConfig(
            name="balanced",
            prefill_tp=2,
            prefill_dp=2,
            decode_tp=2,
            decode_dp=2,
            communication_factor=1.0,
            queue_parallelism_factor=1.0,
        ),
        TopologyConfig(
            name="strong_prefill",
            prefill_tp=4,
            prefill_dp=1,
            decode_tp=2,
            decode_dp=2,
            communication_factor=1.12,
            queue_parallelism_factor=0.9,
        ),
        TopologyConfig(
            name="cross_tp",
            prefill_tp=2,
            prefill_dp=2,
            decode_tp=4,
            decode_dp=1,
            communication_factor=1.18,
            queue_parallelism_factor=0.72,
        ),
        TopologyConfig(
            name="aligned_lane",
            prefill_tp=2,
            prefill_dp=2,
            decode_tp=1,
            decode_dp=4,
            communication_factor=0.86,
            queue_parallelism_factor=1.35,
        ),
    )
