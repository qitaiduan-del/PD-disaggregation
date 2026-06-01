"""Observable scheduler state at routing time."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SystemState:
    """Queue and worker pressure visible to the two-stage router."""

    prefill_queue_len: int
    decode_queue_len: int
    active_prefill_workers: int
    active_decode_workers: int
    gpu_count: int
    current_time_ms: float

    def __post_init__(self) -> None:
        if self.prefill_queue_len < 0 or self.decode_queue_len < 0:
            raise ValueError("Queue lengths cannot be negative.")
        if self.active_prefill_workers <= 0 or self.active_decode_workers <= 0:
            raise ValueError("Active worker counts must be positive.")
        if self.gpu_count <= 0:
            raise ValueError("gpu_count must be positive.")
