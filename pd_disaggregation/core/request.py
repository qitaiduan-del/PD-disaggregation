"""Request and KV-cache metadata models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Request:
    """A serving request and its latency service level objectives."""

    request_id: str
    prompt_len: int
    output_len: int
    arrival_time_ms: float
    slo_ttft_ms: float
    slo_e2e_ms: float

    def __post_init__(self) -> None:
        if self.prompt_len <= 0 or self.output_len <= 0:
            raise ValueError("prompt_len and output_len must be positive.")
        if self.slo_ttft_ms <= 0 or self.slo_e2e_ms <= 0:
            raise ValueError("SLO values must be positive.")


@dataclass(frozen=True)
class KVMetadata:
    """Shape metadata used to estimate a request's KV handoff cost."""

    num_layers: int
    num_heads: int
    head_dim: int
    seq_len: int
    dtype_bytes: int
    tp_size: int
    estimated_kv_bytes: int = 0

    def __post_init__(self) -> None:
        dimensions = (
            self.num_layers,
            self.num_heads,
            self.head_dim,
            self.seq_len,
            self.dtype_bytes,
            self.tp_size,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("All KV metadata dimensions must be positive.")
        if self.estimated_kv_bytes < 0:
            raise ValueError("estimated_kv_bytes cannot be negative.")
        if self.estimated_kv_bytes == 0:
            # Two tensors are transferred per layer: K and V.
            estimated = (
                2
                * self.num_layers
                * self.num_heads
                * self.head_dim
                * self.seq_len
                * self.dtype_bytes
            )
            object.__setattr__(self, "estimated_kv_bytes", estimated)
