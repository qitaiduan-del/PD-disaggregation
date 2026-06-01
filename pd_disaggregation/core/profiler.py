"""Profile-driven transfer-time estimation."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

from .topology import TopologyConfig


@dataclass(frozen=True)
class TransferProfileEntry:
    """Effective bandwidth and startup cost observed for one route."""

    route_name: str
    bandwidth_gbps: float
    startup_latency_ms: float

    def __post_init__(self) -> None:
        if self.bandwidth_gbps <= 0 or self.startup_latency_ms < 0:
            raise ValueError("A transfer profile needs positive bandwidth and non-negative latency.")


class TransferProfiler:
    """Estimate handoff latency from defaults or measured CSV profile rows."""

    def __init__(
        self,
        profiles: Iterable[TransferProfileEntry] | None = None,
        default_bandwidth_gbps: float = 110.0,
        default_startup_latency_ms: float = 0.08,
    ) -> None:
        if default_bandwidth_gbps <= 0 or default_startup_latency_ms < 0:
            raise ValueError("Default transfer profile values are invalid.")
        self._profiles: Dict[str, TransferProfileEntry] = {
            profile.route_name: profile for profile in (profiles or ())
        }
        self.default_bandwidth_gbps = default_bandwidth_gbps
        self.default_startup_latency_ms = default_startup_latency_ms

    @classmethod
    def default(cls) -> "TransferProfiler":
        """Return conservative stand-in profiles until real measurement is supplied."""

        return cls(
            profiles=(
                TransferProfileEntry("balanced", 112.0, 0.08),
                TransferProfileEntry("strong_prefill", 104.0, 0.10),
                TransferProfileEntry("cross_tp", 96.0, 0.13),
                TransferProfileEntry("aligned_lane", 118.0, 0.07),
            )
        )

    @classmethod
    def from_csv(cls, path: str | Path) -> "TransferProfiler":
        """Load `route_name,bandwidth_gbps,startup_latency_ms` CSV rows."""

        with Path(path).open("r", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            required = {"route_name", "bandwidth_gbps", "startup_latency_ms"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(
                    "Transfer profile CSV must contain route_name, bandwidth_gbps, "
                    "and startup_latency_ms columns."
                )
            profiles = [
                TransferProfileEntry(
                    route_name=row["route_name"],
                    bandwidth_gbps=float(row["bandwidth_gbps"]),
                    startup_latency_ms=float(row["startup_latency_ms"]),
                )
                for row in reader
            ]
        return cls(profiles=profiles)

    def estimate_transfer_time_ms(
        self, estimated_kv_bytes: int, route: TopologyConfig
    ) -> float:
        """Estimate point-to-point KV time with route communication overhead."""

        if estimated_kv_bytes < 0:
            raise ValueError("estimated_kv_bytes cannot be negative.")
        profile = self._profiles.get(route.name)
        bandwidth = (
            profile.bandwidth_gbps if profile else self.default_bandwidth_gbps
        )
        startup = (
            profile.startup_latency_ms
            if profile
            else self.default_startup_latency_ms
        )
        payload_ms = estimated_kv_bytes / (bandwidth * 1_000_000.0)
        return startup + payload_ms * route.communication_factor
