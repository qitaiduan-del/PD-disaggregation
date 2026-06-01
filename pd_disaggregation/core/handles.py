"""Static route handle descriptions used by a future communication backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

from .topology import TopologyConfig


@dataclass(frozen=True)
class RouteHandle:
    """Pre-created rank mapping for one topology route."""

    route_name: str
    source_ranks: Tuple[int, ...]
    target_ranks: Tuple[int, ...]


class RouteHandleBundle:
    """Collection of rank mappings addressable by topology name."""

    def __init__(self, handles: Iterable[RouteHandle]) -> None:
        self._handles: Dict[str, RouteHandle] = {
            handle.route_name: handle for handle in handles
        }

    @classmethod
    def from_topologies(
        cls, topologies: Iterable[TopologyConfig]
    ) -> "RouteHandleBundle":
        """Construct simple contiguous prefill/decode rank mappings."""

        handles = []
        for topology in topologies:
            source_ranks = tuple(range(topology.prefill_world_size))
            target_ranks = tuple(
                range(
                    topology.prefill_world_size,
                    topology.prefill_world_size + topology.decode_world_size,
                )
            )
            handles.append(RouteHandle(topology.name, source_ranks, target_ranks))
        return cls(handles)

    def get(self, route_name: str) -> RouteHandle:
        """Return a named route mapping or raise a descriptive error."""

        try:
            return self._handles[route_name]
        except KeyError as exc:
            raise KeyError(f"No route handle registered for {route_name!r}.") from exc
