"""FluxVectorTable — backward-compatible re-export.

The canonical implementation now lives in ``swarm.flux_vector_table``.
This module re-exports the same symbols so existing imports continue
working without change.
"""

from __future__ import annotations

from swarm.flux_vector_table import AgentMeta, AgentVector, FluxVectorTable

__all__ = ["AgentMeta", "AgentVector", "FluxVectorTable"]
