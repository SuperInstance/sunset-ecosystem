"""FluxVectorTable — re-export with diversity-search extensions.

The canonical implementation lives in swarm.vector_table; this module
exists so that task specs and imports can reference
``swarm.flux_vector_table`` directly.
"""
from __future__ import annotations

from swarm.vector_table import AgentMeta, AgentVector, FluxVectorTable

__all__ = ["AgentMeta", "AgentVector", "FluxVectorTable"]
