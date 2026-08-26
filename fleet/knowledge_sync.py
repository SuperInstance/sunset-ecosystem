"""
Knowledge Graph Sync

Synchronizes fleet state with the Lucineer knowledge graph system.
Bidirectional sync: fleet events -> knowledge graph, and
knowledge graph queries -> fleet context.

Usage:
    from fleet.knowledge_sync import KnowledgeSync
    sync = KnowledgeSync()
    sync.ingest_fleet_event({"type": "breeding", "generation": 42})
    context = sync.query_knowledge("What is the best breeding strategy?")
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class KnowledgeNode:
    """A node in the knowledge graph."""

    id: str
    type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    source: str = "fleet"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "properties": self.properties,
            "timestamp": self.timestamp,
            "source": self.source,
        }


@dataclass
class KnowledgeEdge:
    """An edge between knowledge nodes."""

    source: str
    target: str
    relation: str
    weight: float = 1.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "weight": self.weight,
            "timestamp": self.timestamp,
        }


class KnowledgeSync:
    """
    Syncs fleet state with a knowledge graph.

    In production, this would connect to Lucineer or Neo4j.
    For now, maintains an in-memory graph that can export to
    standard graph formats.
    """

    def __init__(self, fleet_node_id: str = "default"):
        self.fleet_node_id = fleet_node_id
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: List[KnowledgeEdge] = []
        self._node_types: Dict[str, int] = {}

    def add_node(
        self, node_id: str, node_type: str, properties: Optional[Dict[str, Any]] = None
    ) -> KnowledgeNode:
        """Add a node to the knowledge graph."""
        node = KnowledgeNode(
            id=node_id,
            type=node_type,
            properties=properties or {},
            source=self.fleet_node_id,
        )
        self.nodes[node_id] = node
        self._node_types[node_type] = self._node_types.get(node_type, 0) + 1
        return node

    def add_edge(
        self, source: str, target: str, relation: str, weight: float = 1.0
    ) -> KnowledgeEdge:
        """Add an edge between nodes."""
        edge = KnowledgeEdge(source, target, relation, weight)
        self.edges.append(edge)
        return edge

    def ingest_fleet_event(self, event: Dict[str, Any]) -> str:
        """
        Ingest a fleet event into the knowledge graph.
        Returns the node ID created.
        """
        event_type = event.get("type", "unknown")
        event_id = event.get("id", f"{event_type}_{time.time()}")

        node = self.add_node(event_id, event_type, event)

        # Link to related entities
        for key in ["agent_id", "parent_id", "room_id", "vessel_id"]:
            if key in event and event[key]:
                related_id = str(event[key])
                if related_id in self.nodes:
                    self.add_edge(event_id, related_id, f"has_{key}")

        return event_id

    def query_knowledge(
        self, query_type: str, filters: Optional[Dict[str, Any]] = None
    ) -> List[KnowledgeNode]:
        """
        Query the knowledge graph.

        Args:
            query_type: Type of nodes to query
            filters: Property filters
        """
        results = []
        for node in self.nodes.values():
            if node.type != query_type:
                continue
            if filters:
                match = all(node.properties.get(k) == v for k, v in filters.items())
                if not match:
                    continue
            results.append(node)
        return results

    def get_related(
        self, node_id: str, relation: Optional[str] = None
    ) -> List[KnowledgeNode]:
        """Get nodes related to a given node."""
        related_ids = []
        for edge in self.edges:
            if edge.source == node_id:
                if relation is None or edge.relation == relation:
                    related_ids.append(edge.target)
            elif edge.target == node_id:
                if relation is None or edge.relation == relation:
                    related_ids.append(edge.source)

        return [self.nodes[rid] for rid in related_ids if rid in self.nodes]

    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph."""
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "node_types": self._node_types.copy(),
            "avg_degree": len(self.edges) / len(self.nodes) if self.nodes else 0,
        }

    def export_neo4j(self) -> List[str]:
        """Export as Neo4j Cypher statements."""
        statements = []
        for node in self.nodes.values():
            props = json.dumps(node.properties)
            statements.append(
                f"CREATE (n:{node.type} {{id: '{node.id}', properties: '{props}'}})"
            )
        for edge in self.edges:
            statements.append(
                f"MATCH (a {{id: '{edge.source}'}}), (b {{id: '{edge.target}'}}) "
                f"CREATE (a)-[:{edge.relation} {{weight: {edge.weight}}}]->(b)"
            )
        return statements

    def export_graphml(self) -> str:
        """Export as GraphML XML."""
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?',
            '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
            '  <graph id="fleet" edgedefault="directed">',
        ]
        for node in self.nodes.values():
            lines.append(f'    <node id="{node.id}"/>')
        for edge in self.edges:
            lines.append(
                f'    <edge source="{edge.source}" target="{edge.target}" '
                f'label="{edge.relation}"/>'
            )
        lines.extend(
            [
                "  </graph>",
                "</graphml>",
            ]
        )
        return "\n".join(lines)

    def export_json(self) -> str:
        """Export as JSON."""
        return json.dumps(
            {
                "nodes": [n.to_dict() for n in self.nodes.values()],
                "edges": [e.to_dict() for e in self.edges],
                "stats": self.get_graph_stats(),
            },
            indent=2,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.fleet_node_id,
            "stats": self.get_graph_stats(),
            "nodes": len(self.nodes),
            "edges": len(self.edges),
        }
