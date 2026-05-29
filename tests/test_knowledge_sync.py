import pytest
from fleet.knowledge_sync import KnowledgeNode, KnowledgeEdge, KnowledgeSync


class TestKnowledgeNode:
    def test_to_dict(self):
        n = KnowledgeNode(id="n1", type="agent", properties={"x": 1})
        d = n.to_dict()
        assert d["id"] == "n1"
        assert d["type"] == "agent"


class TestKnowledgeEdge:
    def test_to_dict(self):
        e = KnowledgeEdge("a", "b", "rel", 0.5)
        d = e.to_dict()
        assert d["source"] == "a"
        assert d["relation"] == "rel"


class TestKnowledgeSync:
    def test_init(self):
        ks = KnowledgeSync()
        assert ks.nodes == {}
        assert ks.edges == []

    def test_add_node(self):
        ks = KnowledgeSync()
        n = ks.add_node("n1", "agent")
        assert "n1" in ks.nodes
        assert ks._node_types["agent"] == 1

    def test_add_edge(self):
        ks = KnowledgeSync()
        ks.add_node("a", "x")
        ks.add_node("b", "x")
        e = ks.add_edge("a", "b", "rel")
        assert e.source == "a"
        assert len(ks.edges) == 1

    def test_ingest_fleet_event(self):
        ks = KnowledgeSync()
        eid = ks.ingest_fleet_event({"type": "breeding", "id": "e1"})
        assert eid == "e1"
        assert "e1" in ks.nodes

    def test_ingest_with_related(self):
        ks = KnowledgeSync()
        ks.add_node("agent_42", "agent")
        eid = ks.ingest_fleet_event({"type": "breeding", "id": "e2", "agent_id": "agent_42"})
        assert eid == "e2"
        edges = [e for e in ks.edges if e.source == "e2"]
        assert len(edges) == 1
        assert edges[0].target == "agent_42"

    def test_query_knowledge(self):
        ks = KnowledgeSync()
        ks.add_node("n1", "agent", {"role": "test"})
        ks.add_node("n2", "room")
        results = ks.query_knowledge("agent")
        assert len(results) == 1
        assert results[0].id == "n1"

    def test_query_knowledge_with_filters(self):
        ks = KnowledgeSync()
        ks.add_node("n1", "agent", {"role": "test"})
        ks.add_node("n2", "agent", {"role": "other"})
        results = ks.query_knowledge("agent", {"role": "test"})
        assert len(results) == 1
        assert results[0].id == "n1"

    def test_get_related(self):
        ks = KnowledgeSync()
        ks.add_node("a", "x")
        ks.add_node("b", "x")
        ks.add_node("c", "x")
        ks.add_edge("a", "b", "rel")
        ks.add_edge("a", "c", "rel")
        related = ks.get_related("a")
        assert len(related) == 2

    def test_get_related_by_relation(self):
        ks = KnowledgeSync()
        ks.add_node("a", "x")
        ks.add_node("b", "x")
        ks.add_node("c", "x")
        ks.add_edge("a", "b", "rel1")
        ks.add_edge("a", "c", "rel2")
        related = ks.get_related("a", "rel1")
        assert len(related) == 1
        assert related[0].id == "b"

    def test_get_graph_stats(self):
        ks = KnowledgeSync()
        ks.add_node("a", "x")
        ks.add_node("b", "x")
        ks.add_edge("a", "b", "rel")
        stats = ks.get_graph_stats()
        assert stats["nodes"] == 2
        assert stats["edges"] == 1
        assert stats["avg_degree"] == 0.5

    def test_export_neo4j(self):
        ks = KnowledgeSync()
        ks.add_node("a", "x", {"prop": 1})
        ks.add_edge("a", "b", "rel")
        stmts = ks.export_neo4j()
        assert len(stmts) == 2
        assert "CREATE" in stmts[0]

    def test_export_graphml(self):
        ks = KnowledgeSync()
        ks.add_node("a", "x")
        ks.add_node("b", "x")
        ks.add_edge("a", "b", "rel")
        xml = ks.export_graphml()
        assert "graphml" in xml
        assert 'id="a"' in xml

    def test_export_json(self):
        ks = KnowledgeSync()
        ks.add_node("a", "x")
        j = ks.export_json()
        assert "a" in j
        assert "nodes" in j

    def test_to_dict(self):
        ks = KnowledgeSync()
        ks.add_node("a", "x")
        d = ks.to_dict()
        assert d["nodes"] == 1
