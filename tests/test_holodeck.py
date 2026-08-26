"""Tests for fleet.holodeck — 3D Room Grid Visualizer.

Run with pytest from the repo root:
    python -m pytest sunset-ecosystem/tests/test_holodeck.py -v
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from fleet.holodeck import (
    AgentAvatar,
    Holodeck,
    MockPlatoSource,
    RoomNode,
    agent_color_for_phase,
    room_color_for_diversity,
)


# ── colour mapping ─────────────────────────────────────


class TestColourMapping:
    def test_room_colour_low_diversity(self):
        assert room_color_for_diversity(0.0) == 0x0A1628
        assert room_color_for_diversity(0.1) == 0x0A1628
        assert room_color_for_diversity(0.32) == 0x0A1628

    def test_room_colour_mid_diversity(self):
        assert room_color_for_diversity(0.33) == 0x39FF14
        assert room_color_for_diversity(0.50) == 0x39FF14
        assert room_color_for_diversity(0.65) == 0x39FF14

    def test_room_colour_high_diversity(self):
        assert room_color_for_diversity(0.66) == 0xDC143C
        assert room_color_for_diversity(0.99) == 0xDC143C
        assert room_color_for_diversity(1.0) == 0xDC143C

    def test_agent_colour_incubating(self):
        assert agent_color_for_phase("incubating") == 0xF0F0E6

    def test_agent_colour_competing(self):
        assert agent_color_for_phase("competing") == 0xFFB347

    def test_agent_colour_breeding(self):
        assert agent_color_for_phase("breeding") == 0x39FF14

    def test_agent_colour_sunsetting(self):
        assert agent_color_for_phase("sunsetting") == 0x8A8A8A

    def test_agent_colour_unknown(self):
        assert agent_color_for_phase("unknown_phase") == 0xF0F0E6


# ── RoomNode / AgentAvatar dataclasses ─────────────────


class TestDataStructures:
    def test_room_node_basic(self):
        r = RoomNode(room_id="alpha", position=(1.0, 2.0, 3.0), capacity=12)
        assert r.room_id == "alpha"
        np.testing.assert_array_almost_equal(r.position, [1, 2, 3])
        assert r.capacity == 12
        assert r.occupancy == 0
        assert not r.is_overcapacity

    def test_room_node_overcapacity(self):
        r = RoomNode(room_id="beta", occupancy=11, capacity=10)
        assert r.is_overcapacity

    def test_room_node_not_overcapacity(self):
        r = RoomNode(room_id="beta", occupancy=9, capacity=10)
        assert not r.is_overcapacity

    def test_room_node_roundtrip(self):
        r = RoomNode(
            room_id="x",
            position=(0, 1, 2),
            capacity=5,
            occupancy=3,
            diversity_score=0.5,
            thermal_state=0.2,
            agents=["a1", "a2"],
        )
        d = r.to_dict()
        r2 = RoomNode.from_dict(d)
        assert r2.room_id == r.room_id
        assert r2.capacity == r.capacity
        assert r2.agents == r.agents

    def test_agent_avatar_roundtrip(self):
        a = AgentAvatar(
            agent_id="a1",
            room_id="r1",
            position=np.array([1, 2, 3], dtype=np.float32),
            velocity=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            phase="breeding",
            trinity_scores={"ethos": 0.8},
        )
        d = a.to_dict()
        a2 = AgentAvatar.from_dict(d)
        assert a2.agent_id == a.agent_id
        assert a2.phase == a.phase
        assert a2.trinity_scores == a.trinity_scores
        np.testing.assert_array_almost_equal(a2.position, a.position)

    def test_agent_avatar_list_position(self):
        a = AgentAvatar(agent_id="a2", position=[1, 2, 3])
        np.testing.assert_array_almost_equal(a.position, [1, 2, 3])


# ── Holodeck basics ────────────────────────────────────


class TestHolodeckBasics:
    def test_add_room(self):
        hd = Holodeck()
        hd.add_room("r1", (0, 0, 0), capacity=10)
        assert hd.room_count() == 1
        r = hd.get_room("r1")
        assert r is not None
        assert r.capacity == 10
        np.testing.assert_array_almost_equal(r.position, [0, 0, 0])

    def test_add_multiple_rooms(self):
        hd = Holodeck()
        for i in range(5):
            hd.add_room(f"r{i}", (float(i), float(i), float(i)), capacity=i + 1)
        assert hd.room_count() == 5

    def test_add_room_overwrite(self):
        hd = Holodeck()
        hd.add_room("r1", (0, 0, 0), 10)
        hd.add_room("r1", (5, 5, 5), 20)
        r = hd.get_room("r1")
        assert r.capacity == 20

    def test_remove_room(self):
        hd = Holodeck()
        hd.add_room("r1", (0, 0, 0))
        hd.remove_room("r1")
        assert hd.room_count() == 0
        assert hd.get_room("r1") is None

    def test_update_room(self):
        hd = Holodeck()
        hd.add_room("r1", (0, 0, 0), capacity=10)
        hd.update_room("r1", occupancy=5, diversity=0.7, thermal=0.3)
        r = hd.get_room("r1")
        assert r.occupancy == 5
        assert r.diversity_score == 0.7
        assert r.thermal_state == 0.3

    def test_update_room_clamps(self):
        hd = Holodeck()
        hd.add_room("r1", (0, 0, 0))
        hd.update_room("r1", diversity=1.5, thermal=-0.2)
        r = hd.get_room("r1")
        assert r.diversity_score == 1.0
        assert r.thermal_state == 0.0

    def test_update_room_not_found(self):
        hd = Holodeck()
        with pytest.raises(KeyError):
            hd.update_room("missing", occupancy=1)

    def test_add_agent(self):
        hd = Holodeck()
        hd.add_room("r1", (0, 0, 0), capacity=10)
        hd.add_agent("a1", "r1", {"ethos": 0.5})
        assert hd.agent_count() == 1
        a = hd.get_agent("a1")
        assert a.room_id == "r1"
        assert a.trinity_scores["ethos"] == 0.5

    def test_add_agent_without_trinity(self):
        hd = Holodeck()
        hd.add_room("r1", (0, 0, 0))
        hd.add_agent("a1", "r1")
        assert hd.get_agent("a1").trinity_scores == {}

    def test_add_agent_missing_room(self):
        hd = Holodeck()
        with pytest.raises(KeyError):
            hd.add_agent("a1", "r1")

    def test_remove_agent(self):
        hd = Holodeck()
        hd.add_room("r1", (0, 0, 0), capacity=10)
        hd.add_agent("a1", "r1")
        hd.remove_agent("a1")
        assert hd.agent_count() == 0
        assert hd.get_agent("a1") is None
        assert hd.get_room("r1").occupancy == 0
        assert "a1" not in hd.get_room("r1").agents

    def test_set_agent_phase(self):
        hd = Holodeck()
        hd.add_room("r1", (0, 0, 0))
        hd.add_agent("a1", "r1")
        hd.set_agent_phase("a1", "breeding")
        assert hd.get_agent("a1").phase == "breeding"

    def test_set_agent_phase_missing(self):
        hd = Holodeck()
        with pytest.raises(KeyError):
            hd.set_agent_phase("ghost", "breeding")

    def test_move_agent(self):
        hd = Holodeck()
        hd.add_room("r1", (0, 0, 0), capacity=10)
        hd.add_room("r2", (10, 0, 0), capacity=10)
        hd.add_agent("a1", "r1")
        hd.move_agent("a1", "r1", "r2")
        assert hd.get_agent("a1").room_id == "r2"
        assert hd.get_room("r1").occupancy == 0
        assert hd.get_room("r2").occupancy == 1
        assert "a1" in hd.get_room("r2").agents
        assert "a1" not in hd.get_room("r1").agents

    def test_move_agent_missing(self):
        hd = Holodeck()
        hd.add_room("r1", (0, 0, 0))
        hd.add_room("r2", (0, 0, 0))
        with pytest.raises(KeyError):
            hd.move_agent("a1", "r1", "r2")

    def test_move_agent_invalid_room(self):
        hd = Holodeck()
        hd.add_room("r1", (0, 0, 0))
        hd.add_agent("a1", "r1")
        with pytest.raises(KeyError):
            hd.move_agent("a1", "r1", "r2")

    def test_capacity_overflow_detection(self):
        hd = Holodeck()
        hd.add_room("r1", (0, 0, 0), capacity=2)
        hd.add_agent("a1", "r1")
        hd.add_agent("a2", "r1")
        hd.add_agent("a3", "r1")
        assert hd.get_room("r1").is_overcapacity

    def test_capacity_not_overflow(self):
        hd = Holodeck()
        hd.add_room("r1", (0, 0, 0), capacity=5)
        hd.add_agent("a1", "r1")
        assert not hd.get_room("r1").is_overcapacity

    def test_multiple_agent_movement(self):
        hd = Holodeck()
        hd.add_room("r1", (0, 0, 0), capacity=10)
        hd.add_room("r2", (0, 0, 0), capacity=10)
        for i in range(5):
            hd.add_agent(f"a{i}", "r1")
        for i in range(5):
            hd.move_agent(f"a{i}", "r1", "r2")
        assert hd.get_room("r1").occupancy == 0
        assert hd.get_room("r2").occupancy == 5

    def test_connections_tracked(self):
        hd = Holodeck()
        hd.add_room("r1", (0, 0, 0))
        hd.add_room("r2", (0, 0, 0))
        hd.add_room("r3", (0, 0, 0))
        hd.add_agent("a1", "r1")
        hd.move_agent("a1", "r1", "r2")
        hd.move_agent("a1", "r2", "r3")
        scene = hd.get_scene()
        assert ["r1", "r2"] in scene["connections"] or ["r2", "r1"] in scene[
            "connections"
        ]
        assert ["r2", "r3"] in scene["connections"] or ["r3", "r2"] in scene[
            "connections"
        ]

    def test_get_scene_structure(self):
        hd = Holodeck()
        hd.add_room("r1", (0, 0, 0))
        hd.add_agent("a1", "r1")
        scene = hd.get_scene()
        assert "rooms" in scene
        assert "agents" in scene
        assert "connections" in scene
        assert "tick" in scene
        assert scene["rooms"]["r1"]["room_id"] == "r1"
        assert scene["agents"]["a1"]["agent_id"] == "a1"

    def test_snapshot_is_json_serializable(self):
        hd = Holodeck()
        hd.add_room("r1", (0, 0, 0), capacity=5)
        hd.add_agent("a1", "r1", {"ethos": 0.5})
        snap = hd.snapshot()
        dumped = json.dumps(snap)
        loaded = json.loads(dumped)
        assert loaded["rooms"]["r1"]["capacity"] == 5

    def test_snapshot_roundtrip(self):
        hd = Holodeck()
        hd.add_room("r1", (1, 2, 3), capacity=8)
        hd.add_agent("a1", "r1")
        hd.move_agent("a1", "r1", "r1")  # no-op but exercises code
        snap = hd.snapshot()
        assert snap["tick"] == 0
        assert snap["rooms"]["r1"]["position"] == [1.0, 2.0, 3.0]

    def test_jitter_inside(self):
        hd = Holodeck()
        hd.add_room("r1", (0, 0, 0), capacity=10)
        p1 = hd._jitter_inside("r1")
        p2 = hd._jitter_inside("r1")
        assert not np.array_equal(p1, p2)
        # Should be roughly near room centre
        assert np.linalg.norm(p1 - np.array([0, 0, 0])) < 5.0


# ── HTML export ────────────────────────────────────────


class TestHTMLExport:
    def test_export_creates_file(self, tmp_path):
        hd = Holodeck()
        hd.add_room("r1", (0, 0, 0), capacity=5)
        hd.add_agent("a1", "r1")
        out = tmp_path / "deck.html"
        hd.export_html(str(out))
        assert out.exists()
        assert out.stat().st_size > 2000

    def test_export_contains_threejs(self, tmp_path):
        hd = Holodeck()
        hd.add_room("r1", (0, 0, 0))
        out = tmp_path / "deck.html"
        hd.export_html(str(out))
        text = out.read_text()
        assert "three" in text.lower() or "three.module" in text
        assert "OrbitControls" in text

    def test_export_contains_room_data(self, tmp_path):
        hd = Holodeck()
        hd.add_room("r1", (0, 0, 0), capacity=7)
        hd.add_agent("a1", "r1")
        out = tmp_path / "deck.html"
        hd.export_html(str(out))
        text = out.read_text()
        assert "roomsData" in text
        assert "agentsData" in text
        assert "r1" in text

    def test_export_contains_tooltips(self, tmp_path):
        hd = Holodeck()
        hd.add_room("r1", (0, 0, 0))
        hd.add_agent("a1", "r1")
        out = tmp_path / "deck.html"
        hd.export_html(str(out))
        text = out.read_text()
        assert "tooltip" in text
        assert "mousemove" in text

    def test_export_contains_auto_rotate(self, tmp_path):
        hd = Holodeck()
        hd.add_room("r1", (0, 0, 0))
        out = tmp_path / "deck.html"
        hd.export_html(str(out))
        text = out.read_text()
        assert "autoRotate" in text
        assert "autoRotateSpeed" in text


# ── MockPlatoSource ────────────────────────────────────


class TestMockPlatoSource:
    def test_init_counts(self):
        src = MockPlatoSource(room_count=8, agent_count=30)
        assert src.get_room_count() == 8
        assert src.get_agent_count() == 30

    def test_room_states_present(self):
        src = MockPlatoSource(room_count=3, agent_count=5)
        rooms = src.get_room_states()
        assert len(rooms) == 3
        for r in rooms.values():
            assert "position" in r
            assert "capacity" in r
            assert "occupancy" in r
            assert "diversity" in r
            assert "thermal" in r

    def test_agent_states_present(self):
        src = MockPlatoSource(room_count=3, agent_count=5)
        agents = src.get_agent_states()
        assert len(agents) == 5
        for a in agents.values():
            assert "room_id" in a
            assert "phase" in a
            assert "trinity" in a

    def test_tick_increments(self):
        src = MockPlatoSource(room_count=2, agent_count=2)
        src.tick()
        src.tick()
        assert src._tick == 2

    def test_tick_thermal_changes(self):
        src = MockPlatoSource(room_count=2, agent_count=2, seed=99)
        before = src.get_room_states()["room_00"]["thermal"]
        src.tick()
        after = src.get_room_states()["room_00"]["thermal"]
        assert before != after

    def test_tick_agents_move(self):
        src = MockPlatoSource(room_count=5, agent_count=20, movement_prob=0.8, seed=7)
        before = {aid: a["room_id"] for aid, a in src.get_agent_states().items()}
        for _ in range(10):
            src.tick()
        after = {aid: a["room_id"] for aid, a in src.get_agent_states().items()}
        assert before != after

    def test_occupancy_consistent(self):
        src = MockPlatoSource(room_count=4, agent_count=10, seed=11)
        for _ in range(5):
            src.tick()
        rooms = src.get_room_states()
        agents = src.get_agent_states()
        counts = {rid: 0 for rid in rooms}
        for a in agents.values():
            counts[a["room_id"]] += 1
        for rid, r in rooms.items():
            assert r["occupancy"] == counts[rid]

    def test_reproducible_seed(self):
        src1 = MockPlatoSource(room_count=3, agent_count=4, seed=123)
        src2 = MockPlatoSource(room_count=3, agent_count=4, seed=123)
        assert src1.get_room_states() == src2.get_room_states()
        assert src1.get_agent_states() == src2.get_agent_states()


# ── Integration: Holodeck + MockPlatoSource ──────────────


class TestIntegration:
    def test_ingest_mock_source(self):
        hd = Holodeck()
        src = MockPlatoSource(room_count=4, agent_count=10, seed=1)
        hd.ingest_mock_source(src)
        assert hd.room_count() == 4
        assert hd.agent_count() == 10
        assert hd.get_scene()["tick"] == 1

    def test_ingest_after_tick(self):
        hd = Holodeck()
        src = MockPlatoSource(room_count=2, agent_count=4, seed=2, movement_prob=0.5)
        hd.ingest_mock_source(src)
        src.tick()
        hd.ingest_mock_source(src)
        assert hd.get_scene()["tick"] == 2

    def test_tick_counts_match(self):
        hd = Holodeck()
        src = MockPlatoSource(room_count=3, agent_count=6, seed=3)
        for _ in range(5):
            src.tick()
            hd.ingest_mock_source(src)
        assert hd.get_scene()["tick"] == 5

    def test_phase_sync(self):
        hd = Holodeck()
        src = MockPlatoSource(room_count=2, agent_count=3, seed=4)
        hd.ingest_mock_source(src)
        for aid, astate in src.get_agent_states().items():
            assert hd.get_agent(aid).phase == astate["phase"]

    def test_trinity_sync(self):
        hd = Holodeck()
        src = MockPlatoSource(room_count=2, agent_count=3, seed=5)
        hd.ingest_mock_source(src)
        for aid, astate in src.get_agent_states().items():
            assert hd.get_agent(aid).trinity_scores == astate["trinity"]

    def test_connections_after_movement(self):
        hd = Holodeck()
        src = MockPlatoSource(room_count=3, agent_count=6, seed=6, movement_prob=1.0)
        hd.ingest_mock_source(src)
        src.tick()  # everyone moves
        hd.ingest_mock_source(src)
        scene = hd.get_scene()
        assert len(scene["connections"]) > 0


# ── Thread safety ──────────────────────────────────────


class TestThreadSafety:
    def test_concurrent_adds(self):
        hd = Holodeck()
        hd.add_room("r1", (0, 0, 0), capacity=1000)
        errors = []

        def add_many(n, prefix):
            try:
                for i in range(n):
                    hd.add_agent(f"{prefix}_t{i}", "r1")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=add_many, args=(50, f"th{i}")) for i in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert hd.agent_count() == 200

    def test_concurrent_moves(self):
        hd = Holodeck()
        hd.add_room("r1", (0, 0, 0), capacity=100)
        hd.add_room("r2", (0, 0, 0), capacity=100)
        for i in range(50):
            hd.add_agent(f"a{i}", "r1")
        errors = []

        def shuffle():
            try:
                for i in range(50):
                    hd.move_agent(f"a{i}", "r1", "r2")
                    hd.move_agent(f"a{i}", "r2", "r1")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=shuffle) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert hd.agent_count() == 50

    def test_concurrent_reads_and_writes(self):
        hd = Holodeck()
        for i in range(5):
            hd.add_room(f"r{i}", (float(i), 0, 0), capacity=50)
            for j in range(10):
                hd.add_agent(f"a{i}_{j}", f"r{i}")
        errors = []

        def reader():
            try:
                for _ in range(100):
                    _ = hd.get_scene()
                    _ = hd.snapshot()
            except Exception as e:
                errors.append(e)

        def writer():
            try:
                for _ in range(50):
                    hd.update_room("r0", occupancy=_rnd.randint(1, 10))
            except Exception as e:
                errors.append(e)

        import random as _rnd

        _rnd.seed(0)
        threads = [threading.Thread(target=reader) for _ in range(3)]
        threads += [threading.Thread(target=writer) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors

    def test_export_while_mutating(self):
        hd = Holodeck()
        hd.add_room("r1", (0, 0, 0), capacity=50)
        for i in range(20):
            hd.add_agent(f"a{i}", "r1")
        errors = []

        def mutator():
            try:
                for i in range(50):
                    hd.add_agent(f"x{i}", "r1")
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def exporter(tmp_path):
            try:
                for i in range(20):
                    hd.export_html(str(tmp_path / f"e{i}.html"))
                    time.sleep(0.002)
            except Exception as e:
                errors.append(e)

        from pathlib import Path as _P

        tmp = _P("/tmp/holodeck_thread_test")
        tmp.mkdir(exist_ok=True)
        t1 = threading.Thread(target=mutator)
        t2 = threading.Thread(target=exporter, args=(tmp,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert not errors
        # cleanup
        for f in tmp.iterdir():
            f.unlink()
        tmp.rmdir()


# ── Demo runner (smoke test) ───────────────────────────


class TestDemo:
    def test_demo_script_importable(self):
        # Just ensure the demo module parses and key symbols exist
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "holodeck_demo",
            Path(__file__).parent.parent / "examples" / "holodeck_demo.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "run_demo")

    def test_demo_runs(self, tmp_path):
        import importlib.util
        import sys

        demo_path = Path(__file__).parent.parent / "examples" / "holodeck_demo.py"
        spec = importlib.util.spec_from_file_location("holodeck_demo", demo_path)
        mod = importlib.util.module_from_spec(spec)

        # Override output path in demo if it hardcodes one
        # The demo writes to holodeck_demo.html — we'll just run it
        # and verify the file is created, then clean up.
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            spec.loader.exec_module(mod)
            mod.run_demo(output_dir=str(tmp_path))
            assert (tmp_path / "holodeck_demo.html").exists()
        finally:
            os.chdir(original_cwd)
