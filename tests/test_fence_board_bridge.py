"""tests/test_fence_board_bridge.py — Fence Board / Tom Sawyer Protocol tests."""

import pytest
from datetime import datetime, timedelta
from fleet.fence_board_bridge import FenceBoard, FenceStatus, Challenger, Fence


class TestFenceBoard:
    def test_post_fence(self):
        board = FenceBoard(max_active=5)
        fence = board.post_fence(
            title="Map Opcodes",
            brush="16 ops undefined",
            view="Name on every runtime",
            challengers={
                "Babel": (3, "Built the concept"),
                "Oracle1": (7, "Good at specs"),
            },
            reward="0x70-0x7F attributed",
        )
        assert fence.id == "fence-0x42"
        assert fence.title == "Map Opcodes"
        assert fence.status == FenceStatus.OPEN
        assert len(fence.challengers) == 2
        assert fence.claim_window_hours == 48

    def test_post_fence_default_window(self):
        board = FenceBoard(max_active=5)
        fence = board.post_fence(
            title="Test",
            brush="brush",
            view="view",
            challengers={"A": (5, "edge")},
            reward="reward",
        )
        assert fence.claim_window_hours == 48

    def test_max_active(self):
        board = FenceBoard(max_active=2)
        board.post_fence("A", "b", "v", {"A": (1, "e")}, "r")
        board.post_fence("B", "b", "v", {"A": (1, "e")}, "r")
        with pytest.raises(ValueError):
            board.post_fence("C", "b", "v", {"A": (1, "e")}, "r")

    def test_claim_fence(self):
        board = FenceBoard(max_active=5)
        fence = board.post_fence("T", "b", "v", {"A": (1, "e")}, "r")
        claimed = board.claim_fence(fence.id, "Agent1", "My approach")
        assert claimed.status == FenceStatus.CLAIMED
        assert claimed.claimed_by == "Agent1"
        assert claimed.claimed_approach == "My approach"
        assert claimed.claimed_at is not None

    def test_claim_not_open(self):
        board = FenceBoard(max_active=5)
        fence = board.post_fence("T", "b", "v", {"A": (1, "e")}, "r")
        board.claim_fence(fence.id, "Agent1", "approach")
        with pytest.raises(ValueError):
            board.claim_fence(fence.id, "Agent2", "approach2")

    def test_claim_unknown(self):
        board = FenceBoard(max_active=5)
        with pytest.raises(KeyError):
            board.claim_fence("fence-0x99", "Agent1", "approach")

    def test_complete_fence(self):
        board = FenceBoard(max_active=5)
        fence = board.post_fence("T", "b", "v", {"A": (1, "e")}, "r")
        board.claim_fence(fence.id, "Agent1", "approach")
        completed = board.complete_fence(fence.id, ["artifact.py"], "🥇 Gold")
        assert completed.status == FenceStatus.COMPLETED
        assert completed.badge == "🥇 Gold"
        assert completed.completed_artifacts == ["artifact.py"]
        assert completed.completed_at is not None

    def test_complete_not_claimed(self):
        board = FenceBoard(max_active=5)
        fence = board.post_fence("T", "b", "v", {"A": (1, "e")}, "r")
        with pytest.raises(ValueError):
            board.complete_fence(fence.id)

    def test_active_fences(self):
        board = FenceBoard(max_active=5)
        f1 = board.post_fence("A", "b", "v", {"A": (1, "e")}, "r")
        f2 = board.post_fence("B", "b", "v", {"A": (1, "e")}, "r")
        board.claim_fence(f2.id, "Agent1", "approach")
        active = board.active_fences()
        assert len(active) == 2

    def test_active_excludes_completed(self):
        board = FenceBoard(max_active=5)
        f1 = board.post_fence("A", "b", "v", {"A": (1, "e")}, "r")
        board.claim_fence(f1.id, "Agent1", "approach")
        board.complete_fence(f1.id)
        assert len(board.active_fences()) == 0
        assert len(board.completed_fences()) == 1

    def test_best_challenger(self):
        board = FenceBoard(max_active=5)
        fence = board.post_fence(
            "T",
            "b",
            "v",
            {
                "Babel": (3, "Built it"),
                "Oracle1": (7, "Specs"),
                "JC1": (4, "Hardware"),
            },
            "r",
        )
        assert board.best_challenger(fence.id) == "Babel"

    def test_best_challenger_unknown(self):
        board = FenceBoard(max_active=5)
        assert board.best_challenger("fence-0x99") is None

    def test_render_board(self):
        board = FenceBoard(max_active=5)
        f1 = board.post_fence("A", "b", "v", {"A": (1, "e")}, "r")
        board.claim_fence(f1.id, "Agent1", "approach")
        text = board.render_board()
        assert "FENCE BOARD" in text
        assert "🟡" in text
        assert "fence-0x42" in text
        assert "Agent1" in text

    def test_to_dict(self):
        board = FenceBoard(max_active=3)
        board.post_fence("A", "b", "v", {"A": (1, "e")}, "r")
        d = board.to_dict()
        assert d["max_active"] == 3
        assert "fence-0x42" in d["fences"]

    def test_serialization(self):
        board = FenceBoard(max_active=5)
        fence = board.post_fence(
            "T", "b", "v", {"A": (1, "e")}, "r", claim_window_hours=72
        )
        board.claim_fence(fence.id, "X", "approach")
        d = fence.to_dict()
        assert d["status"] == "CLAIMED"
        assert d["claimed_by"] == "X"
        assert d["claim_window_hours"] == 72
        assert d["posted_at"] is not None

    def test_challenger_edge_field(self):
        board = FenceBoard(max_active=5)
        fence = board.post_fence(
            "T", "b", "v", {"Babel": (3, "Built the concept")}, "r"
        )
        c = fence.challengers[0]
        assert c.name == "Babel"
        assert c.difficulty == 3
        assert c.edge == "Built the concept"
