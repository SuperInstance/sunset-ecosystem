"""tests/test_fleet_task_board_bridge.py — Fleet Task Board tests."""

import pytest
from fleet.fleet_task_board_bridge import FleetTaskBoard, TaskPriority, TaskStatus


class TestFleetTaskBoard:
    def test_add_task(self):
        board = FleetTaskBoard()
        task = board.add_task(
            "Conformance", TaskPriority.CRITICAL, ["c", "python"], owner="JC1"
        )
        assert task.id == "task-1"
        assert task.title == "Conformance"
        assert task.priority == TaskPriority.CRITICAL
        assert task.tags == ["c", "python"]
        assert task.owner == "JC1"
        assert task.status == TaskStatus.OPEN

    def test_claim_task(self):
        board = FleetTaskBoard()
        t = board.add_task("X", TaskPriority.HIGH, ["rust"])
        claimed = board.claim_task(t.id, "Oracle1")
        assert claimed.status == TaskStatus.CLAIMED
        assert claimed.owner == "Oracle1"

    def test_claim_not_open(self):
        board = FleetTaskBoard()
        t = board.add_task("X", TaskPriority.HIGH, ["rust"])
        board.claim_task(t.id, "A")
        with pytest.raises(ValueError):
            board.claim_task(t.id, "B")

    def test_claim_unknown(self):
        board = FleetTaskBoard()
        with pytest.raises(KeyError):
            board.claim_task("task-99", "A")

    def test_set_eta(self):
        board = FleetTaskBoard()
        t = board.add_task("X", TaskPriority.HIGH, ["rust"])
        board.set_eta(t.id, "T-24h")
        assert board.tasks[t.id].eta == "T-24h"

    def test_complete_task(self):
        board = FleetTaskBoard()
        t = board.add_task("X", TaskPriority.HIGH, ["rust"])
        board.claim_task(t.id, "A")
        done = board.complete_task(t.id, commit_hash="abc123")
        assert done.status == TaskStatus.DONE
        assert done.commit_hash == "abc123"
        assert done.completed_at is not None

    def test_complete_not_done(self):
        board = FleetTaskBoard()
        t = board.add_task("X", TaskPriority.HIGH, ["rust"])
        done = board.complete_task(t.id)
        assert done.status == TaskStatus.DONE
        with pytest.raises(ValueError):
            board.complete_task(t.id)  # already done

    def test_by_priority(self):
        board = FleetTaskBoard()
        board.add_task("Low", TaskPriority.LOW, ["docs"])
        board.add_task("Critical", TaskPriority.CRITICAL, ["c"])
        board.add_task("High", TaskPriority.HIGH, ["python"])
        ordered = board.by_priority()
        assert ordered[0].priority == TaskPriority.CRITICAL
        assert ordered[1].priority == TaskPriority.HIGH
        assert ordered[2].priority == TaskPriority.LOW

    def test_by_owner(self):
        board = FleetTaskBoard()
        board.add_task("A", TaskPriority.HIGH, ["rust"], owner="JC1")
        board.add_task("B", TaskPriority.HIGH, ["rust"], owner="Oracle1")
        board.add_task("C", TaskPriority.HIGH, ["rust"])
        assert len(board.by_owner("JC1")) == 1
        assert board.by_owner("JC1")[0].title == "A"

    def test_critical_path(self):
        board = FleetTaskBoard()
        board.add_task("Done", TaskPriority.CRITICAL, ["c"])
        t2 = board.add_task("Open", TaskPriority.CRITICAL, ["c"])
        board.add_task("High", TaskPriority.HIGH, ["c"])
        board.complete_task(board.tasks["task-1"].id)
        path = board.critical_path()
        assert len(path) == 1
        assert path[0].id == t2.id

    def test_ready_tasks(self):
        board = FleetTaskBoard()
        t1 = board.add_task("A", TaskPriority.HIGH, ["rust"])
        t2 = board.add_task("B", TaskPriority.HIGH, ["rust"], blocked_by=t1.id)
        ready = board.ready_tasks()
        assert len(ready) == 1
        assert ready[0].id == t1.id

    def test_unblock(self):
        board = FleetTaskBoard()
        t1 = board.add_task("A", TaskPriority.HIGH, ["rust"])
        t2 = board.add_task("B", TaskPriority.HIGH, ["rust"], blocked_by=t1.id)
        board.unblock_task(t2.id)
        assert board.tasks[t2.id].blocked_by is None

    def test_render_text(self):
        board = FleetTaskBoard()
        t = board.add_task(
            "Conformance", TaskPriority.CRITICAL, ["c", "python"], owner="JC1"
        )
        board.set_eta(t.id, "T-24h")
        board.add_task("Dashboard", TaskPriority.HIGH, ["infra"], owner="Oracle1")
        text = board.render_text()
        assert "🔴" in text
        assert "🟠" in text
        assert "Conformance" in text
        assert "JC1" in text
        assert "T-24h" in text
        assert "[c]" in text

    def test_render_org_chart(self):
        board = FleetTaskBoard()
        board.add_task("A", TaskPriority.HIGH, ["rust"], owner="JC1")
        board.add_task("B", TaskPriority.HIGH, ["rust"], owner="Oracle1")
        chart = board.render_org_chart()
        assert "Captain Casey" in chart
        assert "JC1" in chart

    def test_to_dict(self):
        board = FleetTaskBoard()
        board.add_task("X", TaskPriority.HIGH, ["rust"])
        d = board.to_dict()
        assert "task-1" in d["tasks"]
        assert d["tasks"]["task-1"]["priority"] == "HIGH"

    def test_blocked_task_render(self):
        board = FleetTaskBoard()
        t1 = board.add_task("Dep", TaskPriority.HIGH, ["rust"])
        t2 = board.add_task("Blocked", TaskPriority.HIGH, ["rust"], blocked_by=t1.id)
        text = board.render_text()
        assert "BLOCKED" in text
        assert "Dep" in text

    def test_done_task_render(self):
        board = FleetTaskBoard()
        t = board.add_task("Done", TaskPriority.HIGH, ["rust"])
        board.claim_task(t.id, "A")
        board.complete_task(t.id, "abc123")
        text = board.render_text()
        assert "✓" in text
        assert "abc123" in text
