"""Tests for CCC decision rubric."""
from fleet.ccc_decision_rubric import Input, decide, explain


class TestDecide:
    def test_blocker_tells_now(self):
        inp = Input("discussion5", "Push failed", "401", is_blocker=True)
        assert decide(inp) == "TELL_NOW"

    def test_breakthrough_tells_now(self):
        inp = Input("discussion5", "Speedup", "5x faster", is_breakthrough=True)
        assert decide(inp) == "TELL_NOW"

    def test_architecture_multi_repo_tells_now(self):
        inp = Input("discussion5", "Refactor", "Change bridge", is_architecture=True, affects_repos=3)
        assert decide(inp) == "TELL_NOW"

    def test_routine_ignored(self):
        inp = Input("discussion5", "Tick", ":45", is_routine_status=True)
        assert decide(inp) == "IGNORE"

    def test_zc_feed_logged(self):
        inp = Input("zc_feed", "Tile", "New thing")
        assert decide(inp) == "LOG"

    def test_health_ignored(self):
        inp = Input("health_check", "OK", "All green")
        assert decide(inp) == "IGNORE"

    def test_explain_returns_string(self):
        inp = Input("discussion5", "X", "Y", is_blocker=True)
        assert "TELL_NOW" in explain(inp)
