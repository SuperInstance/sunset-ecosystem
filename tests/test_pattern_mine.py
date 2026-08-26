"""Tests for PatternMine — operational pattern extraction and rule generation.

Reference: fleet/pattern_mine.py
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fleet.pattern_mine import AlertRule, OperationalPattern, PatternMine, TaskTemplate


class TestPatternLoading:
    def test_empty_miner(self) -> None:
        miner = PatternMine()
        assert miner.repo_path is None
        assert miner.patterns == []
        assert miner.rules == []
        assert miner.templates == []

    def test_load_defaults(self) -> None:
        miner = PatternMine()
        patterns = miner.load_patterns()
        assert len(patterns) > 0
        assert len(patterns) == len(miner.DEFAULT_PATTERNS)
        # Check critical patterns exist
        names = [p.name for p in patterns]
        assert "repo_sweep_batch_size" in names
        assert "silent_failure_detection" in names
        assert "a2a_handoff_contract" in names

    def test_load_from_missing_repo(self) -> None:
        miner = PatternMine(repo_path="/nonexistent/path")
        patterns = miner.load_patterns()
        # Should fall back to defaults
        assert len(patterns) == len(miner.DEFAULT_PATTERNS)

    def test_load_from_repo_with_files(self, tmp_path: Path) -> None:
        # Create mock repo structure
        repo = tmp_path / "agent-ops"
        (repo / "patterns").mkdir(parents=True)
        (repo / "patterns" / "repo-sweeps.md").write_text(
            "- **Rule:** Batch in groups of 5.\n- Verify output after each batch."
        )
        (repo / "docs").mkdir(parents=True)
        (repo / "docs" / "agent-reliability.md").write_text(
            "- **Rule:** 5 repos max per task.\n- Agents fail silently."
        )

        miner = PatternMine(repo_path=repo)
        patterns = miner.load_patterns()
        assert len(patterns) > 0
        # Should have extracted rules from markdown
        repo_rules = [p for p in patterns if p.source_file == "patterns/repo-sweeps.md"]
        assert len(repo_rules) >= 1

    def test_pattern_fields(self) -> None:
        miner = PatternMine()
        miner.load_patterns()
        p = next(x for x in miner.patterns if x.name == "repo_sweep_batch_size")
        assert p.category == "dispatch"
        assert p.severity == "warning"
        assert p.metric == "repos_per_task"
        assert p.threshold == 5.0
        assert p.action is not None


class TestRuleGeneration:
    def test_to_fleet_monitor_rules(self) -> None:
        miner = PatternMine()
        miner.load_patterns()
        rules = miner.to_fleet_monitor_rules()
        assert len(rules) > 0
        assert len(rules) == len(miner.patterns)

    def test_rule_fields(self) -> None:
        miner = PatternMine()
        miner.load_patterns()
        rules = miner.to_fleet_monitor_rules()
        r = next(x for x in rules if x.name == "repo_sweep_batch_size")
        assert r.component == "dispatch"
        assert r.level == "warning"
        assert "repos_per_task" in r.condition
        assert r.action is not None

    def test_critical_rule(self) -> None:
        miner = PatternMine()
        miner.load_patterns()
        rules = miner.to_fleet_monitor_rules()
        r = next(x for x in rules if x.name == "silent_failure_detection")
        assert r.level == "critical"
        assert "output_token_count" in r.condition

    def test_info_rule(self) -> None:
        miner = PatternMine()
        miner.load_patterns()
        rules = miner.to_fleet_monitor_rules()
        r = next(x for x in rules if x.name == "procedural_prompt_success")
        assert r.level == "info"
        assert "prompt_style_score" in r.condition

    def test_rule_message_template(self) -> None:
        miner = PatternMine()
        miner.load_patterns()
        rules = miner.to_fleet_monitor_rules()
        r = next(x for x in rules if x.name == "repo_sweep_batch_size")
        assert "5" in r.message_template
        assert "repos" in r.message_template.lower()

    def test_pattern_to_rule_metric_threshold(self) -> None:
        miner = PatternMine()
        pattern = OperationalPattern(
            name="test_metric",
            source_file="test.md",
            rule_text="Test rule",
            category="reliability",
            severity="warning",
            metric="repos_per_task",
            threshold=3.0,
            action="Split task",
        )
        rule = miner._pattern_to_rule(pattern)
        assert rule is not None
        assert rule.condition == "repos_per_task > 3.0"
        assert "Split task" in rule.message_template

    def test_pattern_to_rule_no_metric(self) -> None:
        miner = PatternMine()
        pattern = OperationalPattern(
            name="test_no_metric",
            source_file="test.md",
            rule_text="Always verify",
            category="reliability",
            severity="info",
        )
        rule = miner._pattern_to_rule(pattern)
        assert rule is not None
        assert rule.condition == "True"
        assert "Always verify" in rule.message_template


class TestTaskTemplates:
    def test_to_task_templates(self) -> None:
        miner = PatternMine()
        miner.load_patterns()
        templates = miner.to_task_templates()
        assert len(templates) > 0
        # Should have repo_sweep template
        names = [t.name for t in templates]
        assert "repo_sweep" in names

    def test_repo_sweep_template(self) -> None:
        miner = PatternMine()
        miner.load_patterns()
        templates = miner.to_task_templates()
        t = next(x for x in templates if x.name == "repo_sweep")
        assert t.max_repos == 5
        assert t.verify_output is True
        assert t.procedural is True
        assert "HANDOFF.md" in t.expected_deliverables

    def test_get_task_template(self) -> None:
        miner = PatternMine()
        miner.load_patterns()
        miner.to_task_templates()
        t = miner.get_task_template("repo_sweep")
        assert t is not None
        assert t.name == "repo_sweep"
        assert miner.get_task_template("nonexistent") is None

    def test_procedural_template(self) -> None:
        miner = PatternMine()
        miner.load_patterns()
        templates = miner.to_task_templates()
        t = next((x for x in templates if x.name == "procedural_task"), None)
        if t:
            assert t.procedural is True
            assert "style" in t.instructions.lower()


class TestReports:
    def test_generate_report(self) -> None:
        miner = PatternMine()
        report = miner.generate_report()
        assert report["patterns_loaded"] > 0
        assert report["rules_generated"] > 0
        assert "categories" in report
        assert "critical_patterns" in report
        assert "warning_patterns" in report
        assert "info_patterns" in report
        assert "top_recommendations" in report

    def test_categorize_patterns(self) -> None:
        miner = PatternMine()
        miner.load_patterns()
        cats = miner._categorize_patterns()
        assert "dispatch" in cats
        assert cats["dispatch"] >= 2  # repo_sweep and max_repos

    def test_top_recommendations(self) -> None:
        miner = PatternMine()
        miner.load_patterns()
        recs = miner._top_recommendations()
        assert len(recs) > 0
        # Should include critical patterns
        assert any("CRITICAL" in r for r in recs)
        # Should include warning patterns with metrics
        assert any("WARNING" in r for r in recs)

    def test_write_report(self, tmp_path: Path) -> None:
        miner = PatternMine()
        miner.load_patterns()
        miner.to_fleet_monitor_rules()
        miner.to_task_templates()
        report_path = tmp_path / "report.md"
        result = miner.write_report(path=report_path)
        assert result == report_path
        assert report_path.exists()
        content = report_path.read_text()
        assert "PatternMine Report" in content
        assert "repo_sweep_batch_size" in content
        assert "repo_sweep" in content

    def test_critical_patterns_in_report(self, tmp_path: Path) -> None:
        miner = PatternMine()
        report_path = tmp_path / "report.md"
        miner.write_report(path=report_path)
        content = report_path.read_text()
        # Should mention critical patterns
        assert (
            "silent_failure_detection" in content or "a2a_handoff_contract" in content
        )


class TestApplyToMonitor:
    def test_apply_to_monitor(self) -> None:
        miner = PatternMine()
        miner.load_patterns()
        # Use a mock monitor (just a dict)
        monitor = {}
        added = miner.apply_to_monitor(monitor)
        assert len(added) > 0
        assert "repo_sweep_batch_size" in added
        assert "silent_failure_detection" in added

    def test_apply_logs_rules(self) -> None:
        miner = PatternMine()
        miner.load_patterns()
        added = miner.apply_to_monitor({})
        # All patterns should have been converted
        assert len(added) == len(miner.patterns)
