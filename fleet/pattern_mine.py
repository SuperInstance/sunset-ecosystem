"""PatternMine — Extract operational patterns from agent-operations and apply to fleet monitoring.

Reads pattern files from the agent-operations repo (or local clone) and converts
hard-won rules into FleetMonitor alert conditions and task dispatch templates.

Patterns Mined
-------------
- **Repo Sweep Pattern**: "Batch in groups of 5. Verify output."
- **Task Prompt Pattern**: "Procedural prompts succeed at 90%+. Style guides kill agents."
- **A2A Handoff Pattern**: "HANDOFF.md is the contract. Zero-token output = silent failure."
- **Reliability Pattern**: "5 repos per task max. Separate task from style."

Usage
-----
    miner = PatternMine(repo_path="/path/to/agent-operations")
    miner.load_patterns()
    
    # Convert to FleetMonitor rules
    rules = miner.to_fleet_monitor_rules()
    for rule in rules:
        monitor.add_rule(rule)
    
    # Generate task templates
    template = miner.get_task_template("repo_sweep")
"""

from __future__ import annotations

__all__ = [
    "PatternMine",
    "OperationalPattern",
    "AlertRule",
    "TaskTemplate",
]

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class OperationalPattern:
    """A mined operational pattern with metadata."""
    name: str
    source_file: str
    rule_text: str
    category: str  # reliability, dispatch, coordination, style
    severity: str  # info, warning, critical
    metric: str | None = None  # What to measure
    threshold: float | None = None  # Alert threshold
    action: str | None = None  # Recommended action


@dataclass
class AlertRule:
    """A FleetMonitor-compatible alert rule."""
    name: str
    component: str
    condition: str  # Python expression as string
    level: str  # info, warning, critical
    message_template: str
    action: str | None = None


@dataclass
class TaskTemplate:
    """A subagent task template derived from a pattern."""
    name: str
    pattern_name: str
    instructions: str
    max_repos: int | None = None
    verify_output: bool = False
    procedural: bool = True
    expected_deliverables: list[str] = field(default_factory=list)


class PatternMine:
    """Mine operational patterns from agent-operations repo.

    Parameters
    ----------
    repo_path : Path | str | None
        Path to local clone of agent-operations repo.
        If None, uses default patterns (hard-coded from analysis).
    """

    # Built-in patterns extracted from agent-operations analysis
    DEFAULT_PATTERNS: list[dict[str, Any]] = [
        {
            "name": "repo_sweep_batch_size",
            "source_file": "patterns/repo-sweeps.md",
            "rule_text": "Batch in groups of 5. Verify output (agents fail silently). Use handoff files between waves.",
            "category": "dispatch",
            "severity": "warning",
            "metric": "repos_per_task",
            "threshold": 5.0,
            "action": "Split task into multiple waves of 5 repos each. Write HANDOFF.md between waves.",
        },
        {
            "name": "procedural_prompt_success",
            "source_file": "patterns/task-prompts.md",
            "rule_text": "Procedural prompts (do X, then Y) succeed at 90%+. Style guides kill agents. Separate task from style.",
            "category": "style",
            "severity": "info",
            "metric": "prompt_style_score",
            "threshold": 0.9,
            "action": "Rewrite prompt to be procedural: step 1, step 2, step 3. Remove style requirements.",
        },
        {
            "name": "a2a_handoff_contract",
            "source_file": "a2a-protocol/README.md",
            "rule_text": "HANDOFF.md is the contract. Zero-token output = silent failure. Different models for different task types.",
            "category": "coordination",
            "severity": "critical",
            "metric": "handoff_compliance",
            "threshold": 1.0,
            "action": "Abort task and request handoff file. Check model assignment matches task type.",
        },
        {
            "name": "silent_failure_detection",
            "source_file": "docs/agent-reliability.md",
            "rule_text": "Zero-token output = silent failure. Agents fail silently — always verify output.",
            "category": "reliability",
            "severity": "critical",
            "metric": "output_token_count",
            "threshold": 1.0,
            "action": "Mark task as failed. Require non-empty output with specific deliverables.",
        },
        {
            "name": "task_style_separation",
            "source_file": "patterns/task-prompts.md",
            "rule_text": "Separate task from style. Be procedural, not descriptive. Reference files, don't inline them.",
            "category": "style",
            "severity": "warning",
            "metric": "style_in_task_ratio",
            "threshold": 0.2,
            "action": "Extract style requirements into separate STYLE.md file. Keep task instructions procedural.",
        },
        {
            "name": "model_type_matching",
            "source_file": "a2a-protocol/README.md",
            "rule_text": "Different models for different task types. Auditors need deep reasoning. Coders need tool use. Writers need creativity.",
            "category": "coordination",
            "severity": "warning",
            "metric": "model_task_match_score",
            "threshold": 0.8,
            "action": "Reassign task to model with appropriate capabilities. Document model assignment rationale.",
        },
        {
            "name": "verification_required",
            "source_file": "patterns/repo-sweeps.md",
            "rule_text": "Verify output (agents fail silently). Check that files actually exist and contain expected content.",
            "category": "reliability",
            "severity": "warning",
            "metric": "verification_pass_rate",
            "threshold": 1.0,
            "action": "Run verification script on output. Check file existence and content matches deliverables.",
        },
        {
            "name": "max_repos_per_task",
            "source_file": "docs/agent-reliability.md",
            "rule_text": "5 repos per task max. More than 5 and agents get overwhelmed and produce garbage.",
            "category": "dispatch",
            "severity": "warning",
            "metric": "repos_per_task",
            "threshold": 5.0,
            "action": "Split into multiple tasks. Use conductor to orchestrate parallel waves.",
        },
    ]

    def __init__(self, repo_path: Path | str | None = None) -> None:
        self.repo_path = Path(repo_path) if repo_path else None
        self.patterns: list[OperationalPattern] = []
        self.rules: list[AlertRule] = []
        self.templates: list[TaskTemplate] = []

    # ── Pattern Loading ─────────────────────────────────────

    def load_patterns(self) -> list[OperationalPattern]:
        """Load patterns from repo or use defaults."""
        if self.repo_path and self.repo_path.exists():
            self._load_from_repo()
        else:
            self._load_defaults()
        return self.patterns

    def _load_defaults(self) -> None:
        """Load built-in patterns."""
        for p in self.DEFAULT_PATTERNS:
            self.patterns.append(OperationalPattern(**p))
        logger.info("Loaded %d default patterns", len(self.patterns))

    def _load_from_repo(self) -> None:
        """Parse markdown files in agent-operations repo."""
        pattern_files = [
            ("patterns/repo-sweeps.md", "dispatch", "warning"),
            ("patterns/task-prompts.md", "style", "info"),
            ("a2a-protocol/README.md", "coordination", "critical"),
            ("docs/agent-reliability.md", "reliability", "warning"),
        ]

        for rel_path, category, default_severity in pattern_files:
            full_path = self.repo_path / rel_path
            if not full_path.exists():
                continue

            content = full_path.read_text()
            # Extract rules: lines with "- **Rule:**" or numbered rules
            rules = self._extract_rules(content)
            for i, rule_text in enumerate(rules):
                self.patterns.append(OperationalPattern(
                    name=f"{full_path.stem}_rule_{i+1}",
                    source_file=rel_path,
                    rule_text=rule_text,
                    category=category,
                    severity=default_severity,
                ))

        # If no patterns found from repo, fall back to defaults
        if not self.patterns:
            self._load_defaults()
        else:
            logger.info("Loaded %d patterns from repo", len(self.patterns))

    def _extract_rules(self, content: str) -> list[str]:
        """Extract rule sentences from markdown content."""
        rules = []
        # Look for bold rule markers, bullet points with key phrases, or numbered lists
        lines = content.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("- **") and ("Rule" in line or "Max" in line or "Always" in line or "Never" in line):
                rules.append(re.sub(r"^-\s*\*\*.*?\*\*\s*", "", line))
            elif line.startswith("- ") and len(line) > 20 and any(kw in line for kw in ["max", "limit", "batch", "verify", "fail", "silent", "procedural"]):
                rules.append(line[2:])
        return rules

    # ── Rule Conversion ─────────────────────────────────────

    def to_fleet_monitor_rules(self) -> list[AlertRule]:
        """Convert patterns to FleetMonitor alert rules."""
        if not self.patterns:
            self.load_patterns()

        self.rules = []
        for p in self.patterns:
            rule = self._pattern_to_rule(p)
            if rule:
                self.rules.append(rule)

        return self.rules

    def _pattern_to_rule(self, pattern: OperationalPattern) -> AlertRule | None:
        """Convert a single pattern to an alert rule."""
        # Map severity to FleetMonitor level
        level_map = {"info": "info", "warning": "warning", "critical": "critical"}
        level = level_map.get(pattern.severity, "info")

        # Build condition based on metric
        if pattern.metric == "repos_per_task" and pattern.threshold:
            condition = f"repos_per_task > {pattern.threshold}"
            msg = f"Task has {{repos_per_task}} repos — exceeds {pattern.threshold} max. {pattern.action}"
        elif pattern.metric == "output_token_count" and pattern.threshold:
            condition = f"output_token_count < {pattern.threshold}"
            msg = f"Zero-token output detected — silent failure. {pattern.action}"
        elif pattern.metric == "prompt_style_score" and pattern.threshold:
            condition = f"prompt_style_score < {pattern.threshold}"
            msg = f"Prompt style score {{prompt_style_score:.0%}} below {pattern.threshold:.0%}. {pattern.action}"
        elif pattern.metric == "handoff_compliance" and pattern.threshold:
            condition = f"handoff_compliance < {pattern.threshold}"
            msg = f"Handoff compliance {{handoff_compliance:.0%}} — contract breach. {pattern.action}"
        elif pattern.metric == "verification_pass_rate" and pattern.threshold:
            condition = f"verification_pass_rate < {pattern.threshold}"
            msg = f"Verification pass rate {{verification_pass_rate:.0%}} — output may be invalid. {pattern.action}"
        elif pattern.metric == "model_task_match_score" and pattern.threshold:
            condition = f"model_task_match_score < {pattern.threshold}"
            msg = f"Model-task mismatch {{model_task_match_score:.0%}}. {pattern.action}"
        elif pattern.metric == "style_in_task_ratio" and pattern.threshold:
            condition = f"style_in_task_ratio > {pattern.threshold}"
            msg = f"Style content {{style_in_task_ratio:.0%}} exceeds task. {pattern.action}"
        else:
            # Generic rule without metric
            condition = "True"  # Always evaluate, manual inspection
            msg = f"Pattern '{pattern.name}': {pattern.rule_text}"

        return AlertRule(
            name=pattern.name,
            component=pattern.category,
            condition=condition,
            level=level,
            message_template=msg,
            action=pattern.action,
        )

    # ── Task Templates ──────────────────────────────────────

    def to_task_templates(self) -> list[TaskTemplate]:
        """Generate task templates from dispatch patterns."""
        if not self.patterns:
            self.load_patterns()

        self.templates = []
        for p in self.patterns:
            if p.category == "dispatch":
                template = self._pattern_to_template(p)
                if template:
                    self.templates.append(template)

        return self.templates

    def _pattern_to_template(self, pattern: OperationalPattern) -> TaskTemplate | None:
        """Convert a dispatch pattern to a task template."""
        text = pattern.rule_text.lower()
        name = pattern.name.lower()
        if "repo" in text or "sweep" in name or "batch" in text:
            return TaskTemplate(
                name="repo_sweep",
                pattern_name=pattern.name,
                instructions="Sweep repositories in batches of 5. Verify output after each batch. Write HANDOFF.md between waves.",
                max_repos=5,
                verify_output=True,
                procedural=True,
                expected_deliverables=["HANDOFF.md", "audit_report.md"],
            )
        elif "task" in text or "procedural" in text or "prompt" in name:
            return TaskTemplate(
                name="procedural_task",
                pattern_name=pattern.name,
                instructions="Write procedural prompts: step 1, step 2, step 3. Separate style from task. Reference files, don't inline.",
                max_repos=None,
                verify_output=True,
                procedural=True,
                expected_deliverables=["task_prompt.md", "style_guide.md"],
            )
        return None

    def get_task_template(self, name: str) -> TaskTemplate | None:
        """Get a specific task template by name."""
        if not self.templates:
            self.to_task_templates()
        for t in self.templates:
            if t.name == name:
                return t
        return None

    # ── Reports ─────────────────────────────────────────────

    def generate_report(self) -> dict[str, Any]:
        """Generate a comprehensive pattern mining report."""
        if not self.patterns:
            self.load_patterns()
        if not self.rules:
            self.to_fleet_monitor_rules()
        if not self.templates:
            self.to_task_templates()

        return {
            "patterns_loaded": len(self.patterns),
            "rules_generated": len(self.rules),
            "templates_generated": len(self.templates),
            "categories": self._categorize_patterns(),
            "critical_patterns": [p.name for p in self.patterns if p.severity == "critical"],
            "warning_patterns": [p.name for p in self.patterns if p.severity == "warning"],
            "info_patterns": [p.name for p in self.patterns if p.severity == "info"],
            "top_recommendations": self._top_recommendations(),
        }

    def _categorize_patterns(self) -> dict[str, int]:
        """Count patterns by category."""
        counts: dict[str, int] = {}
        for p in self.patterns:
            counts[p.category] = counts.get(p.category, 0) + 1
        return counts

    def _top_recommendations(self) -> list[str]:
        """Generate top recommendations from patterns."""
        recs = []
        for p in self.patterns:
            if p.severity == "critical":
                recs.append(f"[CRITICAL] {p.name}: {p.action}")
            elif p.severity == "warning" and p.metric:
                recs.append(f"[WARNING] {p.name}: Monitor {p.metric} > {p.threshold}")
        return recs

    def write_report(self, path: Path | None = None) -> Path:
        """Write pattern mining report to markdown."""
        report = self.generate_report()
        path = path or Path("docs/PATTERN_MINE_REPORT.md")

        lines = [
            "# PatternMine Report",
            "",
            f"**Patterns Loaded:** {report['patterns_loaded']}",
            f"**Rules Generated:** {report['rules_generated']}",
            f"**Templates Generated:** {report['templates_generated']}",
            "",
            "## Categories",
            "",
        ]
        for cat, count in report["categories"].items():
            lines.append(f"- **{cat}:** {count}")
        lines.append("")
        lines.append("## Critical Patterns")
        lines.append("")
        for name in report["critical_patterns"]:
            p = next((x for x in self.patterns if x.name == name), None)
            if p:
                lines.append(f"### {p.name}")
                lines.append(f"- **Rule:** {p.rule_text}")
                lines.append(f"- **Action:** {p.action}")
                lines.append("")
        lines.append("## Top Recommendations")
        lines.append("")
        for rec in report["top_recommendations"]:
            lines.append(f"- {rec}")
        lines.append("")
        lines.append("## Task Templates")
        lines.append("")
        for t in self.templates:
            lines.append(f"### {t.name}")
            lines.append(f"- **Pattern:** {t.pattern_name}")
            lines.append(f"- **Instructions:** {t.instructions}")
            lines.append(f"- **Max Repos:** {t.max_repos}")
            lines.append(f"- **Verify Output:** {t.verify_output}")
            lines.append(f"- **Deliverables:** {', '.join(t.expected_deliverables)}")
            lines.append("")

        with open(path, "w") as f:
            f.write("\n".join(lines))

        logger.info("Wrote pattern report to %s", path)
        return path

    # ── Integration with FleetMonitor ───────────────────────

    def apply_to_monitor(self, monitor: Any) -> list[str]:
        """Apply mined rules to a FleetMonitor instance.

        Returns list of rule names that were added.
        """
        if not self.rules:
            self.to_fleet_monitor_rules()

        added: list[str] = []
        for rule in self.rules:
            # FleetMonitor doesn't have a formal add_rule API yet,
            # but we can document what rules should be added
            added.append(rule.name)
            logger.info("Rule '%s' ready for FleetMonitor: %s", rule.name, rule.condition)

        return added
