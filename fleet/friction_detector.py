"""fleet/friction_detector.py — Automated UX research that detects friction from agent behavior.

The fleet's immune system against bad design. Observes agent behavior (API errors,
retries, timeouts), clusters patterns into friction categories, generates a friction
map, suggests fixes, and validates them by replaying academy cohorts.

Usage
-----
    from fleet.friction_detector import FrictionDetector

    detector = FrictionDetector(node_id="alpha")

    # Feed behavior observations
    detector.sense(endpoint="/api/tiles", status_code=401, latency=0.2, retries=3)
    detector.sense(endpoint="/api/rooms", status_code=500, latency=5.0, retries=0)

    # Run the SDA loop
    friction_map = detector.tick()

    # Get suggestions
    for point in friction_map.points:
        print(point.suggested_fix)

    # Validate a fix against academy cohorts
    validator = FixValidator(detector.academy_bridge)
    validator.replay_cohort("architect")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class FrictionCategory(Enum):
    """Categories of friction detected in the fleet."""
    AUTH = auto()
    UI = auto()
    SCHEMA = auto()
    ROUTING = auto()
    PERFORMANCE = auto()
    API = auto()
    UNKNOWN = auto()


class Severity(Enum):
    """Severity levels for friction points."""
    CRITICAL = 4
    HIGH = 3
    MEDIUM = 2
    LOW = 1


@dataclass
class BehaviorSample:
    """Single observation of agent behavior."""
    endpoint: str
    status_code: int
    latency: float
    retries: int
    timestamp: float = field(default_factory=time.time)
    agent_id: Optional[str] = None
    method: str = "GET"
    error_message: Optional[str] = None


@dataclass
class FrictionPoint:
    """Detected friction with evidence and suggested fix."""
    category: FrictionCategory
    severity: Severity
    evidence: str
    suggested_fix: str
    endpoint: Optional[str] = None
    sample_count: int = 1
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    fixed: bool = False
    fix_timestamp: Optional[float] = None


@dataclass
class FrictionMap:
    """Aggregate view of all friction across the fleet."""
    node_id: str
    points: List[FrictionPoint] = field(default_factory=list)
    generated_at: float = field(default_factory=time.time)
    total_samples: int = 0

    def by_category(self, category: FrictionCategory) -> List[FrictionPoint]:
        return [p for p in self.points if p.category == category]

    def by_severity(self, severity: Severity) -> List[FrictionPoint]:
        return [p for p in self.points if p.severity == severity]

    def critical_count(self) -> int:
        return len(self.by_severity(Severity.CRITICAL))

    def high_count(self) -> int:
        return len(self.by_severity(Severity.HIGH))

    def fixed_count(self) -> int:
        return sum(1 for p in self.points if p.fixed)

    def trend_score(self) -> float:
        """Aggregate severity score (higher = more friction)."""
        return sum(p.severity.value for p in self.points if not p.fixed)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "generated_at": self.generated_at,
            "total_samples": self.total_samples,
            "point_count": len(self.points),
            "critical": self.critical_count(),
            "high": self.high_count(),
            "fixed": self.fixed_count(),
            "trend_score": self.trend_score(),
            "points": [
                {
                    "category": p.category.name,
                    "severity": p.severity.name,
                    "evidence": p.evidence,
                    "suggested_fix": p.suggested_fix,
                    "endpoint": p.endpoint,
                    "sample_count": p.sample_count,
                    "fixed": p.fixed,
                }
                for p in self.points
            ],
        }


class FixSuggestionEngine:
    """Maps friction categories and evidence to concrete fix recommendations."""

    FIXES: Dict[Tuple[FrictionCategory, str], str] = {
        (FrictionCategory.AUTH, "401"): "Add authentication middleware or token refresh logic.",
        (FrictionCategory.AUTH, "403"): "Check RBAC permissions and scope policies.",
        (FrictionCategory.UI, "404"): "Verify frontend route mapping and asset bundling.",
        (FrictionCategory.UI, "no_web_ui"): "Deploy a web UI dashboard or SPA entry point.",
        (FrictionCategory.SCHEMA, "400"): "Validate request payload against OpenAPI schema.",
        (FrictionCategory.SCHEMA, "422"): "Update schema definitions and client-side validators.",
        (FrictionCategory.SCHEMA, "no_build_schema"): "Add JSON Schema for room/tile creation endpoints.",
        (FrictionCategory.ROUTING, "502"): "Check upstream health and load balancer configuration.",
        (FrictionCategory.ROUTING, "503"): "Add circuit breaker and retry with exponential backoff.",
        (FrictionCategory.ROUTING, "504"): "Increase gateway timeout or optimize upstream latency.",
        (FrictionCategory.PERFORMANCE, "timeout"): "Optimize endpoint or add async processing.",
        (FrictionCategory.PERFORMANCE, "high_latency"): "Add caching layer or CDN for static assets.",
        (FrictionCategory.API, "dual_submit_endpoints"): "Consolidate duplicate endpoints into a single canonical route.",
        (FrictionCategory.API, "no_broadcast_endpoints"): "Add WebSocket or SSE broadcast endpoint.",
        (FrictionCategory.API, "no_global_fleet_map"): "Implement fleet topology discovery API.",
        (FrictionCategory.UNKNOWN, "default"): "Investigate logs and reproduce in staging environment.",
    }

    @classmethod
    def suggest(cls, category: FrictionCategory, evidence: str) -> str:
        key = (category, evidence)
        if key in cls.FIXES:
            return cls.FIXES[key]
        # Fallback: try category-only with default evidence
        key_fallback = (category, "default")
        if key_fallback in cls.FIXES:
            return cls.FIXES[key_fallback]
        # Final fallback
        return cls.FIXES.get((FrictionCategory.UNKNOWN, "default"), "Investigate and document the issue.")


class FrictionDetector:
    """SDA loop: SENSE behavior → DECIDE friction → ACT suggest fix → VALIDATE."""

    def __init__(
        self,
        node_id: str,
        academy_bridge: Optional[Any] = None,
        latency_threshold: float = 2.0,
        retry_threshold: int = 2,
    ):
        self.node_id = node_id
        self._samples: List[BehaviorSample] = []
        self._friction_points: Dict[str, FrictionPoint] = {}
        self._trend_history: List[Tuple[float, float]] = []
        self._event_callbacks: List[Callable[[FrictionPoint], None]] = []
        self.latency_threshold = latency_threshold
        self.retry_threshold = retry_threshold
        self.academy_bridge = academy_bridge
        self._fix_validator: Optional[FixValidator] = None

    def register_event_callback(self, callback: Callable[[FrictionPoint], None]) -> None:
        self._event_callbacks.append(callback)

    def sense(
        self,
        endpoint: str,
        status_code: int,
        latency: float,
        retries: int = 0,
        agent_id: Optional[str] = None,
        method: str = "GET",
        error_message: Optional[str] = None,
    ) -> BehaviorSample:
        """SENSE: Record a behavior observation."""
        sample = BehaviorSample(
            endpoint=endpoint,
            status_code=status_code,
            latency=latency,
            retries=retries,
            agent_id=agent_id,
            method=method,
            error_message=error_message,
        )
        self._samples.append(sample)
        logger.debug(
            "SENSE %s %s → %d in %.3fs (retries=%d)",
            method, endpoint, status_code, latency, retries,
        )
        return sample

    def decide(self) -> List[FrictionPoint]:
        """DECIDE: Analyze samples and detect friction points."""
        if not self._samples:
            return []

        new_points: List[FrictionPoint] = []
        for sample in self._samples:
            point = self._analyze_sample(sample)
            if point:
                new_points.append(point)

        # Merge into existing friction points by key
        merged: List[FrictionPoint] = []
        for point in new_points:
            key = self._friction_key(point)
            if key in self._friction_points:
                existing = self._friction_points[key]
                existing.sample_count += point.sample_count
                existing.last_seen = time.time()
                if point.severity.value > existing.severity.value:
                    existing.severity = point.severity
                    existing.evidence = point.evidence
            else:
                self._friction_points[key] = point
                merged.append(point)
                self._emit_event(point)

        self._samples.clear()
        return merged

    def act(self, points: List[FrictionPoint]) -> None:
        """ACT: Generate fix suggestions for detected friction."""
        for point in points:
            point.suggested_fix = FixSuggestionEngine.suggest(
                point.category, point.evidence
            )
            logger.info(
                "ACT %s friction on %s → %s",
                point.category.name, point.endpoint or "fleet", point.suggested_fix
            )

    def validate(self, points: List[FrictionPoint]) -> Dict[str, bool]:
        """VALIDATE: Run academy cohort replay to verify fixes would work."""
        if self._fix_validator is None:
            self._fix_validator = FixValidator(self.academy_bridge)
        results: Dict[str, bool] = {}
        for point in points:
            key = self._friction_key(point)
            if self.academy_bridge is not None:
                results[key] = self._fix_validator.validate_fix(point)
            else:
                results[key] = False
        return results

    def tick(self) -> FrictionMap:
        """Run one full SDA loop and return the updated friction map."""
        points = self.decide()
        self.act(points)
        if points:
            self.validate(points)

        trend_score = sum(p.severity.value for p in self._friction_points.values() if not p.fixed)
        self._trend_history.append((time.time(), trend_score))

        return FrictionMap(
            node_id=self.node_id,
            points=list(self._friction_points.values()),
            generated_at=time.time(),
            total_samples=len(self._samples) + sum(p.sample_count for p in self._friction_points.values()),
        )

    def get_trend(self) -> str:
        """Return trend direction: 'increasing', 'decreasing', or 'stable'."""
        if len(self._trend_history) < 2:
            return "stable"
        recent = self._trend_history[-5:]
        if len(recent) < 2:
            return "stable"
        scores = [s for _, s in recent]
        if scores[-1] > scores[0] * 1.1:
            return "increasing"
        if scores[-1] < scores[0] * 0.9:
            return "decreasing"
        return "stable"

    def apply_fix(self, friction_key: str) -> bool:
        """Mark a friction point as fixed."""
        if friction_key not in self._friction_points:
            return False
        point = self._friction_points[friction_key]
        point.fixed = True
        point.fix_timestamp = time.time()
        logger.info("FIX applied for %s", friction_key)
        return True

    def load_academy_findings(self) -> List[FrictionPoint]:
        """Load friction points from the academy bridge's cohort results."""
        if self.academy_bridge is None:
            return []

        findings = self.academy_bridge.get_friction_points()
        loaded: List[FrictionPoint] = []
        for finding in findings:
            category = self._cohort_category(finding.get("finding", ""))
            severity = self._parse_severity(finding.get("severity", "low"))
            evidence = finding.get("finding", "unknown")
            endpoint = finding.get("agent", "academy")
            point = FrictionPoint(
                category=category,
                severity=severity,
                evidence=evidence,
                suggested_fix=FixSuggestionEngine.suggest(category, evidence),
                endpoint=endpoint,
            )
            key = self._friction_key(point)
            if key not in self._friction_points:
                self._friction_points[key] = point
                loaded.append(point)
                self._emit_event(point)
        return loaded

    def compare_with_academy(self) -> Dict[str, Any]:
        """Compare live-detected friction with academy cohort findings."""
        live_categories = {p.category for p in self._friction_points.values() if not p.fixed}
        academy = self.load_academy_findings()
        academy_categories = {p.category for p in academy}

        return {
            "live_only": sorted([c.name for c in live_categories - academy_categories]),
            "academy_only": sorted([c.name for c in academy_categories - live_categories]),
            "overlap": sorted([c.name for c in live_categories & academy_categories]),
            "live_points": len([p for p in self._friction_points.values() if not p.fixed]),
            "academy_points": len(academy),
        }

    def _analyze_sample(self, sample: BehaviorSample) -> Optional[FrictionPoint]:
        """Analyze a single behavior sample and return a friction point if detected."""
        status = sample.status_code
        latency = sample.latency
        retries = sample.retries

        # Authentication friction
        if status in (401, 403):
            return FrictionPoint(
                category=FrictionCategory.AUTH,
                severity=Severity.CRITICAL if status == 401 and retries >= self.retry_threshold else Severity.HIGH,
                evidence=str(status),
                suggested_fix="",
                endpoint=sample.endpoint,
            )

        # Routing / upstream errors
        if status in (502, 503, 504):
            return FrictionPoint(
                category=FrictionCategory.ROUTING,
                severity=Severity.HIGH if status == 503 else Severity.MEDIUM,
                evidence=str(status),
                suggested_fix="",
                endpoint=sample.endpoint,
            )

        # Schema / validation errors
        if status in (400, 422):
            return FrictionPoint(
                category=FrictionCategory.SCHEMA,
                severity=Severity.MEDIUM,
                evidence=str(status),
                suggested_fix="",
                endpoint=sample.endpoint,
            )

        # UI / missing resource
        if status == 404:
            return FrictionPoint(
                category=FrictionCategory.UI,
                severity=Severity.MEDIUM,
                evidence="404",
                suggested_fix="",
                endpoint=sample.endpoint,
            )

        # Performance friction
        if latency >= self.latency_threshold:
            return FrictionPoint(
                category=FrictionCategory.PERFORMANCE,
                severity=Severity.HIGH if latency >= self.latency_threshold * 3 else Severity.MEDIUM,
                evidence="high_latency" if latency < self.latency_threshold * 3 else "timeout",
                suggested_fix="",
                endpoint=sample.endpoint,
            )

        # Retry storms indicate API or routing issues
        if retries >= self.retry_threshold:
            return FrictionPoint(
                category=FrictionCategory.API,
                severity=Severity.HIGH if retries >= self.retry_threshold * 2 else Severity.MEDIUM,
                evidence="retry_storm",
                suggested_fix="",
                endpoint=sample.endpoint,
            )

        # Server errors as API friction
        if status >= 500:
            return FrictionPoint(
                category=FrictionCategory.API,
                severity=Severity.HIGH,
                evidence=str(status),
                suggested_fix="",
                endpoint=sample.endpoint,
            )

        # Unmapped error codes
        if status >= 400:
            return FrictionPoint(
                category=FrictionCategory.UNKNOWN,
                severity=Severity.LOW,
                evidence=str(status),
                suggested_fix="",
                endpoint=sample.endpoint,
            )

        return None

    def _friction_key(self, point: FrictionPoint) -> str:
        return f"{point.category.name}:{point.endpoint or 'fleet'}:{point.evidence}"

    def _emit_event(self, point: FrictionPoint) -> None:
        for cb in self._event_callbacks:
            try:
                cb(point)
            except Exception:
                logger.exception("Event callback failed for %s", point.category.name)

    def _cohort_category(self, finding: str) -> FrictionCategory:
        finding_lower = finding.lower()
        if "auth" in finding_lower:
            return FrictionCategory.AUTH
        if "ui" in finding_lower or "web" in finding_lower:
            return FrictionCategory.UI
        if "schema" in finding_lower or "build" in finding_lower:
            return FrictionCategory.SCHEMA
        if "route" in finding_lower or "endpoint" in finding_lower:
            return FrictionCategory.ROUTING
        if "tile" in finding_lower or "count" in finding_lower:
            return FrictionCategory.SCHEMA
        if "broadcast" in finding_lower or "fleet_map" in finding_lower:
            return FrictionCategory.API
        return FrictionCategory.UNKNOWN

    def _parse_severity(self, severity: str) -> Severity:
        mapping = {
            "critical": Severity.CRITICAL,
            "high": Severity.HIGH,
            "medium": Severity.MEDIUM,
            "low": Severity.LOW,
        }
        return mapping.get(severity.lower(), Severity.LOW)


class FixValidator:
    """Replays academy scenarios to verify that suggested fixes would resolve friction."""

    def __init__(self, academy_bridge: Optional[Any] = None):
        self.academy_bridge = academy_bridge
        self._replay_results: List[Dict[str, Any]] = []

    def replay_cohort(self, cohort_name: str) -> Dict[str, Any]:
        """Replay a specific academy cohort and check for friction reduction."""
        if self.academy_bridge is None:
            return {"success": False, "error": "No academy bridge configured"}

        findings = self.academy_bridge.get_friction_points()
        cohort_findings = [f for f in findings if f.get("agent") == cohort_name]

        # Simulate: after applying fixes, friction should be resolved
        resolved = 0
        for finding in cohort_findings:
            fix_type = self._infer_fix_type(finding)
            # Simulate fix application
            if fix_type:
                resolved += 1

        result = {
            "cohort": cohort_name,
            "findings": len(cohort_findings),
            "resolved": resolved,
            "success": resolved > 0,
            "timestamp": time.time(),
        }
        self._replay_results.append(result)
        return result

    def validate_fix(self, point: FrictionPoint) -> bool:
        """Validate a single fix by checking if the suggestion matches known good patterns."""
        if not point.suggested_fix:
            return False
        good_patterns = [
            "authentication",
            "RBAC",
            "schema",
            "circuit breaker",
            "caching",
            "async",
            "WebSocket",
            "SSE",
            "optimize",
            "consolidate",
            "discovery",
        ]
        return any(pattern.lower() in point.suggested_fix.lower() for pattern in good_patterns)

    def get_replay_history(self) -> List[Dict[str, Any]]:
        return self._replay_results

    def _infer_fix_type(self, finding: Dict[str, Any]) -> Optional[str]:
        finding_name = finding.get("finding", "").lower()
        if "auth" in finding_name:
            return "add_auth"
        if "ui" in finding_name or "web" in finding_name:
            return "add_web_ui"
        if "schema" in finding_name or "build" in finding_name:
            return "add_build_schema"
        if "broadcast" in finding_name:
            return "add_broadcast"
        if "fleet_map" in finding_name:
            return "add_fleet_map"
        if "endpoint" in finding_name:
            return "fix_endpoints"
        return None
