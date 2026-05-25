"""WAL Query — searchable, auditable interface over SignedWAL.

Provides operators with time-range, event-type, node, and room filters
over the cryptographically-verified append-only log.

**Index-aware mode**: pass a ``WALIndex`` to use inverted indices for
sub-linear compound queries instead of linear scans.
"""

from __future__ import annotations

__all__ = ["WALQuery", "BatchQueryResult"]

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from logos.signed_wal import SignedWAL, SignedEntry, WALEntry

logger = logging.getLogger(__name__)


@dataclass
class BatchQueryResult:
    """Result of a batched multi-query execution."""

    query_id: str
    entries: list[SignedEntry] = field(default_factory=list)
    duration_ms: float = 0.0
    index_used: bool = False
    error: str | None = None

    def __len__(self) -> int:
        return len(self.entries)

    def __bool__(self) -> bool:
        return len(self.entries) > 0


def _parse_iso8601(ts: str) -> float:
    """Parse ISO-8601 string to POSIX timestamp."""
    # Support both 'Z' suffix and explicit offsets
    ts = ts.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError as exc:
        raise ValueError(f"Invalid ISO-8601 timestamp: {ts}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _format_iso8601(ts: float) -> str:
    """Format POSIX timestamp as ISO-8601 UTC string."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


class WALQuery:
    """Query interface for a SignedWAL.

    Usage::

        wal = SignedWAL(log_path="/data/fleet.wal")
        q = WALQuery(wal)
        spawns = q.by_event_type("spawn")
        today = q.by_time_range("2024-05-25T00:00:00Z", "2024-05-26T00:00:00Z")

    Index-aware usage::

        from logos.wal_index import WALIndex
        idx = WALIndex(wal)
        q = WALQuery(wal, index=idx)
        # compound queries now use inverted indices
        results = q.compound_query(
            conjunction="and",
            filters=[
                {"field": "event_type", "value": "spawn"},
                {"field": "node_id", "value": "node-alpha"},
            ],
        )
    """

    def __init__(self, wal: SignedWAL, index: Any | None = None) -> None:
        self._wal = wal
        self._index = index

    # ── Simple filters (index-backed when available) ─────────────

    def by_time_range(self, start: str, end: str) -> list[SignedEntry]:
        """Return entries whose timestamp falls in [*start*, *end*).

        *start* and *end* are ISO-8601 strings.
        """
        if self._index is not None:
            return self._index.query(
                conjunction="and",
                filters=[{"field": "time_range", "start": start, "end": end}],
            )
        t0 = _parse_iso8601(start)
        t1 = _parse_iso8601(end)
        return [se for se in self._wal.entries if t0 <= se.entry.timestamp < t1]

    def by_event_type(self, event_type: str) -> list[SignedEntry]:
        """Filter by *operation* field (e.g. ``'spawn'``, ``'breed'``)."""
        if self._index is not None:
            return self._index.query(
                conjunction="and",
                filters=[{"field": "event_type", "value": event_type}],
            )
        return [se for se in self._wal.entries if se.entry.operation == event_type]

    def by_node_id(self, node_id: str) -> list[SignedEntry]:
        """Filter by *node_id* field."""
        if self._index is not None:
            return self._index.query(
                conjunction="and",
                filters=[{"field": "node_id", "value": node_id}],
            )
        return [se for se in self._wal.entries if se.entry.node_id == node_id]

    def by_room_id(self, room_id: str) -> list[SignedEntry]:
        """Filter by *room_id* field."""
        if self._index is not None:
            return self._index.query(
                conjunction="and",
                filters=[{"field": "room_id", "value": room_id}],
            )
        return [se for se in self._wal.entries if se.entry.room_id == room_id]

    # ── Compound queries (index-backed when available) ────────────

    def compound_query(
        self,
        *,
        conjunction: str = "and",
        filters: list[dict[str, Any]] | None = None,
    ) -> list[SignedEntry]:
        """Multi-filter compound query.

        When an index is attached this uses inverted indices for
        O(log N) intersections instead of O(N) linear scans.
        """
        if self._index is not None:
            return self._index.query(conjunction=conjunction, filters=filters)

        # Fallback: linear scan
        if not filters:
            return list(self._wal.entries)

        candidate_sets: list[set[int]] = []
        for filt in filters:
            field = filt.get("field")
            value = filt.get("value")
            matches = set()
            for i, se in enumerate(self._wal.entries):
                if field == "event_type" and se.entry.operation == value:
                    matches.add(i)
                elif field == "node_id" and se.entry.node_id == value:
                    matches.add(i)
                elif field == "room_id" and se.entry.room_id == value:
                    matches.add(i)
                elif field == "time_range":
                    t0 = _parse_iso8601(filt["start"])
                    t1 = _parse_iso8601(filt["end"])
                    if t0 <= se.entry.timestamp < t1:
                        matches.add(i)
                elif self._match_generic(se, filt):
                    matches.add(i)
            candidate_sets.append(matches)

        if not candidate_sets:
            return []

        if conjunction == "and":
            indices = set.intersection(*candidate_sets)
        else:  # "or"
            indices = set.union(*candidate_sets)

        return [self._wal.entries[i] for i in sorted(indices)]

    # ── Batch queries ─────────────────────────────────────────────

    def batch_query(
        self,
        queries: list[dict[str, Any]],
        *,
        timeout_ms: float = 5000.0,
    ) -> list[BatchQueryResult]:
        """Execute multiple queries in one pass.

        Each query dict must have keys::

            {
                "query_id": str,
                "conjunction": "and" | "or",
                "filters": [...],  # same shape as compound_query
            }

        Returns a list of ``BatchQueryResult`` in the same order.
        """
        import time

        results: list[BatchQueryResult] = []
        t0 = time.monotonic()
        for q in queries:
            qid = q.get("query_id", f"q-{len(results)}")
            q_start = time.monotonic()
            try:
                entries = self.compound_query(
                    conjunction=q.get("conjunction", "and"),
                    filters=q.get("filters"),
                )
                elapsed = (time.monotonic() - q_start) * 1000
                results.append(
                    BatchQueryResult(
                        query_id=qid,
                        entries=entries,
                        duration_ms=elapsed,
                        index_used=self._index is not None,
                    )
                )
            except Exception as exc:
                elapsed = (time.monotonic() - q_start) * 1000
                results.append(
                    BatchQueryResult(
                        query_id=qid,
                        entries=[],
                        duration_ms=elapsed,
                        index_used=self._index is not None,
                        error=str(exc),
                    )
                )
            # Global timeout check
            if (time.monotonic() - t0) * 1000 > timeout_ms:
                logger.warning("Batch query global timeout reached after %d queries", len(results))
                break
        return results

    # ── Index hint helpers ──────────────────────────────────────

    def with_index(self, index: Any) -> "WALQuery":
        """Return a new WALQuery backed by the given index."""
        return WALQuery(self._wal, index=index)

    def without_index(self) -> "WALQuery":
        """Return a new WALQuery that performs linear scans."""
        return WALQuery(self._wal, index=None)

    def explain(self, filters: list[dict[str, Any]]) -> dict[str, Any]:
        """Explain how a query would be executed (index vs scan)."""
        plan: dict[str, Any] = {
            "index_available": self._index is not None,
            "index_name": type(self._index).__name__ if self._index else None,
            "filters": len(filters),
            "indexable_filters": 0,
            "fallback_filters": 0,
        }
        for filt in filters:
            field = filt.get("field", "")
            if field in ("event_type", "node_id", "room_id", "time_range"):
                plan["indexable_filters"] += 1
            else:
                plan["fallback_filters"] += 1
        plan["strategy"] = (
            "index_intersection"
            if plan["index_available"] and plan["indexable_filters"] > 0
            else "linear_scan"
        )
        return plan

    # ── Integrity ───────────────────────────────────────────────────

    def verify_subset(self, entries: list[SignedEntry]) -> bool:
        """Verify signatures on a subset without scanning the full WAL.

        Returns ``True`` only if every entry in *entries* has a valid
        signature and a correct hash chain (each entry's *previous_hash*
        matches the SHA-256 of its predecessor in *entries*).
        """
        if not entries:
            return True
        prev_hash = ""
        for se in entries:
            if not self._wal.verify(se):
                logger.warning("verify_subset: signature invalid for agent %s", se.entry.agent_id)
                return False
            if se.previous_hash != prev_hash:
                logger.warning(
                    "verify_subset: hash mismatch (expected %s, got %s)",
                    prev_hash[:16], se.previous_hash[:16],
                )
                return False
            prev_hash = se.compute_hash()
        return True

    def verify_chain(self) -> list[Any]:
        """Run a full chain-of-custody audit."""
        return self._wal.verify_chain()

    # ── Summary ─────────────────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        """Fleet-health dashboard data.

        Returns a dict with:
            - ``event_counts``: ``{event_type: count}``
            - ``time_range``: ``{"start": ISO, "end": ISO}``
            - ``node_coverage``: ``{node_id: count}``
            - ``room_coverage``: ``{room_id: count}``
            - ``total_entries``: int
        """
        entries = self._wal.entries
        if not entries:
            return {
                "event_counts": {},
                "time_range": {"start": None, "end": None},
                "node_coverage": {},
                "room_coverage": {},
                "total_entries": 0,
            }

        event_counts: dict[str, int] = {}
        node_coverage: dict[str, int] = {}
        room_coverage: dict[str, int] = {}
        timestamps: list[float] = []

        for se in entries:
            op = se.entry.operation
            event_counts[op] = event_counts.get(op, 0) + 1

            nid = se.entry.node_id
            if nid:
                node_coverage[nid] = node_coverage.get(nid, 0) + 1

            rid = se.entry.room_id
            if rid:
                room_coverage[rid] = room_coverage.get(rid, 0) + 1

            timestamps.append(se.entry.timestamp)

        return {
            "event_counts": event_counts,
            "time_range": {
                "start": _format_iso8601(min(timestamps)),
                "end": _format_iso8601(max(timestamps)),
            },
            "node_coverage": node_coverage,
            "room_coverage": room_coverage,
            "total_entries": len(entries),
        }

    # ── Export ──────────────────────────────────────────────────────

    def export_jsonl(self, path: str | Path, entries: list[SignedEntry] | None = None) -> None:
        """Write entries (or all WAL entries) as newline-delimited JSON.

        Each line is a JSON object with the *entry* fields, signature,
        previous_hash, and public_key hex-encoded.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = entries if entries is not None else self._wal.entries
        with open(target, "w", encoding="utf-8") as f:
            for se in data:
                record = {
                    "entry": {
                        "timestamp": se.entry.timestamp,
                        "agent_id": se.entry.agent_id,
                        "operation": se.entry.operation,
                        "vector_hash": se.entry.vector_hash,
                        "parent_ids": se.entry.parent_ids,
                        "generation": se.entry.generation,
                        "node_id": se.entry.node_id,
                        "room_id": se.entry.room_id,
                    },
                    "signature": se.signature.hex(),
                    "previous_hash": se.previous_hash,
                    "public_key": se.public_key.hex(),
                }
                f.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        logger.info("Exported %d entries to %s", len(data), target)

    # ── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _match_generic(se: SignedEntry, filt: dict[str, Any]) -> bool:
        """Fallback linear-scan matcher for arbitrary fields."""
        field = filt.get("field")
        value = filt.get("value")
        if field == "agent_id":
            return se.entry.agent_id == value
        if field == "generation":
            return se.entry.generation == value
        if field == "vector_hash":
            return se.entry.vector_hash == value
        # Allow dotted access like "entry.timestamp"
        if field and field.startswith("entry."):
            attr = field[6:]
            return getattr(se.entry, attr, None) == value
        return False
