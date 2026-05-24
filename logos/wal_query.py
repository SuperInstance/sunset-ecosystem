"""WAL Query — searchable, auditable interface over SignedWAL.

Provides operators with time-range, event-type, node, and room filters
over the cryptographically-verified append-only log.
"""

from __future__ import annotations

__all__ = ["WALQuery"]

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from logos.signed_wal import SignedWAL, SignedEntry, WALEntry

logger = logging.getLogger(__name__)


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
    """

    def __init__(self, wal: SignedWAL) -> None:
        self._wal = wal

    # ── Simple filters ────────────────────────────────────────────

    def by_time_range(self, start: str, end: str) -> list[SignedEntry]:
        """Return entries whose timestamp falls in [*start*, *end*).

        *start* and *end* are ISO-8601 strings.
        """
        t0 = _parse_iso8601(start)
        t1 = _parse_iso8601(end)
        return [se for se in self._wal.entries if t0 <= se.entry.timestamp < t1]

    def by_event_type(self, event_type: str) -> list[SignedEntry]:
        """Filter by *operation* field (e.g. ``'spawn'``, ``'breed'``)."""
        return [se for se in self._wal.entries if se.entry.operation == event_type]

    def by_node_id(self, node_id: str) -> list[SignedEntry]:
        """Filter by *node_id* field."""
        return [se for se in self._wal.entries if se.entry.node_id == node_id]

    def by_room_id(self, room_id: str) -> list[SignedEntry]:
        """Filter by *room_id* field."""
        return [se for se in self._wal.entries if se.entry.room_id == room_id]

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
