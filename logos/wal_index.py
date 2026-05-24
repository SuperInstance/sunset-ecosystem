"""WAL Index — in-memory inverted indices for fast compound queries.

Builds on WAL load and stays incrementally updated on every append.
"""

from __future__ import annotations

__all__ = ["WALIndex"]

import bisect
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from logos.signed_wal import SignedWAL, SignedEntry, WALEntry

logger = logging.getLogger(__name__)


def _ts_bucket(ts: float) -> str:
    """Round timestamp to the minute for time-index bucketing."""
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M")


class WALIndex:
    """In-memory inverted indices over a SignedWAL.

    Attributes:
        by_time:   ``{ISO-minute: [entry_idx, ...]}``
        by_type:   ``{event_type: [entry_idx, ...]}``
        by_node:   ``{node_id: [entry_idx, ...]}``
        by_room:   ``{room_id: [entry_idx, ...]}``
        _sorted_ts: list of ``(timestamp, entry_idx)`` kept sorted for range queries.
    """

    def __init__(self, wal: SignedWAL) -> None:
        self._wal = wal
        self.by_time: dict[str, list[int]] = defaultdict(list)
        self.by_type: dict[str, list[int]] = defaultdict(list)
        self.by_node: dict[str, list[int]] = defaultdict(list)
        self.by_room: dict[str, list[int]] = defaultdict(list)
        self._sorted_ts: list[tuple[float, int]] = []
        self.rebuild()

    # ── Rebuild / Update ──────────────────────────────────────────

    def rebuild(self) -> None:
        """Reconstruct all indices from a full WAL scan."""
        self.by_time.clear()
        self.by_type.clear()
        self.by_node.clear()
        self.by_room.clear()
        self._sorted_ts.clear()

        for idx, se in enumerate(self._wal.entries):
            self._index_entry(idx, se)
            self._sorted_ts.append((se.entry.timestamp, idx))

        self._sorted_ts.sort(key=lambda x: x[0])
        logger.info("WALIndex rebuilt: %d entries indexed", len(self._wal.entries))

    def update(self, new_entry: SignedEntry) -> None:
        """Incrementally index a newly appended entry."""
        idx = len(self._wal.entries) - 1
        # In case the entry was appended before update() was called,
        # find the actual index.
        try:
            idx = self._wal.entries.index(new_entry)
        except ValueError:
            idx = len(self._wal.entries) - 1
        self._index_entry(idx, new_entry)
        bisect.insort(self._sorted_ts, (new_entry.entry.timestamp, idx))

    def _index_entry(self, idx: int, se: SignedEntry) -> None:
        """Add one entry to all inverted indices."""
        bucket = _ts_bucket(se.entry.timestamp)
        self.by_time[bucket].append(idx)
        self.by_type[se.entry.operation].append(idx)
        if se.entry.node_id:
            self.by_node[se.entry.node_id].append(idx)
        if se.entry.room_id:
            self.by_room[se.entry.room_id].append(idx)

    # ── Compound queries ────────────────────────────────────────────

    def query(
        self,
        *,
        conjunction: str = "and",
        filters: list[dict[str, Any]] | None = None,
    ) -> list[SignedEntry]:
        """Multi-filter compound query.

        Args:
            conjunction: ``'and'`` (default) or ``'or'``.
            filters: list of filter dicts, each shaped like::

                {"field": "event_type", "value": "spawn"}
                {"field": "node_id",    "value": "node-7"}
                {"field": "room_id",    "value": "forge"}
                {"field": "time_range", "start": "2024-01-01T00:00:00Z",
                                         "end":   "2024-01-02T00:00:00Z"}

        Returns:
            Matching ``SignedEntry`` objects in chronological order.
        """
        if not filters:
            return list(self._wal.entries)

        candidate_sets: list[set[int]] = []
        for filt in filters:
            field = filt.get("field")
            if field == "event_type":
                candidate_sets.append(set(self.by_type.get(filt["value"], [])))
            elif field == "node_id":
                candidate_sets.append(set(self.by_node.get(filt["value"], [])))
            elif field == "room_id":
                candidate_sets.append(set(self.by_room.get(filt["value"], [])))
            elif field == "time_range":
                candidate_sets.append(self._time_range_set(filt["start"], filt["end"]))
            else:
                # Fallback: linear scan over the whole WAL
                matches = {
                    i
                    for i, se in enumerate(self._wal.entries)
                    if self._match_generic(se, filt)
                }
                candidate_sets.append(matches)

        if not candidate_sets:
            return []

        if conjunction == "and":
            indices = set.intersection(*candidate_sets)
        else:  # "or"
            indices = set.union(*candidate_sets)

        return [self._wal.entries[i] for i in sorted(indices)]

    def _time_range_set(self, start: str, end: str) -> set[int]:
        """Return entry indices whose timestamps fall in the range."""
        from logos.wal_query import _parse_iso8601

        t0 = _parse_iso8601(start)
        t1 = _parse_iso8601(end)
        # Use binary search on _sorted_ts for efficiency
        left = bisect.bisect_left(self._sorted_ts, (t0, -1))
        right = bisect.bisect_left(self._sorted_ts, (t1, -1))
        return {idx for _, idx in self._sorted_ts[left:right]}

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

    # ── Convenience accessors ─────────────────────────────────────

    def all_event_types(self) -> list[str]:
        """Return all indexed event types."""
        return list(self.by_type.keys())

    def all_nodes(self) -> list[str]:
        """Return all indexed node IDs."""
        return list(self.by_node.keys())

    def all_rooms(self) -> list[str]:
        """Return all indexed room IDs."""
        return list(self.by_room.keys())

    def __repr__(self) -> str:
        return (
            f"WALIndex(entries={len(self._wal.entries)}, "
            f"types={len(self.by_type)}, nodes={len(self.by_node)}, "
            f"rooms={len(self.by_room)})"
        )
