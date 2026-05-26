# SignedWAL Query Layer

`logos/wal_query.py` adds fast indexed queries on top of `SignedWAL`.

## Why

The base WAL only supports `append`, `verify`, and `entries`. Any query
(e.g. "all spawn ops for agent 42 in the last hour") required a full
O(n) scan. This module adds secondary indexes for O(1) agent lookups and
O(log n) time-range queries.

## Components

| Class | Role |
|-------|------|
| `WALQueryIndex` | Secondary indexes (by_agent, by_operation, by_node, time_index, generation_index). Rebuilds on load, incrementally updates on append. |
| `WALQueryFilter` | Declarative filter: agent_id, operation, node_id, time range, generation range, parent_ids, custom predicate. |
| `WALBatchQuery` | High-level interface: `filter()`, `count()`, convenience methods (`by_agent`, `by_time_range`, `latest_by_agent`, `descendants`, `genealogy`), `batch_verify()`, `range_scan()`. |

## Query Planning

`WALQueryIndex.plan(filter)` picks the most restrictive indexed hint:

1. If `agent_id` given → use `by_agent` index
2. If `operation` given → use `by_operation` index
3. If `node_id` given → use `by_node` index
4. If time range given → bisect `time_index` (O(log n))
5. If generation range given → bisect `generation_index` (O(log n))

The smallest candidate set wins. If no hint applies, falls back to full scan.

## Example

```python
from logos.signed_wal import SignedWAL
from logos.wal_query import WALQueryIndex, WALBatchQuery, WALQueryFilter

wal = SignedWAL("/tmp/test.wal")
idx = WALQueryIndex()
idx.rebuild(wal.entries)
query = WALBatchQuery(wal.entries, idx)

# All spawn ops for agent 42 in the last hour
results = query.filter(
    WALQueryFilter(
        agent_id=42,
        operation="spawn",
        time_start=time.time() - 3600,
    )
)

# Latest 5 entries for agent 42
latest = query.latest_by_agent(42, n=5)

# Direct range scan: last 100 entries
recent = query.range_scan(-100, None)
```

## Tests

`tests/test_wal_query.py` — 33 tests covering:
- Index rebuild and incremental updates
- Hint accuracy (agent, operation, time, generation)
- Query planning (smallest candidate set)
- Filter combinations (AND semantics)
- Convenience queries (by_agent, by_time_range, descendants, genealogy)
- Count without materialization
- Batch verify
- Range scans
- Edge cases (empty WAL, no matches, offset/limit)
