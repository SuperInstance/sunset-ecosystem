# Commit-Caster I2I Router

Broadcasts commit events from any SuperInstance repository to the Cocapn Fleet mesh.

## For Humans

**What it does:** When someone pushes code to any SuperInstance repo, the fleet learns about it within seconds. This means:
- Agents can auto-update their knowledge of ecosystem changes
- Breeders can incorporate new commits into their fitness evaluation
- Fleet operators see real-time activity across all repos

**Setup:**
1. Add `.github/workflows/commit-caster.yml` to your repo
2. Set repository secrets:
   - `FLEET_WEBHOOK_SECRET` — shared secret for HMAC-SHA256 signing
   - `FLEET_GATEWAY_URL` — fleet gateway endpoint (e.g., `https://fleet.cocapn.com`)
3. Done. Every push triggers the action.

## For Agents

**Primary interface:** `fleet/commit_caster.py`

```python
from fleet.commit_caster import CommitCaster, CommitEvent

# Initialize with shared secret and mesh broadcast function
caster = CommitCaster(
    secret="shared-secret", mesh_broadcast=lambda d: mesh_gossip.broadcast("commit", d)
)

# Receive webhook payload
event = caster.receive(payload_bytes, signature_header)
if event:
    print(f"New commit in {event.repo}: {event.commit}")
```

**Key behaviors:**
- **Deduplication:** 60-second window prevents duplicate broadcasts
- **Signature validation:** HMAC-SHA256 prevents spoofing
- **Queueing:** If mesh is down, events are queued and retried
- **Graceful degradation:** Never blocks the GitHub Action on fleet unavailability

## Architecture

```
GitHub Push
    ↓
GitHub Action (commit-caster.yml)
    ↓
HMAC-SHA256 signed POST
    ↓
Fleet Gateway /i2i/commit
    ↓
CommitCaster.receive()
    ↓
Validate → Deduplicate → Broadcast
    ↓
Fleet Mesh Gossip / Event Bus
```

## Data Format

```json
{
  "repo": "SuperInstance/sunset-ecosystem",
  "commit": "abc123def...",
  "author": "kimi1",
  "message": "Add spatial projector",
  "branch": "main",
  "timestamp": "2024-01-01T00:00:00Z",
  "files": ["fleet/spatial_projector.py", "tests/test_spatial_projector.py"]
}
```

## Security

- Webhook signatures are mandatory. Unsigned requests are rejected.
- Deduplication prevents replay attacks within the 60s window.
- Secrets are never logged or exposed in error messages.

## Integration Points

- `fleet/event_bus.py` — broadcast commit events to subscribers
- `swarm/mesh_gossip.py` — cross-node propagation
- `fleet/fleet_korok.py` — index commit messages for search
- `fleet/fleet_mem0.py` — store commit context in agent memory
