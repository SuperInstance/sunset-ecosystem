# SPEC-MULTI-INSTANCE-MESH.md
**Author:** CCC (Fleet Architect)  
**Date:** 2026-05-22  
**Status:** DRAFT — Awaiting Forgemaster Review  
**Target:** sunset-ecosystem v0.4.0  

---

## 1. Problem Statement

The sunset-ecosystem currently runs as a single-node Python process on one machine (Oracle1, ProArt, JetsonClaw1, or Alibaba Cloud). Each node has its own RoomGrid, tournament state, breeding queue, and FLUX VM. There is no mechanism for:

1. **Discovery** — Oracle1 does not know JetsonClaw1 exists unless manually configured.
2. **State sharing** — Tournament scores on ProArt are invisible to the breeder on Oracle1.
3. **Cross-instance breeding** — A winning agent on JetsonClaw1 cannot fertilize a cold room on Alibaba Cloud.
4. **Failure recovery** — When Oracle1 drops, its rooms are lost; no rebirth on surviving nodes.
5. **Fleet-wide constraints** — FLUX constraint violations on one node do not raise the alarm fleet-wide.

The `nexus/federation.py` module provides node registration and heartbeats, but it is a **lobby**, not a **mesh**. It tells you who is online; it does not route data, sync state, or coordinate breeding.

We need a **Multi-Instance Mesh** (MIM) that transforms N isolated sunset-ecosystem instances into a single distributed organism.

---

## 2. Design Principles

1. **Gossip over consensus.** Room weights and tournament scores are eventually consistent. We do not run Raft or Paxos — the fleet is too dynamic, nodes too heterogeneous. Vector table entries propagate via anti-entropy gossip.

2. **Security by default.** No plaintext secrets. All inter-node traffic is TLS 1.3 with mTLS (client certificates). No open ports without auth. The nexus registration endpoint is the *only* public face; mesh traffic flows over authenticated gRPC streams.

3. **Thermal-aware routing.** A breeding request from a thermally saturated node is rejected or redirected to a cooler peer. The mesh respects the thermal budget *fleet-wide*, not just locally.

4. **Sunset + Rebirth at the mesh layer.** When a node drops, its rooms' vectors are not lost — they exist in the replicated vector tables of surviving nodes. A "rebirth" on the mesh means: clone a vector from node A onto node B.

---

## 3. Architecture Diagram

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Oracle1   │◄───►│   ProArt    │◄───►│ JetsonClaw1 │◄───►│  Alibaba    │
│  (n=1000)   │ mesh│  (n=250)    │ mesh│  (n=100)    │ mesh│ Cloud (n=500)│
│  GPU + Rust │     │  GPU + Rust │     │  ARM + CUDA │     │  x86 + AVX  │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │                   │
       │                   │                   │                   │
       └───────────────────┴───────────────────┴───────────────────┘
                           │
                    ┌───────┴───────┐
                    │  Federated    │  ← 147.224.38.131:4047
                    │    Nexus      │     (Registration + DNS-SD)
                    │   (SQLite)    │
                    └───────────────┘
                           │
                    ┌───────┴───────┐
                    │   gRPC Mesh   │  ← mTLS, bidirectional streams
                    │   Backplane   │     (State sync + breeding RPCs)
                    └───────────────┘
```

### Discovery Layer
- **Federated Nexus** (`nexus/federation.py`) remains the canonical registry. Nodes register on startup with capabilities (`["gpu", "cuda_12", "rust_kernel", "flux_vm"]`).
- **DNS-SD via mDNS** (optional): On local LANs (ProArt ↔ Oracle1), nodes also broadcast `_sunset._tcp` services. The nexus merges mDNS discoveries with its registry.
- **Bootstrap**: Each node carries a `SEEDS.yaml` with 3–5 well-known endpoints. First boot connects to seeds; subsequent boots connect to the nexus.

### Communication Layer
- **gRPC bidirectional streams** between every pair of mesh peers. Why gRPC?
  - Native support for streaming RPCs (perfect for propagating vector deltas).
  - Dead-simple code generation from `.proto` specs.
  - HTTP/2 multiplexing — one TCP connection carries many concurrent streams.
  - Python `grpcio` is battle-tested; Rust `tonic` is async-native.
- **Alternative rejected**: WebSocket (no streaming RPC semantics), MQTT (broker is a single point of failure), custom binary (reinventing gRPC).

### State Sync Layer
- **Shared (replicated across mesh):**
  - `FluxVectorTable` entries — agent DNA vectors + metadata.
  - Tournament scores (top-k per node, not full population).
  - FLUX constraint violation log (last 1000 entries).
  - Node heartbeat + capability advertisements.
- **Local (not replicated):**
  - Full room weight tensors (too large; 3.4K params × 1000 rooms = 3.4 MB per node — borderline, but we sync *parent vectors* not full weights).
  - Thermal budget state (local cooling curves differ per hardware).
  - Detailed room activity histories.

### Mesh Topology
- **Full mesh** for N ≤ 8 nodes (Cocapn fleet is small).
- **Gossip protocol** for N > 8: each node gossips to 3 random peers every 5 seconds. Vector table deltas propagate via anti-entropy.

---

## 4. API Surface

```protobuf
// proto/mesh.proto
syntax = "proto3";
package sunset.mesh;

service MeshNode {
  // Bidirectional stream: exchange delta updates
  rpc SyncStream(stream DeltaUpdate) returns (stream DeltaUpdate);
  
  // Request a cross-instance breed
  rpc RequestBreed(BreedRequest) returns (BreedResponse);
  
  // Fleet-wide constraint violation broadcast
  rpc AlertConstraint(ConstraintAlert) returns (Ack);
  
  // Heartbeat + capability ping
  rpc Ping(PingMsg) returns (PongMsg);
}

message DeltaUpdate {
  string node_id = 1;
  int64 sequence = 2;           // Lamport-like logical clock
  oneof payload {
    VectorDelta vectors = 10;   // New/modified agent vectors
    TournamentDelta scores = 11; // Top-k score updates
    ConstraintViolation violation = 12;
    NodeCapability caps = 13;
  }
}

message VectorDelta {
  repeated AgentVector agents = 1;
}

message AgentVector {
  uint64 agent_id = 1;
  bytes quantized_vector = 2;   // turbovec 2–4 bit quantized
  float fitness = 3;
  uint32 generation = 4;
  uint32 capability_mask = 5;
  float thermal_pressure = 6;
  string origin_node = 7;       // which node birthed this agent
}

message BreedRequest {
  string requesting_node = 1;
  uint64 parent_a_id = 2;
  uint64 parent_b_id = 3;
  uint32 target_room_idx = 4;   // local room on requester
  string preferred_device = 5; // "cuda", "metal", "cpu"
}

message BreedResponse {
  bool accepted = 1;
  string responder_node = 2;
  uint64 child_id = 3;
  bytes child_vector = 4;
  string reason = 5;            // if rejected: "thermal_full", "incompatible_caps"
}
```

### Python API

```python
# mesh/node.py
class MeshNode:
    """One node in the multi-instance mesh."""
    
    def __init__(
        self,
        node_id: str,
        nexus: FederatedNexus,
        grid: RoomGrid,
        vector_table: FluxVectorTable,
        thermal: ThermalBudget,
    ) -> None:
        ...
    
    async def start(self) -> None:
        """Register with nexus, discover peers, open gRPC streams."""
        
    async def stop(self) -> None:
        """Graceful shutdown: notify peers, close streams."""
        
    async def sync_vectors(self) -> None:
        """Push local vector deltas to all peers."""
        
    async def request_remote_breed(
        self,
        parent_a: int,
        parent_b: int,
        target_room: int,
    ) -> BreedResponse:
        """Ask the coolest peer to produce a child from two parents."""
        
    def peers(self) -> list[PeerInfo]:
        """Currently reachable mesh peers."""
```

---

## 5. Open Questions

1. **gRPC vs QUIC**: QUIC has better NAT traversal (Oracle1 is behind a router). Do we use gRPC-over-QUIC via `quiche`? Or stick to TCP gRPC and handle NAT via the nexus relay?

2. **Vector sync frequency**: Every vector table add triggers a delta push — is this too chatty for 1000-room nodes? Should we batch every 500ms?

3. **Conflict resolution**: Two nodes breed from the same parent vector simultaneously. The child IDs will diverge. Is this acceptable (eventual consistency) or do we need a breeding lease mechanism?

4. **Cross-cloud latency**: Alibaba Cloud ↔ Oracle1 is ~150ms RTT. Cross-instance breeding adds at least 2 RTTs (request + response). Is 300ms acceptable for a breeding cycle that otherwise takes 10ms locally? Should we restrict remote breeding to "emergency diversity" scenarios?

5. **FLUX proof propagation**: FLUX VM produces `ProofCertificate` per constraint check. Should these certificates flow across the mesh for fleet-wide audit, or stay local? Each certificate is ~256 bytes; 1000 rooms × 30 ticks/sec = 7.5 MB/sec of proof traffic — too much for WAN.

---

## 6. Implementation Order

### P0 — Foundation (Week 1)
- [ ] Define `proto/mesh.proto` and generate Python/Rust stubs.
- [ ] Implement `MeshNode` class with gRPC `SyncStream` (empty payload for now — just heartbeats).
- [ ] Wire `FederatedNexus` discovery into `MeshNode.start()`.
- [ ] mTLS: generate CA + node certs, add `MeshNode` TLS config.
- [ ] Integration test: 2-node mesh on localhost, verify bidirectional stream stays alive.

### P1 — State Sync (Week 2)
- [ ] `VectorDelta` serialization: turbovec → protobuf → turbovec roundtrip.
- [ ] `sync_vectors()`: push on every `FluxVectorTable.add()`.
- [ ] `TournamentDelta`: sync top-20 scores per node every 5 seconds.
- [ ] `ConstraintAlert`: broadcast FLUX violations fleet-wide.
- [ ] Integration test: 3-node mesh, verify vector table converges across all nodes.

### P2 — Cross-Instance Features (Week 3–4)
- [ ] `request_remote_breed()`: thermal-aware peer selection + RPC.
- [ ] Mesh-layer sunset: when a node drops, its vectors are marked `orphaned`; surviving nodes may adopt them.
- [ ] FLUX proof aggregation: local proofs → batch summary → fleet-wide alert (not per-proof).
- [ ] Prometheus metrics: mesh_latency_ms, sync_divergence, remote_breed_success_rate.

---

## References

- `nexus/federation.py` — existing registration/heartbeat (lobby layer)
- `swarm/vector_table.py` — `FluxVectorTable` (state to sync)
- `swarm/breeder_daemon.py` — `AutoBreeder` (consumer of mesh breeding)
- `sunset/flux_integration.py` — `FluxConstraintChecker` (producer of constraint alerts)
- `docs/SPEC-FLUX-RESOLUTION.md` — CCC's FLUX v3 decision (certificates format)
- `docs/SPEC-BREEDER.md` — agent template + thermal spawning (breeding semantics)
