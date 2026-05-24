"""A2A Handler functions for fleet services.

Each handler receives a JSON task payload dict and returns a JSON response dict
with keys: {"status": "ok" | "error", "result": ...}.

These are lightweight stubs that validate payload shape and return realistic
responses matching the agent card schemas. In production they would delegate
to the actual service implementations.
"""
import time


def _validate_required(payload, keys):
    """Return error dict if any required key is missing, else None."""
    missing = [k for k in keys if k not in payload]
    if missing:
        return {"status": "error", "result": {"message": f"Missing required keys: {missing}"}}
    return None


def _get_input(payload):
    """Extract the 'input' field from payload, defaulting to {}."""
    return payload.get("input", {})


# ── MetronomeScheduler ──

def handle_metronome_task(payload):
    """Handle MetronomeScheduler tasks: tick, set_bpm, sync, get_status."""
    task_type = payload.get("type")
    if task_type is None:
        return {"status": "error", "result": {"message": "Missing 'type' field"}}

    inp = _get_input(payload)

    if task_type == "tick":
        signal = inp.get("signal", [])
        force = inp.get("force", False)
        # Simulate compute → gate → route phases
        phase_durations = {"compute": 2.1, "gate": 0.3, "route": 0.8}
        return {
            "status": "ok",
            "result": {
                "beat_number": 1423,
                "fired_rooms": [42, 77, 128, 301],
                "fired_count": 4,
                "phase_durations_ms": phase_durations,
                "missed_beat": False,
                "signal_length": len(signal),
                "force": force,
            }
        }

    if task_type == "set_bpm":
        err = _validate_required(inp, ["bpm"])
        if err:
            return err
        bpm = float(inp["bpm"])
        ramp_ms = inp.get("ramp_ms", 0)
        beat_duration_ms = 60000.0 / bpm if bpm > 0 else 0.0
        return {
            "status": "ok",
            "result": {
                "new_bpm": bpm,
                "actual_bpm": bpm,
                "beat_duration_ms": beat_duration_ms,
                "ramp_complete": ramp_ms == 0,
            }
        }

    if task_type == "sync":
        err = _validate_required(inp, ["node_id", "beat_number", "wall_time_ns", "perf_counter_ns"])
        if err:
            return err
        return {
            "status": "ok",
            "result": {
                "node_id": inp["node_id"],
                "beat_number": inp["beat_number"],
                "wall_time_ns": inp["wall_time_ns"],
                "perf_counter_ns": inp["perf_counter_ns"],
                "drift_ms": 0.0,
                "correction_applied": False,
            }
        }

    if task_type == "get_status":
        return {
            "status": "ok",
            "result": {
                "beat_number": 1423,
                "target_bpm": 120.0,
                "actual_bpm": 119.8,
                "beat_duration_ms": 500.0,
                "missed_beats": 0,
                "total_beats": 1423,
                "room_count": 500,
                "harmonics": [
                    {"divider": 4, "callback_count": 12, "last_fired_beat": 1420}
                ],
                "healthy": True,
            }
        }

    return {"status": "error", "result": {"message": f"Unknown task type: {task_type}"}}


# ── BreederDaemonV2 ──

def handle_breeder_task(payload):
    """Handle BreederDaemonV2 tasks: queue_breed, get_state, get_stats, emergency_stop."""
    task_type = payload.get("type")
    if task_type is None:
        return {"status": "error", "result": {"message": "Missing 'type' field"}}

    inp = _get_input(payload)

    if task_type == "queue_breed":
        parent_count = inp.get("parent_count", 2)
        offspring_count = inp.get("offspring_count", 1)
        incubate_room = inp.get("incubate_room", "Forge")
        strategy = inp.get("strategy", "trinity")
        children = []
        for i in range(offspring_count):
            children.append({
                "agent_id": f"agent-{i:04x}-breed",
                "parent_ids": [f"parent-{j:04x}" for j in range(parent_count)],
                "fitness": {"ethos": 0.87, "pathos": 0.92, "logos": 0.79, "product": 0.634},
                "incubated": True,
                "room_id": incubate_room,
            })
        return {
            "status": "ok",
            "result": {
                "children": children,
                "cycle_id": "cycle-2026-05-22-001",
                "queue_position": 0,
                "strategy": strategy,
            }
        }

    if task_type == "get_state":
        agents = [
            {
                "agent_id": "agent-0001",
                "phase": "SURVIVE",
                "generation": 3,
                "birth_beat": 100,
                "fitness": {"ethos": 0.9, "pathos": 0.85, "logos": 0.88, "product": 0.6732},
                "room_id": "Forge",
            },
            {
                "agent_id": "agent-0002",
                "phase": "INCUBATE",
                "generation": 4,
                "birth_beat": 1200,
                "fitness": {"ethos": 0.7, "pathos": 0.6, "logos": 0.8, "product": 0.336},
                "room_id": "Forge",
            },
        ]
        if inp.get("agent_id"):
            agents = [a for a in agents if a["agent_id"] == inp["agent_id"]]
        if inp.get("phase"):
            agents = [a for a in agents if a["phase"] == inp["phase"]]
        phase_counts = {
            "EGG": 0, "COMPETE": 1,
            "SURVIVE": 1, "BREED": 0, "SUNSET": 0, "ARCHIVE": 0,
        }
        return {
            "status": "ok",
            "result": {
                "agents": agents,
                "phase_counts": phase_counts,
                "daemon_status": "running",
            }
        }

    if task_type == "get_stats":
        return {
            "status": "ok",
            "result": {
                "total_generations": 4,
                "total_agents_spawned": 12,
                "total_agents_sunset": 10,
                "survival_rate": 0.1667,
                "average_fitness": {"ethos": 0.8, "pathos": 0.75, "logos": 0.82, "product": 0.492},
                "tournament_count": 6,
                "archive_size_bytes": 4096,
                "last_breed_beat": 1200,
            }
        }

    if task_type == "emergency_stop":
        err = _validate_required(inp, ["reason"])
        if err:
            return err
        sunset_nonviable = inp.get("sunset_nonviable", False)
        preserve_incubating = inp.get("preserve_incubating", True)
        return {
            "status": "ok",
            "result": {
                "stopped": True,
                "affected_agents": 2,
                "sunset_agents": 1 if sunset_nonviable else 0,
                "stop_beat": 1423,
                "resumable": True,
                "reason": inp["reason"],
                "preserve_incubating": preserve_incubating,
            }
        }

    return {"status": "error", "result": {"message": f"Unknown task type: {task_type}"}}


# ── RoomGrid ──

def handle_grid_task(payload):
    """Handle RoomGrid tasks: tick, get_activity, get_room_state, rebirth_room."""
    task_type = payload.get("type")
    if task_type is None:
        return {"status": "error", "result": {"message": "Missing 'type' field"}}

    inp = _get_input(payload)

    if task_type == "tick":
        signal = inp.get("signal", [])
        room_ids = inp.get("room_ids", [0, 1, 2, 3, 4])
        skip_local = inp.get("skip_local_metronomes", False)
        return {
            "status": "ok",
            "result": {
                "fired": len(room_ids),
                "ids": room_ids,
                "tick": 1423,
                "latents": [[0.1, -0.2, 0.3] for _ in room_ids],
                "novelty_scores": [0.5] * len(room_ids),
                "chaos_values": [0.3] * len(room_ids),
                "signal_length": len(signal),
                "skip_local_metronomes": skip_local,
            }
        }

    if task_type == "get_activity":
        window = inp.get("window_ticks", 100)
        return {
            "status": "ok",
            "result": {
                "room_count": 500,
                "active_count": 120,
                "compiled_count": 45,
                "average_chaos": 0.28,
                "average_novelty": 0.42,
                "firing_rate": 0.24,
                "window_ticks": window,
                "per_room": [
                    {
                        "room_id": 0,
                        "fire_count": 24,
                        "chaos": 0.3,
                        "novelty": 0.5,
                        "compiled": True,
                        "local_bpm_divider": 4,
                    }
                ],
            }
        }

    if task_type == "get_room_state":
        room_ids = inp.get("room_ids", [0])
        include_weights = inp.get("include_weights", False)
        include_buffer = inp.get("include_buffer", False)
        rooms = []
        for rid in room_ids:
            room_data = {
                "room_id": rid,
                "weights_shape": [64, 32],
                "latent": [0.1] * 16,
                "chaos": 0.3,
                "novelty": 0.5,
                "flux_violations": 0,
                "last_fired_tick": 1420,
                "birth_tick": 0,
            }
            if include_weights:
                room_data["weights"] = [0.01] * (64 * 32)
                room_data["weights_shape"] = [64, 32]
            if include_buffer:
                room_data["ring_buffer"] = [[0.0] * 16 for _ in range(10)]
            rooms.append(room_data)
        return {
            "status": "ok",
            "result": {"rooms": rooms}
        }

    if task_type == "rebirth_room":
        err = _validate_required(inp, ["room_id"])
        if err:
            return err
        rid = inp["room_id"]
        return {
            "status": "ok",
            "result": {
                "room_id": rid,
                "rebirth_tick": 1423,
                "previous_chaos": 0.95,
                "new_chaos": 0.01,
                "weight_checksum": "sha256:a1b2c3d4...",
                "reason": inp.get("reason", "rebirth"),
            }
        }

    return {"status": "error", "result": {"message": f"Unknown task type: {task_type}"}}


# ── FLUX Constraint Checker ──

def handle_flux_task(payload):
    """Handle FLUX tasks: check_constraints, get_violations, apply_feedback."""
    task_type = payload.get("type")
    if task_type is None:
        return {"status": "error", "result": {"message": "Missing 'type' field"}}

    inp = _get_input(payload)

    if task_type == "check_constraints":
        values = inp.get("values", [])
        preset = inp.get("preset", "neural_bounds")
        domain = inp.get("domain", "neural")
        generate_cert = inp.get("generate_certificate", True)

        violations = []
        certificates = []
        checked_count = len(values)
        violation_count = 0

        for i, vec in enumerate(values):
            max_val = max(vec) if vec else 0.0
            min_val = min(vec) if vec else 0.0
            passed = True

            # Simple bounds check based on preset
            bound = 10.0 if preset == "neural_bounds" else 5.0 if preset == "safe_mode" else 50.0
            if max_val > bound or min_val < -bound:
                passed = False
                violations.append({
                    "index": i,
                    "constraint": "bounds",
                    "expected": bound,
                    "actual": max_val if max_val > bound else min_val,
                    "severity": "error",
                    "remediation": f"clip to [-{bound}, {bound}]"
                })
                violation_count += 1

            if generate_cert:
                certificates.append({
                    "result": "PASS" if passed else "FAIL",
                    "hash": f"sha256:{i:08x}...",
                    "timestamp": "2026-05-22T13:00:00Z",
                    "verified": True,
                })

        return {
            "status": "ok",
            "result": {
                "pass": violation_count == 0,
                "checked_count": checked_count,
                "violation_count": violation_count,
                "violations": violations,
                "certificates": certificates,
                "preset_used": preset,
                "domain": domain,
            }
        }

    if task_type == "get_violations":
        since_beat = inp.get("since_beat")
        severities = inp.get("severity", ["warning", "error", "critical"])
        limit = inp.get("limit", 100)
        all_violations = [
            {
                "beat": 1400,
                "index": 1,
                "constraint": "bounds",
                "expected": 10.0,
                "actual": 12.5,
                "severity": "error",
                "domain": "neural",
                "remediation": "clip to [-10, 10]"
            },
            {
                "beat": 1410,
                "index": 3,
                "constraint": "l2_norm",
                "expected": 25.0,
                "actual": 30.0,
                "severity": "warning",
                "domain": "neural",
                "remediation": "scale by 0.83"
            },
        ]
        if since_beat is not None:
            all_violations = [v for v in all_violations if v["beat"] >= since_beat]
        if severities:
            all_violations = [v for v in all_violations if v["severity"] in severities]
        result_violations = all_violations[:limit]
        return {
            "status": "ok",
            "result": {
                "violations": result_violations,
                "total": len(result_violations),
                "unique_indices": len({v["index"] for v in result_violations}),
            }
        }

    if task_type == "apply_feedback":
        err = _validate_required(inp, ["target_id"])
        if err:
            return err
        target_id = inp["target_id"]
        chaos_delta = inp.get("chaos_delta", 0.1)
        rebirth_threshold = inp.get("rebirth_threshold", 3)
        dry_run = inp.get("dry_run", False)
        return {
            "status": "ok",
            "result": {
                "actions": [
                    {
                        "room_id": 42,
                        "action": "chaos_increase" if not dry_run else "none",
                        "chaos_before": 0.3,
                        "chaos_after": 0.3 + chaos_delta,
                        "consecutive_violations": 2,
                    }
                ],
                "rebirths_triggered": 0,
                "chaos_adjustments": 0 if dry_run else 1,
                "dry_run": dry_run,
                "target_id": target_id,
                "rebirth_threshold": rebirth_threshold,
            }
        }

    return {"status": "error", "result": {"message": f"Unknown task type: {task_type}"}}
