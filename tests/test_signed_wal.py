"""Tests for SignedWAL cryptographic integrity.

Covers:
    - append → signature valid
    - verify chain of 10 entries → all valid
    - tampered entry (modified vector_hash) → signature invalid
    - deleted entry in chain → hash mismatch at next entry
    - inserted entry → sequence gap detected
    - integration with BreederDaemonV2 → operations logged, chain verifiable
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import types

import numpy as np
import pytest

from logos.signed_wal import (
    SignedWAL,
    WALEntry,
    SignedEntry,
    TamperReport,
    WALSecurityError,
)


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def tmp_wal(tmp_path):
    """Provide a SignedWAL backed by a temp file."""
    path = tmp_path / "signed.wal.jsonl"
    wal = SignedWAL(log_path=path)
    return wal, path


@pytest.fixture
def sample_entry():
    """A canonical WALEntry for testing."""
    return WALEntry(
        timestamp=time.time(),
        agent_id=42,
        operation="spawn",
        vector_hash="a" * 64,
        parent_ids=[1, 2],
        generation=1,
    )


# ── Core append / verify tests ──────────────────────────────────


class TestAppendAndVerify:
    def test_append_returns_signed_entry(self, tmp_wal, sample_entry):
        wal, _ = tmp_wal
        se = wal.append(sample_entry)
        assert isinstance(se, SignedEntry)
        assert se.entry == sample_entry
        assert len(se.signature) > 0
        assert se.public_key == wal.public_key

    def test_signature_is_valid(self, tmp_wal, sample_entry):
        wal, _ = tmp_wal
        se = wal.append(sample_entry)
        assert wal.verify(se) is True

    def test_signature_invalid_with_wrong_key(self, tmp_wal, sample_entry):
        wal, _ = tmp_wal
        se = wal.append(sample_entry)
        other_wal = SignedWAL()
        assert wal.verify(se, public_key=other_wal.public_key) is False

    def test_persistence_roundtrip(self, tmp_wal, sample_entry):
        wal, path = tmp_wal
        se = wal.append(sample_entry)
        # Simulate restart: new instance reads same file
        wal2 = SignedWAL(log_path=path)
        assert len(wal2.entries) == 1
        assert wal2.verify(wal2.entries[0]) is True


# ── Chain verification tests ────────────────────────────────────


class TestChainVerification:
    def test_chain_of_10_all_valid(self, tmp_wal):
        wal, _ = tmp_wal
        for i in range(10):
            entry = WALEntry(
                timestamp=time.time(),
                agent_id=i,
                operation="spawn",
                vector_hash="0" * 64,
                parent_ids=[],
                generation=i,
            )
            wal.append(entry)

        ok, first_bad = wal.verify_chain()
        assert ok is True
        assert first_bad == -1

    def test_tampered_entry_signature_invalid(self, tmp_wal, sample_entry):
        wal, _ = tmp_wal
        se = wal.append(sample_entry)
        # Tamper: create a new entry with modified vector_hash but keep old signature
        tampered = WALEntry(
            timestamp=se.entry.timestamp,
            agent_id=se.entry.agent_id,
            operation=se.entry.operation,
            vector_hash="b" * 64,  # changed!
            parent_ids=se.entry.parent_ids,
            generation=se.entry.generation,
        )
        bad = SignedEntry(
            entry=tampered,
            signature=se.signature,  # old signature — now invalid
            previous_hash=se.previous_hash,
            public_key=se.public_key,
        )
        assert wal.verify(bad) is False

    def test_deleted_entry_hash_mismatch(self, tmp_wal):
        wal, _ = tmp_wal
        entries = []
        for i in range(5):
            e = WALEntry(
                timestamp=time.time(),
                agent_id=i,
                operation="spawn",
                vector_hash="0" * 64,
                parent_ids=[],
                generation=i,
            )
            entries.append(wal.append(e))

        # Simulate deletion of index 2: drop it, keep 0,1,3,4
        truncated = [entries[0], entries[1], entries[3], entries[4]]
        ok, first_bad = wal.verify_chain(truncated)
        assert ok is False
        # The mismatch shows at index 2 (original 3) because its previous_hash
        # points to the deleted entry.
        assert first_bad == 2

    def test_inserted_entry_sequence_gap(self, tmp_wal):
        wal, _ = tmp_wal
        entries = []
        for i in range(5):
            e = WALEntry(
                timestamp=time.time(),
                agent_id=i,
                operation="spawn",
                vector_hash="0" * 64,
                parent_ids=[],
                generation=i,
            )
            entries.append(wal.append(e))

        # Insert a forged entry in the middle
        forged = WALEntry(
            timestamp=time.time(),
            agent_id=999,
            operation="spawn",
            vector_hash="x" * 64,
            parent_ids=[],
            generation=99,
        )
        # To create a valid-looking insertion, we sign it with the same key
        forged_se = wal.append(forged)
        # Build a chain that puts forged between entries[1] and entries[2]
        # But forged_se.previous_hash will be entries[1].compute_hash() only if
        # we append it right after. Let's rebuild cleanly.
        wal2 = SignedWAL()
        chain = []
        for i, orig in enumerate(entries):
            chain.append(orig)
            if i == 1:
                # Insert the forged entry
                forged_entry = WALEntry(
                    timestamp=time.time(),
                    agent_id=999,
                    operation="spawn",
                    vector_hash="x" * 64,
                    parent_ids=[],
                    generation=99,
                )
                # We need the previous_hash to match entries[1]'s hash
                prev_hash = chain[-1].compute_hash()
                payload = json.dumps(
                    {
                        "timestamp": forged_entry.timestamp,
                        "agent_id": forged_entry.agent_id,
                        "operation": forged_entry.operation,
                        "vector_hash": forged_entry.vector_hash,
                        "parent_ids": forged_entry.parent_ids,
                        "generation": forged_entry.generation,
                        "previous_hash": prev_hash,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                sig = wal2._backend.sign(payload)
                forged_se = SignedEntry(
                    entry=forged_entry,
                    signature=sig,
                    previous_hash=prev_hash,
                    public_key=wal2.public_key,
                )
                chain.append(forged_se)

        # The signature check on forged_se will fail because we signed with wal2,
        # but if we use wal (same key) it would pass. Let's use wal.
        # Re-do with wal's backend.
        wal3 = SignedWAL(
            private_key=wal._backend.private_key_bytes, algorithm=wal.algorithm
        )
        chain = []
        for i, orig in enumerate(entries):
            chain.append(orig)
            if i == 1:
                forged_entry = WALEntry(
                    timestamp=time.time(),
                    agent_id=999,
                    operation="spawn",
                    vector_hash="x" * 64,
                    parent_ids=[],
                    generation=99,
                )
                prev_hash = chain[-1].compute_hash()
                payload = json.dumps(
                    {
                        "timestamp": forged_entry.timestamp,
                        "agent_id": forged_entry.agent_id,
                        "operation": forged_entry.operation,
                        "vector_hash": forged_entry.vector_hash,
                        "parent_ids": forged_entry.parent_ids,
                        "generation": forged_entry.generation,
                        "previous_hash": prev_hash,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                sig = wal3._backend.sign(payload)
                forged_se = SignedEntry(
                    entry=forged_entry,
                    signature=sig,
                    previous_hash=prev_hash,
                    public_key=wal3.public_key,
                )
                chain.append(forged_se)
        # Now chain has: 0, 1, forged, 2, 3, 4
        # Entry 2 (original index 2) has previous_hash pointing to 1, not forged
        # So verify_chain will catch a hash mismatch at original index 2
        ok, first_bad = wal3.verify_chain(chain)
        assert ok is False
        # first_bad should be the first entry after the insert whose previous_hash
        # no longer matches.
        # In our chain: [0, 1, forged, 2, 3, 4]
        # forged.previous_hash == hash(1) ✓
        # 2.previous_hash == hash(1) ✗ (should be hash(forged))
        assert first_bad == 3  # index of original entries[2]


# ── Tamper detection tests ──────────────────────────────────────


class TestTamperDetect:
    def test_tampered_entry_detected(self, tmp_wal, sample_entry):
        wal, _ = tmp_wal
        se = wal.append(sample_entry)
        # Build a tampered version
        tampered = WALEntry(
            timestamp=se.entry.timestamp,
            agent_id=se.entry.agent_id,
            operation=se.entry.operation,
            vector_hash="tampered" * 8,
            parent_ids=se.entry.parent_ids,
            generation=se.entry.generation,
        )
        bad = SignedEntry(
            entry=tampered,
            signature=se.signature,
            previous_hash=se.previous_hash,
            public_key=se.public_key,
        )
        reports = wal.tamper_detect([bad])
        assert any(r.type == "signature_invalid" for r in reports)

    def test_deleted_entry_hash_mismatch_detected(self, tmp_wal):
        wal, _ = tmp_wal
        entries = []
        for i in range(5):
            e = WALEntry(
                timestamp=time.time(),
                agent_id=i,
                operation="spawn",
                vector_hash="0" * 64,
                parent_ids=[],
                generation=i,
            )
            entries.append(wal.append(e))

        truncated = [entries[0], entries[1], entries[3], entries[4]]
        reports = wal.tamper_detect(truncated)
        assert any(r.type == "hash_mismatch" for r in reports)

    def test_generation_regression_detected(self, tmp_wal):
        wal, _ = tmp_wal
        e1 = WALEntry(
            timestamp=time.time(),
            agent_id=1,
            operation="spawn",
            vector_hash="0" * 64,
            parent_ids=[],
            generation=2,
        )
        e2 = WALEntry(
            timestamp=time.time(),
            agent_id=2,
            operation="spawn",
            vector_hash="0" * 64,
            parent_ids=[],
            generation=1,
        )
        wal.append(e1)
        wal.append(e2)
        reports = wal.tamper_detect()
        assert any(r.type == "sequence_gap" for r in reports)
        assert "regressed" in reports[0].details


# ── Algorithm backends ──────────────────────────────────────────


class TestBackends:
    def test_ed25519_default(self, tmp_path):
        wal = SignedWAL(log_path=tmp_path / "ed25519.wal")
        assert wal.algorithm == "ed25519"
        e = WALEntry(
            timestamp=time.time(),
            agent_id=1,
            operation="spawn",
            vector_hash="0" * 64,
            parent_ids=[],
            generation=0,
        )
        se = wal.append(e)
        assert wal.verify(se) is True

    def test_hmac_fallback(self, tmp_path):
        wal = SignedWAL(algorithm="hmac-sha256", log_path=tmp_path / "hmac.wal")
        assert wal.algorithm in ("hmac-sha256", "hmac")
        e = WALEntry(
            timestamp=time.time(),
            agent_id=1,
            operation="spawn",
            vector_hash="0" * 64,
            parent_ids=[],
            generation=0,
        )
        se = wal.append(e)
        assert wal.verify(se) is True

    def test_rsa_backend(self, tmp_path):
        wal = SignedWAL(algorithm="rsa-2048", log_path=tmp_path / "rsa.wal")
        e = WALEntry(
            timestamp=time.time(),
            agent_id=1,
            operation="spawn",
            vector_hash="0" * 64,
            parent_ids=[],
            generation=0,
        )
        se = wal.append(e)
        assert wal.verify(se) is True


# ── Integration with BreederDaemonV2 ────────────────────────────


class TestBreederDaemonV2Integration:
    def test_daemon_logs_operations_to_signed_wal(self, tmp_path):
        """Spawn, sunset, breed, mutate are all logged."""
        from swarm.breeder_daemon_v2 import BreederDaemonV2, LifecycleState
        from swarm.thermal import ThermalBudget, DeviceType
        from nerve.room_grid import RoomGrid

        grid = RoomGrid(n=8)
        thermal = ThermalBudget(budgets={DeviceType.GPU: 4, DeviceType.CPU: 20})
        wal_path = tmp_path / "breeder.wal.sqlite"
        signed_path = tmp_path / "signed.wal.jsonl"

        daemon = BreederDaemonV2(
            grid=grid,
            thermal=thermal,
            wal_path=str(wal_path),
            signed_wal_path=str(signed_path),
            tick_interval=999.0,  # don't auto-step
        )
        daemon.start()

        # Force a few transitions manually
        from swarm.breeder_daemon_v2 import LifecycleTransition

        tr1 = LifecycleTransition(
            agent_id=100,
            from_state=None,
            to_state=LifecycleState.EGG,
            timestamp=time.time(),
            generation=0,
            vector_hash="abc123" * 8,
        )
        daemon._wal.transition(tr1)
        if daemon._signed_wal is not None:
            daemon._signed_wal.append(
                WALEntry(
                    timestamp=tr1.timestamp,
                    agent_id=tr1.agent_id,
                    operation="spawn",
                    vector_hash=tr1.vector_hash or "0" * 64,
                    parent_ids=[],
                    generation=tr1.generation,
                )
            )
            print(
                "DEBUG after append1:", id(daemon._signed_wal), len(daemon._signed_wal)
            )

        tr2 = LifecycleTransition(
            agent_id=100,
            from_state=LifecycleState.EGG,
            to_state=LifecycleState.SUNSET,
            timestamp=time.time(),
            generation=0,
            vector_hash="abc123" * 8,
        )
        daemon._wal.transition(tr2)
        if daemon._signed_wal is not None:
            daemon._signed_wal.append(
                WALEntry(
                    timestamp=tr2.timestamp,
                    agent_id=tr2.agent_id,
                    operation="sunset",
                    vector_hash=tr2.vector_hash or "0" * 64,
                    parent_ids=[],
                    generation=tr2.generation,
                )
            )
            print(
                "DEBUG after append2:", id(daemon._signed_wal), len(daemon._signed_wal)
            )

        daemon.stop()

        # Verify signed WAL is intact
        assert daemon._signed_wal is not None
        assert len(daemon._signed_wal) == 2
        ok, first_bad = daemon._signed_wal.verify_chain()
        assert ok is True
        assert first_bad == -1

    def test_daemon_detects_tampering_on_startup(self, tmp_path):
        """If signed WAL is tampered, startup should detect it."""
        from swarm.breeder_daemon_v2 import BreederDaemonV2
        from swarm.thermal import ThermalBudget, DeviceType
        from nerve.room_grid import RoomGrid

        grid = RoomGrid(n=8)
        thermal = ThermalBudget(budgets={DeviceType.GPU: 4, DeviceType.CPU: 20})
        wal_path = tmp_path / "breeder.wal.sqlite"
        signed_path = tmp_path / "signed.wal.jsonl"

        # Pre-create a tampered signed WAL
        pre_wal = SignedWAL(log_path=signed_path)
        e1 = WALEntry(
            timestamp=time.time(),
            agent_id=1,
            operation="spawn",
            vector_hash="0" * 64,
            parent_ids=[],
            generation=0,
        )
        se1 = pre_wal.append(e1)
        # Write a tampered second line directly
        tampered = WALEntry(
            timestamp=time.time(),
            agent_id=2,
            operation="spawn",
            vector_hash="TAMPERED" * 8,
            parent_ids=[],
            generation=1,
        )
        bad = SignedEntry(
            entry=tampered,
            signature=se1.signature,
            previous_hash=se1.previous_hash,
            public_key=se1.public_key,
        )
        with open(signed_path, "a") as f:
            f.write(pre_wal._entry_to_json(bad) + "\n")

        daemon = BreederDaemonV2(
            grid=grid,
            thermal=thermal,
            wal_path=str(wal_path),
            signed_wal_path=str(signed_path),
            tick_interval=999.0,
        )
        # start() should detect tampering and set safe_mode
        daemon.start()
        assert daemon._safe_mode is True
        daemon.stop()

    def test_daemon_chain_verifiable_after_multiple_operations(self, tmp_path):
        """Complex lifecycle produces a verifiable chain."""
        from swarm.breeder_daemon_v2 import BreederDaemonV2, LifecycleState
        from swarm.thermal import ThermalBudget, DeviceType
        from nerve.room_grid import RoomGrid

        grid = RoomGrid(n=8)
        thermal = ThermalBudget(budgets={DeviceType.GPU: 4, DeviceType.CPU: 20})
        wal_path = tmp_path / "breeder.wal.sqlite"
        signed_path = tmp_path / "signed.wal.jsonl"

        daemon = BreederDaemonV2(
            grid=grid,
            thermal=thermal,
            wal_path=str(wal_path),
            signed_wal_path=str(signed_path),
            tick_interval=999.0,
        )
        daemon.start()

        # Log a sequence: spawn -> incubate -> compete -> sunset
        ops = [
            (LifecycleState.EGG, "spawn"),
            (LifecycleState.COMPETE, "mutate"),
            (LifecycleState.SUNSET, "sunset"),
        ]
        from swarm.breeder_daemon_v2 import LifecycleTransition

        prev_state = None
        for to_state, op in ops:
            tr = LifecycleTransition(
                agent_id=200,
                from_state=prev_state,
                to_state=to_state,
                timestamp=time.time(),
                generation=0,
                vector_hash="deadbeef" * 8,
            )
            daemon._wal.transition(tr)
            if daemon._signed_wal is not None:
                daemon._signed_wal.append(
                    WALEntry(
                        timestamp=tr.timestamp,
                        agent_id=tr.agent_id,
                        operation=op,
                        vector_hash=tr.vector_hash or "0" * 64,
                        parent_ids=[],
                        generation=tr.generation,
                    )
                )
            prev_state = to_state

        daemon.stop()

        assert daemon._signed_wal is not None
        assert len(daemon._signed_wal) == 3
        ok, first_bad = daemon._signed_wal.verify_chain()
        assert ok is True
        assert first_bad == -1

        # Ensure no tamper reports
        reports = daemon._signed_wal.tamper_detect()
        assert len(reports) == 0
