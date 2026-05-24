"""Signed WAL — Write-Ahead Log with cryptographic integrity.

Detects tampering with agent state history via Ed25519 signatures and
a chained hash that links each entry to its predecessor.

Experiment: Signed WAL (RESEARCH_SECURITY.md)
"""

from __future__ import annotations

__all__ = [
    "SignedWAL",
    "WALEntry",
    "SignedEntry",
    "TamperReport",
    "WALSecurityError",
]

import hashlib
import hmac
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Data Structures ───────────────────────────────────────────────


@dataclass(frozen=True)
class WALEntry:
    """A single logical operation in the fleet lifecycle."""

    timestamp: float
    agent_id: int
    operation: str  # 'spawn', 'sunset', 'breed', 'mutate', 'signal'
    vector_hash: str  # SHA-256 of agent vector
    parent_ids: list[int]
    generation: int


@dataclass(frozen=True)
class SignedEntry:
    """A WALEntry plus cryptographic proof."""

    entry: WALEntry
    signature: bytes
    previous_hash: str  # SHA-256 of the *previous* SignedEntry (or "" for genesis)
    public_key: bytes

    def compute_hash(self) -> str:
        """Deterministic hash of this signed entry (used for chaining)."""
        payload = json.dumps(
            {
                "entry": {
                    "timestamp": self.entry.timestamp,
                    "agent_id": self.entry.agent_id,
                    "operation": self.entry.operation,
                    "vector_hash": self.entry.vector_hash,
                    "parent_ids": self.entry.parent_ids,
                    "generation": self.entry.generation,
                },
                "signature": self.signature.hex(),
                "previous_hash": self.previous_hash,
                "public_key": self.public_key.hex(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TamperReport:
    """A single detected anomaly in the chain."""

    index: int
    type: str  # 'signature_invalid' | 'hash_mismatch' | 'sequence_gap'
    details: str


class WALSecurityError(Exception):
    """Raised when chain integrity cannot be verified."""


# ── Crypto Backends ─────────────────────────────────────────────────


class _Ed25519Backend:
    """Preferred: lightweight, fast Ed25519 via cryptography."""

    def __init__(self, private_key: bytes | None = None):
        from cryptography.hazmat.primitives.asymmetric import ed25519

        if private_key is None:
            self._sk = ed25519.Ed25519PrivateKey.generate()
        else:
            self._sk = ed25519.Ed25519PrivateKey.from_private_bytes(private_key)
        self._pk = self._sk.public_key()

    @property
    def private_key_bytes(self) -> bytes:
        return self._sk.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )

    @property
    def public_key_bytes(self) -> bytes:
        return self._pk.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

    def sign(self, message: bytes) -> bytes:
        return self._sk.sign(message)

    def verify(self, message: bytes, signature: bytes, public_key: bytes | None = None) -> bool:
        from cryptography.hazmat.primitives.asymmetric import ed25519
        from cryptography.exceptions import InvalidSignature

        pk = self._pk
        if public_key is not None:
            pk = ed25519.Ed25519PublicKey.from_public_bytes(public_key)
        try:
            pk.verify(signature, message)
            return True
        except InvalidSignature:
            return False


class _HMACBackend:
    """Fallback: HMAC-SHA256 with a shared secret."""

    def __init__(self, secret: bytes | None = None):
        self._secret = secret or os.urandom(32)

    @property
    def private_key_bytes(self) -> bytes:
        return self._secret

    @property
    def public_key_bytes(self) -> bytes:
        # HMAC is symmetric — public key == private key (not ideal, but explicit)
        return self._secret

    def sign(self, message: bytes) -> bytes:
        return hmac.new(self._secret, message, hashlib.sha256).digest()

    def verify(self, message: bytes, signature: bytes, public_key: bytes | None = None) -> bool:
        expected = self.sign(message)
        # In symmetric mode, if a different public_key is supplied we re-key
        if public_key is not None and public_key != self._secret:
            expected = hmac.new(public_key, message, hashlib.sha256).digest()
        return hmac.compare_digest(expected, signature)


# ── Signed WAL ────────────────────────────────────────────────────


class SignedWAL:
    """Write-Ahead Log with cryptographic integrity.

    Args:
        private_key: Optional private key bytes. If None, generates one.
        algorithm: 'ed25519' (default) or 'rsa-2048' or 'hmac-sha256'.
        log_path: Optional path to persist entries as newline-delimited JSON.
    """

    def __init__(
        self,
        private_key: bytes | None = None,
        algorithm: str = "ed25519",
        log_path: str | Path | None = None,
    ) -> None:
        self.algorithm = algorithm
        self.log_path = Path(log_path) if log_path else None

        if algorithm == "ed25519":
            self._backend = _Ed25519Backend(private_key)
        elif algorithm in ("rsa-2048", "rsa"):
            self._backend = _RSABackend(private_key)
        elif algorithm in ("hmac-sha256", "hmac"):
            self._backend = _HMACBackend(private_key)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        self._entries: list[SignedEntry] = []
        self._last_hash: str = ""
        self._public_key: bytes = self._backend.public_key_bytes

        if self.log_path:
            self._load_log()

    # ── Serialization helpers ─────────────────────────────

    def _entry_to_json(self, se: SignedEntry) -> str:
        return json.dumps(
            {
                "entry": {
                    "timestamp": se.entry.timestamp,
                    "agent_id": se.entry.agent_id,
                    "operation": se.entry.operation,
                    "vector_hash": se.entry.vector_hash,
                    "parent_ids": se.entry.parent_ids,
                    "generation": se.entry.generation,
                },
                "signature": se.signature.hex(),
                "previous_hash": se.previous_hash,
                "public_key": se.public_key.hex(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def _json_to_entry(self, raw: str) -> SignedEntry:
        data = json.loads(raw)
        entry = WALEntry(
            timestamp=data["entry"]["timestamp"],
            agent_id=data["entry"]["agent_id"],
            operation=data["entry"]["operation"],
            vector_hash=data["entry"]["vector_hash"],
            parent_ids=data["entry"]["parent_ids"],
            generation=data["entry"]["generation"],
        )
        return SignedEntry(
            entry=entry,
            signature=bytes.fromhex(data["signature"]),
            previous_hash=data["previous_hash"],
            public_key=bytes.fromhex(data["public_key"]),
        )

    def _load_log(self) -> None:
        if not self.log_path or not self.log_path.exists():
            return
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    se = self._json_to_entry(line)
                    self._entries.append(se)
                    self._last_hash = se.compute_hash()
                except Exception as exc:
                    logger.warning("Skipping corrupt WAL line: %s", exc)
        logger.info("Loaded %d entries from %s", len(self._entries), self.log_path)

    def _persist(self, se: SignedEntry) -> None:
        if not self.log_path:
            return
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(self._entry_to_json(se) + "\n")
            f.flush()
            os.fsync(f.fileno())

    # ── Public API ────────────────────────────────────────

    def append(self, entry: WALEntry) -> SignedEntry:
        """Append an entry with signature and hash chain."""
        payload = json.dumps(
            {
                "timestamp": entry.timestamp,
                "agent_id": entry.agent_id,
                "operation": entry.operation,
                "vector_hash": entry.vector_hash,
                "parent_ids": entry.parent_ids,
                "generation": entry.generation,
                "previous_hash": self._last_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

        signature = self._backend.sign(payload)
        se = SignedEntry(
            entry=entry,
            signature=signature,
            previous_hash=self._last_hash,
            public_key=self._public_key,
        )
        self._entries.append(se)
        self._last_hash = se.compute_hash()
        self._persist(se)
        logger.debug("SignedWAL append #%d: agent=%d op=%s", len(self._entries), entry.agent_id, entry.operation)
        return se

    def verify(self, entry: SignedEntry, public_key: bytes | None = None) -> bool:
        """Verify a single entry's signature.

        If *public_key* is not provided, uses the embedded key.
        """
        payload = json.dumps(
            {
                "timestamp": entry.entry.timestamp,
                "agent_id": entry.entry.agent_id,
                "operation": entry.entry.operation,
                "vector_hash": entry.entry.vector_hash,
                "parent_ids": entry.entry.parent_ids,
                "generation": entry.entry.generation,
                "previous_hash": entry.previous_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

        pk = public_key if public_key is not None else entry.public_key
        return self._backend.verify(payload, entry.signature, pk)

    def verify_chain(self, entries: list[SignedEntry] | None = None) -> tuple[bool, int]:
        """Verify entire chain.

        Returns (all_valid, first_invalid_index).
        Also verifies hash chain: each entry's previous_hash must match
        the SHA-256 of the preceding entry.
        """
        if entries is None:
            entries = self._entries

        prev_hash = ""
        for i, se in enumerate(entries):
            # 1. Signature must be valid
            if not self.verify(se):
                logger.error("Chain verify: signature invalid at index %d", i)
                return False, i

            # 2. Hash chain must link correctly
            if se.previous_hash != prev_hash:
                logger.error(
                    "Chain verify: hash mismatch at index %d "
                    "(expected %r, got %r)",
                    i, prev_hash, se.previous_hash,
                )
                return False, i

            prev_hash = se.compute_hash()

        return True, -1

    def tamper_detect(self, entries: list[SignedEntry] | None = None) -> list[TamperReport]:
        """Detect specific tampering patterns.

        Patterns detected:
            - signature_invalid: entry signature does not verify
            - hash_mismatch: previous_hash doesn't match predecessor
            - sequence_gap: missing index (detected by generation jump or
              explicit gap check if we had sequence numbers)
        """
        if entries is None:
            entries = self._entries

        reports: list[TamperReport] = []
        prev_hash = ""
        prev_generation = -1

        for i, se in enumerate(entries):
            # 1. Signature invalid
            if not self.verify(se):
                reports.append(
                    TamperReport(
                        index=i,
                        type="signature_invalid",
                        details=f"Signature verification failed for agent {se.entry.agent_id} op={se.entry.operation}",
                    )
                )

            # 2. Hash mismatch (chain break)
            if se.previous_hash != prev_hash:
                reports.append(
                    TamperReport(
                        index=i,
                        type="hash_mismatch",
                        details=(
                            f"Hash chain broken at index {i}: "
                            f"expected previous_hash={prev_hash[:16]}..., got {se.previous_hash[:16]}..."
                        ),
                    )
                )

            # 3. Sequence gap (heuristic: generation should not jump backwards)
            #    Also flag if generation jumps >1 without explicit breed operation
            if prev_generation >= 0:
                gen_delta = se.entry.generation - prev_generation
                if gen_delta < 0:
                    reports.append(
                        TamperReport(
                            index=i,
                            type="sequence_gap",
                            details=(
                                f"Generation regressed at index {i}: "
                                f"{prev_generation} → {se.entry.generation}"
                            ),
                        )
                    )
                elif gen_delta > 1 and se.entry.operation != "breed":
                    # Large generation jump without a breed operation is suspicious
                    reports.append(
                        TamperReport(
                            index=i,
                            type="sequence_gap",
                            details=(
                                f"Suspicious generation jump at index {i}: "
                                f"{prev_generation} → {se.entry.generation} (op={se.entry.operation})"
                            ),
                        )
                    )

            prev_hash = se.compute_hash()
            prev_generation = se.entry.generation

        return reports

    # ── Properties ────────────────────────────────────────

    @property
    def entries(self) -> list[SignedEntry]:
        return list(self._entries)

    @property
    def public_key(self) -> bytes:
        return self._public_key

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"SignedWAL(alg={self.algorithm}, entries={len(self._entries)}, path={self.log_path})"


# ── RSA Backend (included for spec completeness) ──────────────────


class _RSABackend:
    """RSA-2048 backend via cryptography library.
    Heavier than Ed25519; included for spec completeness."""

    def __init__(self, private_key: bytes | None = None):
        from cryptography.hazmat.primitives.asymmetric import rsa, padding
        from cryptography.hazmat.primitives import hashes, serialization

        if private_key is None:
            self._sk = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        else:
            self._sk = serialization.load_pem_private_key(private_key, password=None)
        self._pk = self._sk.public_key()
        self._padding = padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH)
        self._hash = hashes.SHA256()

    @property
    def private_key_bytes(self) -> bytes:
        from cryptography.hazmat.primitives import serialization
        return self._sk.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

    @property
    def public_key_bytes(self) -> bytes:
        from cryptography.hazmat.primitives import serialization
        return self._pk.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

    def sign(self, message: bytes) -> bytes:
        return self._sk.sign(message, self._padding, self._hash)

    def verify(self, message: bytes, signature: bytes, public_key: bytes | None = None) -> bool:
        from cryptography.hazmat.primitives.asymmetric import padding
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.exceptions import InvalidSignature

        pk = self._pk
        if public_key is not None:
            pk = serialization.load_pem_public_key(public_key)
        try:
            pk.verify(signature, message, self._padding, self._hash)
            return True
        except InvalidSignature:
            return False


# Deferred import for serialization in Ed25519Backend
from cryptography.hazmat.primitives import serialization
