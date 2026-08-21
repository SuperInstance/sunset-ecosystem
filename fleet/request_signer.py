"""Request signing with HMAC and signature verification.

Signs requests using HMAC-SHA256 and verifies signatures. Supports
timestamp-based replay protection and custom headers. Used for fleet
API authentication, webhook verification, and secure inter-service
communication.

Usage:
    signer = RequestSigner(secret="my-secret-key")
    signature = signer.sign("/api/users", method="POST", body=b"data")
    assert signer.verify("/api/users", method="POST", body=b"data", signature=signature)
"""

from __future__ import annotations

import hmac
import hashlib
import time
from typing import Any, Dict, List, Optional


class RequestSigner:
    """
    Request signer with HMAC-SHA256.

    :param secret: Signing secret key.
    :param ttl_sec: Optional TTL for signature validity (replay protection).
    :param clock: Optional clock function for testing.
    """

    def __init__(
        self,
        secret: str,
        ttl_sec: Optional[float] = None,
        clock: Optional[callable] = None,
    ):
        self._secret = secret.encode()
        self._ttl = ttl_sec
        self._clock = clock or time.time

    # ------------------------------------------------------------------
    # Signing
    # ------------------------------------------------------------------

    def sign(
        self,
        path: str,
        method: str = "GET",
        body: Optional[bytes] = None,
        headers: Optional[Dict[str, str]] = None,
        timestamp: Optional[float] = None,
    ) -> str:
        """
        Sign a request.

        :param path: Request path.
        :param method: HTTP method.
        :param body: Request body bytes.
        :param headers: Additional headers to include in signature.
        :param timestamp: Optional timestamp (uses now if None).
        :returns: Signature string.
        """
        ts = timestamp if timestamp is not None else self._clock()
        # Normalize float timestamps that are whole numbers back to int
        # so verify() signatures match sign() signatures exactly.
        if isinstance(ts, float) and ts.is_integer():
            ts = int(ts)
        payload = self._build_payload(path, method, body, headers, ts)
        sig = hmac.new(self._secret, payload.encode(), hashlib.sha256).hexdigest()
        return f"{ts}:{sig}"

    def _build_payload(
        self,
        path: str,
        method: str,
        body: Optional[bytes],
        headers: Optional[Dict[str, str]],
        timestamp: float,
    ) -> str:
        """Build canonical payload string for signing."""
        parts = [method.upper(), path, str(timestamp)]
        if body:
            parts.append(hashlib.sha256(body).hexdigest())
        if headers:
            header_str = "|".join(f"{k}={v}" for k, v in sorted(headers.items()))
            parts.append(header_str)
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify(
        self,
        path: str,
        method: str = "GET",
        body: Optional[bytes] = None,
        headers: Optional[Dict[str, str]] = None,
        signature: str = "",
    ) -> bool:
        """
        Verify a request signature.

        :param path: Request path.
        :param method: HTTP method.
        :param body: Request body bytes.
        :param headers: Additional headers.
        :param signature: Signature to verify.
        :returns: True if signature is valid.
        """
        try:
            ts_str, sig = signature.split(":", 1)
            timestamp = float(ts_str)
        except ValueError:
            return False

        # Check TTL
        if self._ttl is not None:
            if self._clock() - timestamp > self._ttl:
                return False

        expected = self.sign(path, method, body, headers, timestamp)
        return hmac.compare_digest(expected, signature)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "ttl": self._ttl,
        }

    def __repr__(self) -> str:
        return f"<RequestSigner ttl={self._ttl}>"
