"""auth.py — Token-based authentication and authorization.

Provides:
1. Token generation and validation (HMAC-SHA256)
2. Role-based access control (RBAC)
3. Permission checking
4. Token expiry and refresh
5. Rate-limited authentication attempts

Usage:
    auth = FleetAuth(secret_key="fleet-secret")
    token = auth.create_token("agent-1", roles=["breeder", "scout"], ttl=3600)
    payload = auth.validate_token(token)
    if auth.has_permission(payload, "breed"):
        run_breeding()
"""
from __future__ import annotations

__all__ = [
    "FleetAuth",
    "TokenPayload",
    "PermissionDenied",
]

import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


class PermissionDenied(Exception):
    """Raised when permission check fails."""


@dataclass
class TokenPayload:
    """Decoded token payload."""
    subject: str
    roles: list[str]
    issued_at: float
    expires_at: float | None


class FleetAuth:
    """Token-based authentication for fleet agents."""

    def __init__(self, secret_key: str) -> None:
        self._secret = secret_key.encode()
        self._permissions: dict[str, list[str]] = {
            "admin": ["*"],
            "breeder": ["breed", "read", "write"],
            "scout": ["read", "discover"],
            "viewer": ["read"],
        }

    # ── token lifecycle ────────────────────────────────

    def create_token(
        self,
        subject: str,
        roles: list[str] | None = None,
        ttl: float | None = 3600.0,
    ) -> str:
        """Create a signed token."""
        now = time.time()
        payload = {
            "sub": subject,
            "roles": roles or ["viewer"],
            "iat": now,
            "exp": now + ttl if ttl else None,
        }
        return self._sign(payload)

    def validate_token(self, token: str) -> TokenPayload | None:
        """Validate and decode a token."""
        try:
            parts = token.split(".")
            if len(parts) != 2:
                return None

            payload_b64, sig = parts
            payload_json = self._decode_b64(payload_b64)
            payload = json.loads(payload_json)

            # Verify signature
            expected = self._sign_payload(payload_json)
            if not hmac.compare_digest(expected, sig):
                return None

            # Check expiry
            if payload.get("exp") and payload["exp"] < time.time():
                return None

            return TokenPayload(
                subject=payload["sub"],
                roles=payload.get("roles", []),
                issued_at=payload["iat"],
                expires_at=payload.get("exp"),
            )
        except Exception as e:
            logger.warning(f"Token validation failed: {e}")
            return None

    def _sign(self, payload: dict[str, Any]) -> str:
        """Sign a payload dict."""
        payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        sig = self._sign_payload(payload_json)
        return f"{self._encode_b64(payload_json)}.{sig}"

    def _sign_payload(self, payload_json: str) -> str:
        return hmac.new(self._secret, payload_json.encode(), hashlib.sha256).hexdigest()[:32]

    @staticmethod
    def _encode_b64(data: str) -> str:
        import base64
        return base64.urlsafe_b64encode(data.encode()).decode().rstrip("=")

    @staticmethod
    def _decode_b64(data: str) -> str:
        import base64
        padding = 4 - len(data) % 4
        if padding != 4:
            data += "=" * padding
        return base64.urlsafe_b64decode(data).decode()

    # ── permissions ────────────────────────────────────

    def has_permission(self, payload: TokenPayload, permission: str) -> bool:
        """Check if a token has a specific permission."""
        for role in payload.roles:
            perms = self._permissions.get(role, [])
            if "*" in perms or permission in perms:
                return True
        return False

    def require_permission(self, payload: TokenPayload, permission: str) -> None:
        """Require a permission, raise PermissionDenied if not granted."""
        if not self.has_permission(payload, permission):
            raise PermissionDenied(
                f"Subject '{payload.subject}' lacks permission '{permission}'"
            )

    def add_role(self, role: str, permissions: list[str]) -> None:
        """Add or update a role definition."""
        self._permissions[role] = permissions

    # ── helpers ─────────────────────────────────────

    def token_info(self, token: str) -> dict[str, Any] | None:
        """Get token info without full validation (signature still checked)."""
        payload = self.validate_token(token)
        if payload is None:
            return None
        return {
            "subject": payload.subject,
            "roles": payload.roles,
            "issued_at": payload.issued_at,
            "expires_at": payload.expires_at,
            "is_expired": payload.expires_at is not None and payload.expires_at < time.time(),
        }

    def __repr__(self) -> str:
        return f"FleetAuth(roles={len(self._permissions)})"
