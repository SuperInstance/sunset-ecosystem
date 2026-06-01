"""fleet/plato_sdk_bridge.py — Bridge to Cocapn PLATO SDK.

Wraps the cocapn-plato Python client for sunset-ecosystem integration.
Provides tile querying, submission, and fleet-aware domain mapping.

References
----------
- SuperInstance/cocapn-plato-check  v3.2.0
- fleet/plato_sync.py               (existing PLATO sync layer)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Try to import the official SDK; fall back to lightweight urllib
_PLATO_SDK_AVAILABLE = False

try:
    from cocapn_plato import PlatoClient as _SdkPlatoClient
    _PLATO_SDK_AVAILABLE = True
except Exception:
    pass


# ── Lightweight fallback client (mirrors SDK API) ────────────────────────

class _FallbackPlatoClient:
    """Lightweight urllib-based PLATO client when SDK not installed."""

    def __init__(self, base_url: str = "http://localhost:8847", timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        import urllib.request
        url = f"{self.base_url}{path}"
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if data and method in ("POST", "PUT", "PATCH"):
            body = json.dumps(data).encode()
            req = urllib.request.Request(url, data=body, headers=headers, method=method)
        else:
            req = urllib.request.Request(url, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            return {"status": "error", "reason": str(e)}

    def query(self, **kwargs) -> List[Dict[str, Any]]:
        result = self._request("POST", "/query", kwargs)
        return result.get("results", [])

    def get_tile(self, domain: str, question: Optional[str] = None) -> Optional[Dict[str, Any]]:
        where = {"domain": domain}
        if question:
            where["question"] = {"op": "regex", "val": question}
        results = self.query(table="tiles", where=where, limit=1)
        return results[0] if results else None

    def list_domains(self) -> List[str]:
        result = self._request("POST", "/aggregate", {"table": "tiles", "group_by": "domain"})
        if isinstance(result, list) and result and "_key" in result[0]:
            return [r["_key"] for r in result]
        qr = self.query(limit=500)
        domains = {t.get("domain") for t in qr if t.get("domain")}
        return sorted(domains)

    def health(self) -> Dict[str, Any]:
        return self._request("GET", "/health")

    def submit(self, agent: str, question: str, answer: str, domain: str = "general") -> Dict[str, Any]:
        return self._request("POST", "/submit", {
            "agent": agent,
            "question": question,
            "answer": answer,
            "domain": domain,
        })


# ── Public bridge ───────────────────────────────────────────────────────

@dataclass
class TileResult:
    """Typed tile query result."""
    domain: str
    question: str
    answer: str
    metadata: Dict[str, Any]


class PlatoSDKBridge:
    """Bridge between sunset-ecosystem and Cocapn PLATO.

    Uses the official cocapn-plato SDK when available, otherwise falls back
    to a lightweight urllib implementation with identical API.
    """

    def __init__(self, base_url: str = "http://localhost:8847", timeout: float = 10.0):
        self._has_sdk = _PLATO_SDK_AVAILABLE
        self._client = _SdkPlatoClient(base_url, timeout) if _PLATO_SDK_AVAILABLE else _FallbackPlatoClient(base_url, timeout)
        self.base_url = base_url

    @property
    def backend_name(self) -> str:
        return "cocapn-plato-sdk" if self._has_sdk else "urllib-fallback"

    def health(self) -> Dict[str, Any]:
        return self._client.health()

    def list_domains(self) -> List[str]:
        return self._client.list_domains()

    def get_tile(self, domain: str, question: Optional[str] = None) -> Optional[TileResult]:
        raw = self._client.get_tile(domain, question)
        if not raw:
            return None
        return TileResult(
            domain=raw.get("domain", domain),
            question=raw.get("question", ""),
            answer=raw.get("answer", ""),
            metadata={k: v for k, v in raw.items() if k not in ("domain", "question", "answer")},
        )

    def query_tiles(self, domain: Optional[str] = None, limit: int = 50) -> List[TileResult]:
        where = {"domain": domain} if domain else None
        raw_results = self._client.query(table="tiles", where=where, limit=limit)
        return [
            TileResult(
                domain=r.get("domain", ""),
                question=r.get("question", ""),
                answer=r.get("answer", ""),
                metadata={k: v for k, v in r.items() if k not in ("domain", "question", "answer")},
            )
            for r in raw_results
        ]

    def submit_tile(self, agent: str, question: str, answer: str, domain: str = "general") -> Dict[str, Any]:
        return self._client.submit(agent, question, answer, domain)

    def __repr__(self) -> str:
        return f"PlatoSDKBridge(url={self.base_url}, backend={self.backend_name})"


__all__ = [
    "PlatoSDKBridge",
    "TileResult",
]
