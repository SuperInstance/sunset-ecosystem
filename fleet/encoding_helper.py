"""Base64, hex, and URL encoding helpers.

Provides encoding/decoding utilities for common fleet data formats.
Used for serialization, URL-safe encoding, and binary-to-text conversion.

Usage:
    enc = EncodingHelper()
    b64 = enc.base64_encode("hello")
    assert enc.base64_decode(b64) == "hello"
"""
from __future__ import annotations

import base64
import urllib.parse
from typing import Any, Dict


class EncodingHelper:
    """
    Encoding/decoding helper for common formats.
    """

    # ------------------------------------------------------------------
    # Base64
    # ------------------------------------------------------------------

    def base64_encode(self, data: str) -> str:
        """Encode string to base64."""
        return base64.b64encode(data.encode("utf-8")).decode("ascii")

    def base64_decode(self, data: str) -> str:
        """Decode base64 to string."""
        return base64.b64decode(data.encode("ascii")).decode("utf-8")

    def base64_url_encode(self, data: str) -> str:
        """URL-safe base64 encode."""
        return base64.urlsafe_b64encode(data.encode("utf-8")).decode("ascii")

    def base64_url_decode(self, data: str) -> str:
        """URL-safe base64 decode."""
        return base64.urlsafe_b64decode(data.encode("ascii")).decode("utf-8")

    # ------------------------------------------------------------------
    # Hex
    # ------------------------------------------------------------------

    def hex_encode(self, data: str) -> str:
        """Encode string to hex."""
        return data.encode("utf-8").hex()

    def hex_decode(self, data: str) -> str:
        """Decode hex to string."""
        return bytes.fromhex(data).decode("utf-8")

    # ------------------------------------------------------------------
    # URL encoding
    # ------------------------------------------------------------------

    def url_encode(self, data: str) -> str:
        """URL-encode a string."""
        return urllib.parse.quote(data)

    def url_decode(self, data: str) -> str:
        """URL-decode a string."""
        return urllib.parse.unquote(data)

    def url_encode_params(self, params: Dict[str, Any]) -> str:
        """URL-encode query parameters."""
        return urllib.parse.urlencode(params)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, int]:
        return {"formats": 3}

    def __repr__(self) -> str:
        return "<EncodingHelper formats=3>"
