"""Tests for request_signer.py — Request signing with HMAC and signature verification.

Run: python3 -m pytest tests/test_request_signer.py -v --tb=short
"""

from __future__ import annotations

import pytest

from fleet.request_signer import RequestSigner


class TestRequestSigner:
    def test_create(self):
        signer = RequestSigner(secret="my-secret", ttl_sec=60, clock=lambda: 0)
        assert signer.stats()["ttl"] == 60

    def test_sign(self):
        signer = RequestSigner(secret="my-secret", clock=lambda: 0)
        signature = signer.sign("/api/users", method="POST", body=b"data")
        assert signature.startswith("0:")
        assert len(signature) > 2

    def test_verify(self):
        signer = RequestSigner(secret="my-secret", clock=lambda: 0)
        signature = signer.sign("/api/users", method="POST", body=b"data")
        assert (
            signer.verify(
                "/api/users", method="POST", body=b"data", signature=signature
            )
            is True
        )

    def test_verify_invalid(self):
        signer = RequestSigner(secret="my-secret", clock=lambda: 0)
        assert signer.verify("/api/users", signature="invalid") is False

    def test_verify_wrong_path(self):
        signer = RequestSigner(secret="my-secret", clock=lambda: 0)
        signature = signer.sign("/api/users", method="POST", body=b"data")
        assert (
            signer.verify(
                "/api/other", method="POST", body=b"data", signature=signature
            )
            is False
        )

    def test_verify_wrong_method(self):
        signer = RequestSigner(secret="my-secret", clock=lambda: 0)
        signature = signer.sign("/api/users", method="POST", body=b"data")
        assert (
            signer.verify("/api/users", method="GET", body=b"data", signature=signature)
            is False
        )

    def test_verify_wrong_body(self):
        signer = RequestSigner(secret="my-secret", clock=lambda: 0)
        signature = signer.sign("/api/users", method="POST", body=b"data")
        assert (
            signer.verify(
                "/api/users", method="POST", body=b"tampered", signature=signature
            )
            is False
        )

    def test_verify_with_headers(self):
        signer = RequestSigner(secret="my-secret", clock=lambda: 0)
        headers = {"content-type": "application/json"}
        signature = signer.sign(
            "/api/users", method="POST", body=b"data", headers=headers
        )
        assert (
            signer.verify(
                "/api/users",
                method="POST",
                body=b"data",
                headers=headers,
                signature=signature,
            )
            is True
        )

    def test_ttl_expiration(self):
        signer = RequestSigner(secret="my-secret", ttl_sec=60, clock=lambda: 0)
        signature = signer.sign("/api/users", method="POST", body=b"data")
        assert (
            signer.verify(
                "/api/users", method="POST", body=b"data", signature=signature
            )
            is True
        )
        signer._clock = lambda: 70
        assert (
            signer.verify(
                "/api/users", method="POST", body=b"data", signature=signature
            )
            is False
        )

    def test_verify_no_ttl(self):
        signer = RequestSigner(secret="my-secret", clock=lambda: 0)
        signature = signer.sign("/api/users", method="POST", body=b"data")
        signer._clock = lambda: 1000000
        assert (
            signer.verify(
                "/api/users", method="POST", body=b"data", signature=signature
            )
            is True
        )

    def test_repr(self):
        signer = RequestSigner(secret="secret")
        assert "RequestSigner" in repr(signer)
