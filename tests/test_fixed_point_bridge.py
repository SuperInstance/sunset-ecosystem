"""Tests for FixedPointBridge (flux_compat/fixed_point_bridge.py).

Covers:
    - Auto-scaling from pilot evaluations
    - Encode/decode round-trip accuracy
    - Saturation at extremes
    - Wrap behavior
    - Raise behavior
    - Different fractional bit widths (8, 16, 32)
    - Batch operations
    - FLUX constant generation
"""

from __future__ import annotations

import math

import pytest

from flux_compat.fixed_point_bridge import FixedPointBridge, OverflowMode


# ── 1. Auto-scaling ───────────────────────────────────────

class TestAutoScale:
    def test_max_abs_strategy(self):
        bridge = FixedPointBridge.auto_scale(
            [1.0, -2.5, 0.1, 100.0],
            frac_bits=16,
            total_bits=32,
            sample_strategy="max_abs",
        )
        # anchor = 100.0, safety_margin = 2.0
        # max_fp = 2^31 - 1 ≈ 2.147e9
        # scale = 2.147e9 / 200 ≈ 1.07e7
        assert bridge.scale_factor > 0
        # 100.0 should fit comfortably
        assert bridge.encode(100.0) < bridge._max_raw
        assert bridge.encode(-100.0) > bridge._min_raw

    def test_p99_strategy(self):
        # 99th percentile should ignore the single 1e6 outlier
        pilot = [1.0] * 99 + [1_000_000.0]
        bridge = FixedPointBridge.auto_scale(
            pilot,
            sample_strategy="p99",
            safety_margin=1.0,
        )
        # anchor = 1.0 (p99), so scale ≈ 2^31-1
        assert bridge.scale_factor > 1e9
        # 1.0 should encode exactly-ish
        assert bridge.encode(1.0) > 0

    def test_mean_abs_strategy(self):
        bridge = FixedPointBridge.auto_scale(
            [1.0, 2.0, 3.0],
            sample_strategy="mean_abs",
            safety_margin=1.0,
        )
        anchor = 2.0  # mean of [1,2,3]
        expected_scale = ((1 << 31) - 1) / 2.0
        assert abs(bridge.scale_factor - expected_scale) < 1.0

    def test_empty_pilot_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            FixedPointBridge.auto_scale([])

    def test_zero_pilot_all_zeros(self):
        bridge = FixedPointBridge.auto_scale([0.0, 0.0, 0.0])
        # Should get a large arbitrary scale
        assert bridge.scale_factor > 1e8
        assert bridge.encode(0.0) == 0

    def test_safety_margin_headroom(self):
        bridge_narrow = FixedPointBridge.auto_scale(
            [10.0], safety_margin=1.0
        )
        bridge_wide = FixedPointBridge.auto_scale(
            [10.0], safety_margin=4.0
        )
        # Wider margin = smaller scale factor
        assert bridge_wide.scale_factor < bridge_narrow.scale_factor

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="unknown sample_strategy"):
            FixedPointBridge.auto_scale([1.0], sample_strategy="nope")


# ── 2. Round-trip accuracy ────────────────────────────────

class TestRoundTrip:
    def test_identity_at_zero(self):
        bridge = FixedPointBridge(frac_bits=16, total_bits=32, scale_factor=1e6)
        assert bridge.encode(0.0) == 0
        assert bridge.decode(0) == 0.0

    def test_round_trip_16bit(self):
        bridge = FixedPointBridge.auto_scale(
            [1.0, 2.0, 5.0], frac_bits=16, total_bits=32
        )
        for val in [1.0, 2.0, 3.14159, -0.5, 5.0]:
            fp = bridge.encode(val)
            back = bridge.decode(fp)
            err = abs(back - val)
            assert err <= bridge.resolution

    def test_round_trip_8bit(self):
        bridge = FixedPointBridge.auto_scale(
            [10.0, 20.0], frac_bits=8, total_bits=16
        )
        for val in [10.0, 15.0, 20.0]:
            fp = bridge.encode(val)
            back = bridge.decode(fp)
            err = abs(back - val)
            assert err <= bridge.resolution

    def test_round_trip_32bit(self):
        bridge = FixedPointBridge.auto_scale(
            [1e-6, 1e-3, 1.0], frac_bits=32, total_bits=64
        )
        for val in [1e-6, 1e-3, 0.5, 1.0]:
            fp = bridge.encode(val)
            back = bridge.decode(fp)
            err = abs(back - val)
            assert err <= bridge.resolution


# ── 3. Overflow modes ────────────────────────────────────

class TestOverflowModes:
    def test_saturate_at_max(self):
        bridge = FixedPointBridge(
            frac_bits=16, total_bits=32, scale_factor=1e6,
            overflow=OverflowMode.SATURATE,
        )
        huge = bridge.max_representable * 2.0
        raw = bridge.encode(huge)
        assert raw == bridge._max_raw

    def test_saturate_at_min(self):
        bridge = FixedPointBridge(
            frac_bits=16, total_bits=32, scale_factor=1e6,
            overflow=OverflowMode.SATURATE,
        )
        tiny = bridge.min_representable * 2.0
        raw = bridge.encode(tiny)
        assert raw == bridge._min_raw

    def test_wrap_behavior(self):
        bridge = FixedPointBridge(
            frac_bits=8, total_bits=16, scale_factor=1.0,
            overflow=OverflowMode.WRAP,
        )
        max_val = bridge._max_raw  # 32767
        # 32768 should wrap to -32768
        raw = bridge.encode(max_val + 1)
        assert raw == bridge._min_raw  # -32768

    def test_raise_on_overflow(self):
        bridge = FixedPointBridge(
            frac_bits=16, total_bits=32, scale_factor=1.0,
            overflow=OverflowMode.RAISE,
        )
        with pytest.raises(OverflowError):
            bridge.encode(float(bridge._max_raw + 1))

    def test_raise_on_underflow(self):
        bridge = FixedPointBridge(
            frac_bits=16, total_bits=32, scale_factor=1.0,
            overflow=OverflowMode.RAISE,
        )
        with pytest.raises(OverflowError):
            bridge.encode(float(bridge._min_raw - 1))


# ── 4. Different bit widths ───────────────────────────────

class TestBitWidths:
    def test_8_frac_16_total(self):
        bridge = FixedPointBridge(frac_bits=8, total_bits=16, scale_factor=256.0)
        assert bridge.encode(1.0) == 256
        assert bridge.encode(1.5) == 384
        assert bridge.decode(256) == 1.0

    def test_16_frac_32_total(self):
        bridge = FixedPointBridge(frac_bits=16, total_bits=32, scale_factor=65536.0)
        assert bridge.encode(1.0) == 65536
        assert bridge.decode(65536) == 1.0

    def test_32_frac_64_total(self):
        bridge = FixedPointBridge(frac_bits=32, total_bits=64, scale_factor=2**32)
        assert bridge.encode(1.0) == 2**32
        assert bridge.decode(2**32) == 1.0


# ── 5. Batch operations ───────────────────────────────────

class TestBatch:
    def test_encode_batch(self):
        bridge = FixedPointBridge(scale_factor=100.0)
        vals = [1.0, 2.0, 3.0]
        raws = bridge.encode_batch(vals)
        assert raws == [100, 200, 300]

    def test_decode_batch(self):
        bridge = FixedPointBridge(scale_factor=100.0)
        raws = [100, 200, 300]
        vals = bridge.decode_batch(raws)
        assert vals == [1.0, 2.0, 3.0]

    def test_round_trip_batch(self):
        bridge = FixedPointBridge.auto_scale([1.0, 5.0, 10.0])
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        raws = bridge.encode_batch(vals)
        back = bridge.decode_batch(raws)
        for v, b in zip(vals, back):
            assert abs(v - b) <= bridge.resolution


# ── 6. FLUX constant generation ───────────────────────────

class TestFluxConstant:
    def test_flux_constant_structure(self):
        bridge = FixedPointBridge(scale_factor=1e6)
        const = bridge.flux_constant(3.14159)
        assert "raw" in const
        assert "scale" in const
        assert "frac_bits" in const
        assert "total_bits" in const
        assert const["scale"] == 1e6
        assert const["frac_bits"] == 16
        assert const["total_bits"] == 32

    def test_flux_constant_round_trip(self):
        bridge = FixedPointBridge.auto_scale([1.0, 10.0])
        const = bridge.flux_constant(2.71828)
        back = const["raw"] / const["scale"]
        assert abs(back - 2.71828) <= bridge.resolution


# ── 7. Properties ─────────────────────────────────────────

class TestProperties:
    def test_resolution(self):
        bridge = FixedPointBridge(scale_factor=1e6)
        assert bridge.resolution == 1e-6

    def test_max_min_representable(self):
        bridge = FixedPointBridge(total_bits=16, scale_factor=1.0)
        assert bridge.max_representable == 32767.0
        assert bridge.min_representable == -32768.0

    def test_repr(self):
        bridge = FixedPointBridge(scale_factor=1e6)
        r = repr(bridge)
        assert "FixedPointBridge" in r
        assert "scale=" in r
        assert "res=" in r
