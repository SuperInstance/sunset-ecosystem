#!/usr/bin/env python3
"""Tests for swarm/eisenstein_integration.py."""
import math

import pytest

from swarm.eisenstein_integration import (
    E12,
    HexDisk,
    eisenstein_norm,
    snap_from_angle,
    hex_to_cartesian,
    cartesian_to_hex,
    HEX_DIRECTIONS,
)


class TestE12:
    def test_norm(self):
        assert E12(2, 1).norm() == 3  # 4 - 2 + 1 = 3
        assert E12(3, 0).norm() == 9
        assert E12(0, 0).norm() == 0

    def test_to_cartesian(self):
        c = E12(1, 0).to_cartesian()
        assert c == (1.0, 0.0)

        c = E12(0, 1).to_cartesian()
        assert abs(c[0] - (-0.5)) < 1e-9
        assert abs(c[1] - (math.sqrt(3) / 2)) < 1e-9

    def test_add(self):
        assert E12(1, 2) + E12(3, 4) == E12(4, 6)

    def test_sub(self):
        assert E12(5, 3) - E12(2, 1) == E12(3, 2)

    def test_mul(self):
        # (1 + 0ω) * (0 + 1ω) = 0 + 1ω
        assert E12(1, 0) * E12(0, 1) == E12(0, 1)

    def test_frozen(self):
        e = E12(1, 2)
        with pytest.raises(AttributeError):
            e.a = 3

    def test_type_validation(self):
        with pytest.raises(TypeError):
            E12(1.5, 2)

    def test_repr(self):
        assert repr(E12(1, 2)) == "E12(1, 2)"


class TestHexDisk:
    def test_init(self):
        disk = HexDisk(radius=1)
        assert disk.radius == 1
        assert len(disk) == 7  # center + 6 neighbors

    def test_snap_direction_center(self):
        disk = HexDisk()
        assert disk.snap_direction(0, 0) == E12(0, 0)

    def test_snap_direction_unit_x(self):
        disk = HexDisk()
        s = disk.snap_direction(1.0, 0.0)
        assert s == E12(1, 0)

    def test_snap_direction_60_deg(self):
        disk = HexDisk()
        x = math.cos(math.pi / 3)
        y = math.sin(math.pi / 3)
        s = disk.snap_direction(x, y)
        assert s == E12(1, 1)

    def test_neighbors(self):
        disk = HexDisk()
        n = disk.neighbors(E12(0, 0))
        assert len(n) == 6
        assert E12(1, 0) in n
        assert E12(-1, 0) in n

    def test_ring(self):
        disk = HexDisk()
        r = disk.ring(E12(0, 0), distance=1)
        assert len(r) == 6

    def test_distance(self):
        disk = HexDisk()
        assert disk.distance(E12(0, 0), E12(1, 0)) == 1
        assert disk.distance(E12(0, 0), E12(2, 0)) == 2
        assert disk.distance(E12(1, 0), E12(1, 1)) == 1  # adjacent
        assert disk.distance(E12(1, 0), E12(0, 1)) == 2  # not adjacent

    def test_repr(self):
        assert "HexDisk" in repr(HexDisk(radius=2))


class TestModuleFunctions:
    def test_eisenstein_norm(self):
        assert eisenstein_norm(2, 1) == 3
        assert eisenstein_norm(0, 0) == 0

    def test_snap_from_angle_0(self):
        assert snap_from_angle(0.0) == E12(1, 0)

    def test_snap_from_angle_60(self):
        assert snap_from_angle(math.pi / 3) == E12(1, 1)

    def test_snap_from_angle_180(self):
        assert snap_from_angle(math.pi) == E12(-1, 0)

    def test_hex_to_cartesian(self):
        c = hex_to_cartesian(1, 0)
        assert c == (1.0, 0.0)

    def test_cartesian_to_hex(self):
        h = cartesian_to_hex(1.0, 0.0)
        assert h == E12(1, 0)

    def test_hex_roundtrip(self):
        original = E12(2, -1)
        c = hex_to_cartesian(original.a, original.b)
        back = cartesian_to_hex(c[0], c[1])
        assert back == original


class TestHexDirections:
    def test_six_directions(self):
        assert len(HEX_DIRECTIONS) == 6

    def test_directions_distinct(self):
        assert len(set(HEX_DIRECTIONS)) == 6

    def test_directions_euclidean_distance(self):
        for d in HEX_DIRECTIONS:
            cx, cy = d.to_cartesian()
            dist = math.hypot(cx, cy)
            assert abs(dist - 1.0) < 1e-9
