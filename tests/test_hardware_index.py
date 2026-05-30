"""Tests for swarm.hardware_index — workload-aware hardware placement."""

import pytest

from swarm.hardware_index import (
    DeviceProfile,
    DeviceType,
    HardwareProfileIndex,
    WorkloadQuery,
)


class TestDeviceProfile:
    def test_create(self):
        p = DeviceProfile(
            device_id="gpu_0",
            device_type=DeviceType.RTX_4050,
            capabilities=[1.0] * 8,
            current_load=[0.0] * 8,
            metadata={"node": "test"},
        )
        assert p.device_id == "gpu_0"
        assert p.device_type == DeviceType.RTX_4050

    def test_free_capacity(self):
        p = DeviceProfile(
            device_id="gpu_0",
            device_type=DeviceType.RTX_4050,
            capabilities=[1.0] * 8,
            current_load=[0.5] * 8,
            metadata={},
        )
        assert p.free_capacity == pytest.approx(0.5)

    def test_free_capacity_mismatched(self):
        p = DeviceProfile(
            device_id="gpu_0",
            device_type=DeviceType.RTX_4050,
            capabilities=[1.0] * 8,
            current_load=[0.5] * 4,
            metadata={},
        )
        assert p.free_capacity == 0.0


class TestWorkloadQuery:
    def test_create(self):
        q = WorkloadQuery(
            min_capabilities=[0.5] * 8,
            max_load=[0.8] * 8,
            preferred_type=DeviceType.RTX_4050,
            weights=[0.125] * 8,
        )
        assert q.preferred_type == DeviceType.RTX_4050


class TestHardwareProfileIndex:
    def test_create(self):
        idx = HardwareProfileIndex()
        assert idx.device_count() == 0

    def test_register(self):
        idx = HardwareProfileIndex()
        p = DeviceProfile(
            device_id="gpu_0",
            device_type=DeviceType.RTX_4050,
            capabilities=[1.0] * 8,
            current_load=[0.0] * 8,
            metadata={"node": "test"},
        )
        idx.register(p)
        assert idx.device_count() == 1
        assert idx.get_device("gpu_0") == p

    def test_register_update(self):
        idx = HardwareProfileIndex()
        p1 = DeviceProfile(
            device_id="gpu_0",
            device_type=DeviceType.RTX_4050,
            capabilities=[1.0] * 8,
            current_load=[0.0] * 8,
            metadata={},
        )
        idx.register(p1)
        p2 = DeviceProfile(
            device_id="gpu_0",
            device_type=DeviceType.RTX_4050,
            capabilities=[1.0] * 8,
            current_load=[0.5] * 8,
            metadata={},
        )
        idx.register(p2)
        assert idx.device_count() == 1
        assert idx.get_device("gpu_0").current_load[0] == 0.5

    def test_update_load(self):
        idx = HardwareProfileIndex()
        p = DeviceProfile(
            device_id="gpu_0",
            device_type=DeviceType.RTX_4050,
            capabilities=[1.0] * 8,
            current_load=[0.0] * 8,
            metadata={},
        )
        idx.register(p)
        idx.update_load("gpu_0", [0.3] * 8)
        assert idx.get_device("gpu_0").current_load[0] == pytest.approx(0.3)

    def test_update_load_unknown(self):
        idx = HardwareProfileIndex()
        idx.update_load("unknown", [0.3] * 8)  # should not raise

    def test_get_device_none(self):
        idx = HardwareProfileIndex()
        assert idx.get_device("nonexistent") is None

    def test_repr(self):
        idx = HardwareProfileIndex()
        assert "HardwareProfileIndex" in repr(idx)

    def test_hash_device_id(self):
        h1 = HardwareProfileIndex._hash_device_id("gpu_0")
        h2 = HardwareProfileIndex._hash_device_id("gpu_0")
        assert h1 == h2
        assert isinstance(h1, int)

    def test_score_device(self):
        p = DeviceProfile(
            device_id="gpu_0",
            device_type=DeviceType.RTX_4050,
            capabilities=[1.0] * 8,
            current_load=[0.0] * 8,
            metadata={},
        )
        q = WorkloadQuery(
            min_capabilities=[0.5] * 8,
            max_load=[0.8] * 8,
            preferred_type=DeviceType.RTX_4050,
        )
        score = HardwareProfileIndex._score_device(p, q)
        assert score > 0.0

    def test_score_device_type_mismatch(self):
        p = DeviceProfile(
            device_id="gpu_0",
            device_type=DeviceType.RADEON_890M,
            capabilities=[1.0] * 8,
            current_load=[0.0] * 8,
            metadata={},
        )
        q = WorkloadQuery(
            min_capabilities=[0.5] * 8,
            max_load=[0.8] * 8,
            preferred_type=DeviceType.RTX_4050,
        )
        score = HardwareProfileIndex._score_device(p, q)
        assert score > 0.0  # still positive, just lower type bonus

    def test_score_device_with_weights(self):
        p = DeviceProfile(
            device_id="gpu_0",
            device_type=DeviceType.RTX_4050,
            capabilities=[1.0] * 8,
            current_load=[0.0] * 8,
            metadata={},
        )
        q = WorkloadQuery(
            min_capabilities=[0.5] * 8,
            max_load=[0.8] * 8,
            weights=[0.125] * 8,
        )
        score = HardwareProfileIndex._score_device(p, q)
        assert score > 0.0
