"""Tests for UnifiedMemoryPool — allocate, migrate, free, capacity, Jetson path."""

import numpy as np
import pytest

from sunset.unified_memory import MemoryHandle, UnifiedMemoryPool


# ── Helpers ─────────────────────────────────────────────────


def _is_valid_handle(handle):
    """Quick sanity check on a freshly allocated handle."""
    return (
        isinstance(handle, MemoryHandle)
        and handle._valid
        and handle.size_bytes > 0
        and handle.ptr != 0
        and handle.handle_id
    )


# ── Allocation tests ────────────────────────────────────────


class TestAllocate:
    """allocate() must return a valid handle with a usable pointer."""

    def test_allocate_1mb_cpu(self):
        """Allocate 1 MB on CPU → returns valid handle with non-zero pointer."""
        pool = UnifiedMemoryPool(capacity_bytes=10 * 1024 * 1024)  # 10 MB
        handle = pool.allocate(1 * 1024 * 1024, device="cpu")

        assert _is_valid_handle(handle)
        assert handle.size_bytes == 1 * 1024 * 1024
        assert (
            handle.device == "cpu" or handle.unified
        )  # jetson/managed report differently
        assert handle.ptr > 0
        assert pool.used_bytes >= handle.size_bytes

    def test_allocate_zero_raises(self):
        """Zero-byte allocation should be rejected."""
        pool = UnifiedMemoryPool()
        with pytest.raises(ValueError, match="positive"):
            pool.allocate(0)

    def test_allocate_negative_raises(self):
        pool = UnifiedMemoryPool()
        with pytest.raises(ValueError, match="positive"):
            pool.allocate(-1)

    def test_allocate_invalid_device_raises(self):
        pool = UnifiedMemoryPool()
        with pytest.raises(ValueError, match="Unsupported"):
            pool.allocate(1024, device="quantum")


# ── Migrate tests ───────────────────────────────────────────


class TestMigrate:
    """migrate() moves data between devices (or no-ops for unified memory)."""

    def test_migrate_cpu_to_gpu_pointer_valid(self):
        """Migrate CPU→GPU: pointer is still valid (unified) or changes (copy)."""
        pool = UnifiedMemoryPool(capacity_bytes=10 * 1024 * 1024)
        handle = pool.allocate(1 * 1024 * 1024, device="cpu")
        old_ptr = handle.ptr

        migrated = pool.migrate(handle, target_device="gpu")
        assert migrated is handle  # same object mutated in place
        assert migrated._valid

        if pool.mode in ("managed", "mapped", "jetson"):
            # Unified memory: same pointer everywhere
            assert migrated.ptr == old_ptr
            assert migrated.unified is True
        else:
            # Fallback: either pointer stays same (cpu→cpu) or we accept whatever
            assert migrated.ptr > 0

    def test_migrate_freed_handle_raises(self):
        """Migrating a freed handle should raise ValueError."""
        pool = UnifiedMemoryPool(capacity_bytes=10 * 1024 * 1024)
        handle = pool.allocate(1024, device="cpu")
        pool.free(handle)

        with pytest.raises(ValueError, match="freed"):
            pool.migrate(handle, target_device="gpu")

    def test_migrate_unsupported_device_raises(self):
        pool = UnifiedMemoryPool(capacity_bytes=10 * 1024 * 1024)
        handle = pool.allocate(1024, device="cpu")

        with pytest.raises(ValueError, match="Unsupported"):
            pool.migrate(handle, target_device="quantum")


# ── Free tests ──────────────────────────────────────────────


class TestFree:
    """free() releases memory and invalidates the handle."""

    def test_free_releases_memory(self):
        """After free, used_bytes decreases and handle is invalid."""
        pool = UnifiedMemoryPool(capacity_bytes=10 * 1024 * 1024)
        handle = pool.allocate(2 * 1024 * 1024, device="cpu")
        used_before = pool.used_bytes

        pool.free(handle)

        assert not handle._valid
        assert handle.ptr == 0
        assert pool.used_bytes == used_before - 2 * 1024 * 1024

    def test_free_invalidates_handle(self):
        """Handle must not be usable after free."""
        pool = UnifiedMemoryPool(capacity_bytes=10 * 1024 * 1024)
        handle = pool.allocate(1024, device="cpu")
        pool.free(handle)

        with pytest.raises(ValueError, match="freed"):
            pool.get_pointer(handle)

    def test_double_free_ignored(self):
        """Double free should be harmless (no crash, no exception)."""
        pool = UnifiedMemoryPool(capacity_bytes=10 * 1024 * 1024)
        handle = pool.allocate(1024, device="cpu")
        pool.free(handle)
        pool.free(handle)  # should not raise


# ── Capacity tests ──────────────────────────────────────────


class TestCapacity:
    """Pool must enforce capacity limits."""

    def test_over_allocation_raises(self):
        """Allocating more than capacity should raise MemoryError."""
        pool = UnifiedMemoryPool(capacity_bytes=2 * 1024 * 1024)  # 2 MB
        pool.allocate(1 * 1024 * 1024, device="cpu")  # 1 MB OK

        with pytest.raises(MemoryError, match="capacity"):
            pool.allocate(2 * 1024 * 1024, device="cpu")  # 2 MB > remaining 1 MB

    def test_exact_capacity_ok(self):
        """Edge case: allocation exactly matching remaining capacity."""
        pool = UnifiedMemoryPool(capacity_bytes=1 * 1024 * 1024)
        handle = pool.allocate(1 * 1024 * 1024, device="cpu")
        assert handle._valid
        assert pool.used_bytes == pool.capacity

    def test_free_makes_room(self):
        """Freeing memory should allow new allocations."""
        pool = UnifiedMemoryPool(capacity_bytes=1 * 1024 * 1024)
        handle = pool.allocate(1 * 1024 * 1024, device="cpu")
        pool.free(handle)
        handle2 = pool.allocate(1 * 1024 * 1024, device="cpu")
        assert handle2._valid


# ── Jetson / unified path tests ─────────────────────────────


class TestJetsonPath:
    """On Jetson (or any unified memory) migrate is a no-op and pointer is stable."""

    def test_jetson_same_pointer_after_migrate(self):
        """If pool is in unified mode, pointer must stay identical across migrate."""
        pool = UnifiedMemoryPool(capacity_bytes=10 * 1024 * 1024)
        handle = pool.allocate(1 * 1024 * 1024, device="cpu")
        ptr_before = handle.ptr

        pool.migrate(handle, target_device="gpu")
        ptr_after = handle.ptr

        if pool.mode in ("managed", "mapped", "jetson"):
            assert ptr_before == ptr_after, (
                f"Unified memory pointer changed: 0x{ptr_before:x} -> 0x{ptr_after:x} "
                f"(mode={pool.mode})"
            )
        else:
            # Fallback mode: pointer may change, but must remain valid
            assert ptr_after > 0

    def test_jetson_pointer_accessible_on_both_devices(self):
        """For unified memory, get_pointer should return same value for cpu and gpu."""
        pool = UnifiedMemoryPool(capacity_bytes=10 * 1024 * 1024)
        handle = pool.allocate(1 * 1024 * 1024, device="cpu")

        ptr_cpu = pool.get_pointer(handle, device="cpu")
        ptr_gpu = pool.get_pointer(handle, device="gpu")

        if pool.mode in ("managed", "mapped", "jetson"):
            assert ptr_cpu == ptr_gpu, (
                f"get_pointer(cpu)=0x{ptr_cpu:x} != get_pointer(gpu)=0x{ptr_gpu:x} "
                f"(mode={pool.mode})"
            )
        else:
            assert ptr_cpu > 0
            assert ptr_gpu > 0


# ── Pointer retrieval tests ─────────────────────────────────


class TestGetPointer:
    """get_pointer() returns usable addresses and triggers migrate if needed."""

    def test_get_pointer_returns_int(self):
        pool = UnifiedMemoryPool(capacity_bytes=10 * 1024 * 1024)
        handle = pool.allocate(1024, device="cpu")
        ptr = pool.get_pointer(handle, device="cpu")
        assert isinstance(ptr, int)
        assert ptr > 0

    def test_get_pointer_freed_raises(self):
        pool = UnifiedMemoryPool(capacity_bytes=10 * 1024 * 1024)
        handle = pool.allocate(1024, device="cpu")
        pool.free(handle)
        with pytest.raises(ValueError, match="freed"):
            pool.get_pointer(handle)


# ── Pool repr ───────────────────────────────────────────────


class TestPoolRepr:
    def test_repr_contains_mode(self):
        pool = UnifiedMemoryPool(capacity_bytes=10 * 1024 * 1024)
        r = repr(pool)
        assert "UnifiedMemoryPool" in r
        assert pool.mode in r

    def test_handle_repr(self):
        pool = UnifiedMemoryPool(capacity_bytes=10 * 1024 * 1024)
        handle = pool.allocate(1024, device="cpu")
        r = repr(handle)
        assert "MemoryHandle" in r
        assert "valid" in r
        pool.free(handle)
        r2 = repr(handle)
        assert "freed" in r2
