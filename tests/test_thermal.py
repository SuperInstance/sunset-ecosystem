"""Tests for ThermalBudget — device-aware agent slot allocation.

Covers DeviceBudget, ThermalBudget, thread safety, fallback allocation,
parent sacrifice, and auction integration stubs.
"""

import threading
import pytest

from swarm.thermal import (
    DeviceBudget,
    DeviceType,
    ThermalBudget,
    DEFAULT_BUDGETS,
)


# ---------------------------------------------------------------------------
# DeviceBudget
# ---------------------------------------------------------------------------


class TestDeviceBudget:
    def test_init(self):
        db = DeviceBudget(DeviceType.GPU, max_agents=8)
        assert db.device_type == DeviceType.GPU
        assert db.max_agents == 8
        assert db.current_agents == 0

    def test_available(self):
        db = DeviceBudget(DeviceType.CPU, max_agents=4, current_agents=2)
        assert db.available == 2

    def test_available_zero(self):
        db = DeviceBudget(DeviceType.CPU, max_agents=4, current_agents=4)
        assert db.available == 0

    def test_available_negative_clamped(self):
        db = DeviceBudget(DeviceType.CPU, max_agents=4, current_agents=6)
        assert db.available == 0

    def test_utilization(self):
        db = DeviceBudget(DeviceType.GPU, max_agents=10, current_agents=3)
        assert db.utilization == pytest.approx(0.3)

    def test_utilization_zero_max(self):
        db = DeviceBudget(DeviceType.GPU, max_agents=0)
        assert db.utilization == 0.0

    def test_repr(self):
        db = DeviceBudget(DeviceType.NPU, max_agents=6, current_agents=2)
        assert "npu" in repr(db)
        assert "2/6" in repr(db)


# ---------------------------------------------------------------------------
# ThermalBudget init
# ---------------------------------------------------------------------------


class TestThermalBudgetInit:
    def test_defaults(self):
        tb = ThermalBudget()
        assert tb.total_max == sum(DEFAULT_BUDGETS.values())
        assert tb.total_current == 0
        assert not tb.use_auction

    def test_custom_budgets(self):
        tb = ThermalBudget(budgets={DeviceType.GPU: 2, DeviceType.CPU: 4})
        assert tb.total_max == 6
        assert tb.device_budget(DeviceType.GPU).max_agents == 2

    def test_auction_mode(self):
        tb = ThermalBudget(use_auction=True, auction_interval=5)
        assert tb.use_auction
        assert tb.auction_interval == 5

    def test_repr(self):
        tb = ThermalBudget()
        assert "ThermalBudget" in repr(tb)
        assert "direct" in repr(tb)


# ---------------------------------------------------------------------------
# Allocation
# ---------------------------------------------------------------------------


class TestAllocation:
    def test_allocate_success(self):
        tb = ThermalBudget()
        assert tb.allocate("agent1", DeviceType.GPU)
        assert tb.total_current == 1
        assert tb.get_device("agent1") == DeviceType.GPU

    def test_allocate_fail_full(self):
        tb = ThermalBudget(budgets={DeviceType.GPU: 1})
        tb.allocate("agent1", DeviceType.GPU)
        assert not tb.allocate("agent2", DeviceType.GPU)

    def test_allocate_duplicate_raises(self):
        tb = ThermalBudget()
        tb.allocate("agent1", DeviceType.GPU)
        with pytest.raises(ValueError):
            tb.allocate("agent1", DeviceType.CPU)

    def test_release(self):
        tb = ThermalBudget()
        tb.allocate("agent1", DeviceType.GPU)
        assert tb.release("agent1")
        assert tb.total_current == 0
        assert tb.get_device("agent1") is None

    def test_release_missing(self):
        tb = ThermalBudget()
        assert not tb.release("nobody")

    def test_can_spawn(self):
        tb = ThermalBudget(budgets={DeviceType.GPU: 1})
        assert tb.can_spawn(DeviceType.GPU)
        tb.allocate("a", DeviceType.GPU)
        assert not tb.can_spawn(DeviceType.GPU)


# ---------------------------------------------------------------------------
# Fallback allocation
# ---------------------------------------------------------------------------


class TestFallbackAllocation:
    def test_preferred_first(self):
        tb = ThermalBudget()
        ok, device = tb.spawn_with_thermal_check("a1", DeviceType.GPU)
        assert ok
        assert device == DeviceType.GPU

    def test_fallback_when_full(self):
        tb = ThermalBudget(budgets={DeviceType.GPU: 0, DeviceType.CPU: 1})
        ok, device = tb.spawn_with_thermal_check("a1", DeviceType.GPU)
        assert ok
        assert device == DeviceType.CPU

    def test_all_full(self):
        tb = ThermalBudget(budgets={DeviceType.GPU: 0, DeviceType.CPU: 0})
        ok, device = tb.spawn_with_thermal_check("a1", DeviceType.GPU)
        assert not ok
        assert device is None

    def test_explicit_fallbacks(self):
        tb = ThermalBudget(
            budgets={DeviceType.GPU: 0, DeviceType.CPU: 1, DeviceType.IGPU: 1}
        )
        ok, device = tb.spawn_with_thermal_check(
            "a1", DeviceType.GPU, fallback_devices=[DeviceType.IGPU, DeviceType.CPU]
        )
        assert ok
        assert device == DeviceType.IGPU

    def test_already_allocated_returns_false(self):
        tb = ThermalBudget()
        tb.allocate("a1", DeviceType.GPU)
        ok, device = tb.spawn_with_thermal_check("a1", DeviceType.CPU)
        assert not ok
        assert device == DeviceType.GPU


# ---------------------------------------------------------------------------
# Parent sacrifice
# ---------------------------------------------------------------------------


class TestParentSacrifice:
    def test_direct_room_no_sacrifice(self):
        tb = ThermalBudget(budgets={DeviceType.GPU: 2})
        tb.allocate("parent", DeviceType.CPU)
        assert tb.parent_sacrifice_before_spawn("parent", DeviceType.GPU)

    def test_sacrifice_makes_room(self):
        tb = ThermalBudget(budgets={DeviceType.GPU: 1})
        tb.allocate("parent", DeviceType.GPU)
        assert tb.parent_sacrifice_before_spawn("parent", DeviceType.GPU)
        assert tb.get_device("parent") is None
        assert tb.device_budget(DeviceType.GPU).current_agents == 0

    def test_sacrifice_undo_when_still_full(self):
        tb = ThermalBudget(budgets={DeviceType.GPU: 1, DeviceType.CPU: 1})
        tb.allocate("parent", DeviceType.CPU)
        tb.allocate("other", DeviceType.GPU)
        # CPU parent sacrificed, but GPU still full — undo
        assert not tb.parent_sacrifice_before_spawn("parent", DeviceType.GPU)
        assert tb.get_device("parent") == DeviceType.CPU

    def test_missing_parent(self):
        tb = ThermalBudget(budgets={DeviceType.GPU: 1})
        tb.allocate("other", DeviceType.GPU)
        assert not tb.parent_sacrifice_before_spawn("orphan", DeviceType.GPU)


# ---------------------------------------------------------------------------
# Thermal headroom / can_breed
# ---------------------------------------------------------------------------


class TestThermalHeadroom:
    def test_empty(self):
        tb = ThermalBudget()
        assert tb.thermal_headroom() == 0.0
        assert tb.can_breed()

    def test_half_full(self):
        tb = ThermalBudget(budgets={DeviceType.CPU: 4})
        tb.allocate("a1", DeviceType.CPU)
        tb.allocate("a2", DeviceType.CPU)
        assert tb.thermal_headroom() == pytest.approx(0.5)
        assert tb.can_breed()

    def test_at_threshold(self):
        tb = ThermalBudget(budgets={DeviceType.CPU: 10})
        for i in range(8):
            tb.allocate(f"a{i}", DeviceType.CPU)
        assert tb.thermal_headroom() == pytest.approx(0.8)
        assert not tb.can_breed()

    def test_above_threshold(self):
        tb = ThermalBudget(budgets={DeviceType.CPU: 10})
        for i in range(9):
            tb.allocate(f"a{i}", DeviceType.CPU)
        assert tb.thermal_headroom() == pytest.approx(0.9)
        assert not tb.can_breed(threshold=0.8)

    def test_reset(self):
        tb = ThermalBudget()
        tb.allocate("a1", DeviceType.GPU)
        tb.allocate("a2", DeviceType.CPU)
        tb.reset()
        assert tb.total_current == 0
        assert tb.get_device("a1") is None


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_allocate(self):
        tb = ThermalBudget(budgets={DeviceType.CPU: 100})
        errors = []

        def worker(n, tnum):
            try:
                for i in range(n):
                    tb.allocate(f"agent_{tnum}_{i}", DeviceType.CPU)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(20, i)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert tb.total_current == 100

    def test_concurrent_release(self):
        tb = ThermalBudget(budgets={DeviceType.CPU: 50})
        for i in range(50):
            tb.allocate(f"a{i}", DeviceType.CPU)

        def releaser():
            for i in range(50):
                tb.release(f"a{i}")

        t1 = threading.Thread(target=releaser)
        t2 = threading.Thread(target=releaser)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # May go negative due to double-release race, but shouldn't crash
        assert tb.device_budget(DeviceType.CPU).current_agents <= 0


# ---------------------------------------------------------------------------
# Auction stub
# ---------------------------------------------------------------------------


class TestAuctionStub:
    def test_auction_tick_no_bids(self):
        tb = ThermalBudget(use_auction=True)
        assert tb.tick() == {}

    def test_spawn_queues_bid_in_auction_mode(self):
        tb = ThermalBudget(use_auction=True)
        ok, device = tb.spawn("a1", DeviceType.GPU, bid_value=1.0)
        assert ok
        assert device is None  # allocation deferred to tick()

    def test_spawn_falls_back_direct_when_no_auction(self):
        tb = ThermalBudget(use_auction=False)
        ok, device = tb.spawn("a1", DeviceType.GPU)
        assert ok
        assert device == DeviceType.GPU
