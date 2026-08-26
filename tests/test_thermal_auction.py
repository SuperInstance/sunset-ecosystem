"""Tests for VCG thermal auction.

Validates:
1. Truthful bidding is a dominant strategy (VCG incentive compatibility).
2. Higher-fitness agent wins when slots are scarce.
3. VCG price never exceeds the winning bid.
4. ThermalBudget integration respects per-device limits.
"""

from __future__ import annotations

import pytest

from swarm.thermal import DeviceType, ThermalBudget, DEFAULT_BUDGETS
from swarm.thermal_auction import Bid, Allocation, VCGAuction


# ── fixtures ──────────────────────────────────────────────────


@pytest.fixture
def small_budget():
    """ThermalBudget with 1 slot per device for scarcity testing."""
    return ThermalBudget(
        budgets={DeviceType.GPU: 1, DeviceType.CPU: 1},
        use_auction=True,
        auction_interval=1,
    )


@pytest.fixture
def auction(small_budget):
    """VCGAuction backed by the small_budget fixture."""
    return VCGAuction(small_budget)


# ── unit tests for VCGAuction ─────────────────────────────────


class TestVCGAuctionLogic:
    """Pure auction logic without ThermalBudget integration."""

    def test_two_agents_one_slot_higher_value_wins(self, auction, small_budget):
        """With 1 GPU slot, the agent with higher bid value wins."""
        bids = [
            Bid("agent_low", DeviceType.GPU, value=0.5, fitness=0.5),
            Bid("agent_high", DeviceType.GPU, value=0.9, fitness=0.9),
        ]
        results = auction.run_auction(bids)

        assert "agent_high" in results
        assert "agent_low" not in results
        assert results["agent_high"].device_type == DeviceType.GPU
        # VCG price with 1 slot and 2 bids = value of losing bid
        assert results["agent_high"].price_paid == pytest.approx(0.5)

    def test_two_agents_one_slot_tie_break_by_fitness(self, auction):
        """When values are equal, higher fitness wins (determinism)."""
        bids = [
            Bid("agent_a", DeviceType.GPU, value=0.7, fitness=0.6),
            Bid("agent_b", DeviceType.GPU, value=0.7, fitness=0.9),
        ]
        results = auction.run_auction(bids)
        assert "agent_b" in results
        assert "agent_a" not in results

    def test_vcg_price_less_than_or_equal_to_bid(self, auction):
        """No winner pays more than their bid (individual rationality)."""
        bids = [
            Bid(
                f"agent_{i}",
                DeviceType.GPU,
                value=float(i + 1) / 10,
                fitness=float(i + 1) / 10,
            )
            for i in range(5)
        ]
        results = auction.run_auction(bids)

        for agent_id, alloc in results.items():
            bid_value = next(b.value for b in bids if b.agent_id == agent_id)
            assert alloc.price_paid <= bid_value + 1e-9, (
                f"Agent {agent_id} paid {alloc.price_paid:.4f} but bid {bid_value:.4f}"
            )

    def test_vcg_price_is_zero_when_no_losers(self, auction):
        """If there are fewer bids than slots, winners pay 0."""
        bids = [
            Bid("lonely", DeviceType.GPU, value=0.8, fitness=0.8),
        ]
        results = auction.run_auction(bids)
        assert results["lonely"].price_paid == pytest.approx(0.0)

    def test_truthful_bid_dominates(self, auction):
        """VCG incentive compatibility: truthful bid yields ≥ utility than any shade.

        Setup: 2 agents, 1 slot.
        Agent A true value = 0.8, Agent B true value = 0.5.
        If A bids truthfully (0.8): wins, pays 0.5, utility = 0.3.
        If A under-bids (0.4): loses, utility = 0.
        If A over-bids (1.0): wins, pays 0.5, utility = 0.3 (same, no benefit).
        """
        true_value_a = 0.8
        true_value_b = 0.5

        # Truthful scenario
        truthful_bids = [
            Bid("a", DeviceType.GPU, value=true_value_a, fitness=true_value_a),
            Bid("b", DeviceType.GPU, value=true_value_b, fitness=true_value_b),
        ]
        t_results = auction.run_auction(truthful_bids)
        t_util_a = true_value_a - t_results["a"].price_paid

        # A under-bids by 50%
        shade_bids = [
            Bid("a", DeviceType.GPU, value=0.3, fitness=0.3),
            Bid("b", DeviceType.GPU, value=true_value_b, fitness=true_value_b),
        ]
        s_results = auction.run_auction(shade_bids)
        if "a" in s_results:
            s_util_a = true_value_a - s_results["a"].price_paid
        else:
            s_util_a = 0.0  # lost the auction

        # Truthful utility should be ≥ shaded utility
        assert t_util_a >= s_util_a - 1e-9, (
            f"Truthful utility {t_util_a:.3f} < shaded utility {s_util_a:.3f} — "
            "VCG incentive compatibility violated"
        )

        # Over-bid scenario (A bids 1.0 instead of 0.8)
        over_bids = [
            Bid("a", DeviceType.GPU, value=1.0, fitness=1.0),
            Bid("b", DeviceType.GPU, value=true_value_b, fitness=true_value_b),
        ]
        o_results = auction.run_auction(over_bids)
        o_util_a = true_value_a - o_results["a"].price_paid

        # Over-bidding can't improve utility beyond truthful
        assert t_util_a >= o_util_a - 1e-9, (
            f"Truthful utility {t_util_a:.3f} < over-bid utility {o_util_a:.3f}"
        )

    def test_multi_slot_device_all_winners_pay_same_price(self, auction):
        """With 3 slots and 5 bidders, top 3 win and all pay the 4th highest bid."""
        small_budget = ThermalBudget(
            budgets={DeviceType.GPU: 3},
            use_auction=True,
        )
        auction3 = VCGAuction(small_budget)
        bids = [
            Bid(
                f"agent_{i}",
                DeviceType.GPU,
                value=float(i + 1) * 0.1,
                fitness=float(i + 1) * 0.1,
            )
            for i in range(5)
        ]
        # Values: 0.1, 0.2, 0.3, 0.4, 0.5 → top 3: 0.5, 0.4, 0.3 → pay 0.2
        results = auction3.run_auction(bids)
        assert len(results) == 3
        for alloc in results.values():
            assert alloc.price_paid == pytest.approx(0.2)

    def test_empty_bids_returns_empty(self, auction):
        """No bids → no allocations."""
        assert auction.run_auction([]) == {}


# ── integration tests with ThermalBudget ──────────────────────


class TestThermalBudgetAuctionIntegration:
    """Auction mode integrated into ThermalBudget.spawn() and tick()."""

    def test_spawn_queues_bid_in_auction_mode(self):
        """spawn() with use_auction=True queues a bid instead of allocating."""
        budget = ThermalBudget(
            budgets={DeviceType.GPU: 2},
            use_auction=True,
            auction_interval=1,
        )
        success, device = budget.spawn("a1", DeviceType.GPU, bid_value=0.8, fitness=0.8)
        # Bid queued successfully; no immediate allocation
        assert success is True
        assert device is None
        assert budget.total_current == 0

    def test_tick_runs_auction_and_allocates(self):
        """tick() processes queued bids and applies winning allocations."""
        budget = ThermalBudget(
            budgets={DeviceType.GPU: 1},
            use_auction=True,
            auction_interval=1,
        )
        budget.spawn("a1", DeviceType.GPU, bid_value=0.9, fitness=0.9)
        budget.spawn("a2", DeviceType.GPU, bid_value=0.5, fitness=0.5)

        results = budget.tick()

        # a1 should win the single GPU slot
        assert "a1" in results
        assert budget.get_device("a1") == DeviceType.GPU
        assert budget.total_current == 1

    def test_allocations_respect_device_limits(self):
        """Auction never allocates more agents than a device’s max_agents."""
        budget = ThermalBudget(
            budgets={DeviceType.GPU: 2, DeviceType.CPU: 3},
            use_auction=True,
            auction_interval=1,
        )
        # Queue 10 bids for GPU (max 2 slots)
        for i in range(10):
            budget.spawn(
                f"gpu_{i}",
                DeviceType.GPU,
                bid_value=float(i) / 10,
                fitness=float(i) / 10,
            )
        # Queue 10 bids for CPU (max 3 slots)
        for i in range(10):
            budget.spawn(
                f"cpu_{i}",
                DeviceType.CPU,
                bid_value=float(i) / 10,
                fitness=float(i) / 10,
            )

        budget.tick()

        gpu_count = sum(
            1 for aid, dev in budget._allocations.items() if dev == DeviceType.GPU
        )
        cpu_count = sum(
            1 for aid, dev in budget._allocations.items() if dev == DeviceType.CPU
        )

        assert gpu_count <= 2, f"GPU over-allocated: {gpu_count} > 2"
        assert cpu_count <= 3, f"CPU over-allocated: {cpu_count} > 3"
        assert budget.total_current <= 5

    def test_auction_interval_skips_ticks(self):
        """With interval=3, auction only runs every 3rd tick."""
        budget = ThermalBudget(
            budgets={DeviceType.GPU: 1},
            use_auction=True,
            auction_interval=3,
        )
        budget.spawn("a1", DeviceType.GPU, bid_value=0.8, fitness=0.8)

        # Ticks 1 and 2: no auction
        assert budget.tick() == {}
        assert budget.tick() == {}
        assert budget.total_current == 0

        # Tick 3: auction runs
        results = budget.tick()
        assert "a1" in results
        assert budget.total_current == 1

    def test_spawn_fallback_to_direct_when_auction_disabled(self):
        """spawn() with use_auction=False delegates to spawn_with_thermal_check."""
        budget = ThermalBudget(
            budgets={DeviceType.GPU: 1},
            use_auction=False,
        )
        success, device = budget.spawn("a1", DeviceType.GPU)
        assert success is True
        assert device == DeviceType.GPU
        assert budget.total_current == 1

    def test_auction_price_first_principles_matches_highest_loser(self):
        """Explicit VCG formula equals the simplified highest-loser price."""
        budget = ThermalBudget(
            budgets={DeviceType.GPU: 2},
            use_auction=True,
        )
        auction = VCGAuction(budget)
        bids = [
            Bid("a", DeviceType.GPU, value=0.9, fitness=0.9),
            Bid("b", DeviceType.GPU, value=0.7, fitness=0.7),
            Bid("c", DeviceType.GPU, value=0.5, fitness=0.5),
        ]
        results = auction.run_auction(bids)

        # With 2 slots and 3 bids, winners pay the 3rd highest bid = 0.5
        for agent_id, alloc in results.items():
            assert alloc.price_paid == pytest.approx(0.5)

        # Verify via explicit vcg_price() call
        winner_a = next(b for b in bids if b.agent_id == "a")
        price_a = auction.vcg_price(winner_a, bids, {})
        assert price_a == pytest.approx(0.5)

    def test_reset_clears_auction_state(self):
        """reset() drops queued bids and applied allocations."""
        budget = ThermalBudget(
            budgets={DeviceType.GPU: 1},
            use_auction=True,
            auction_interval=1,
        )
        budget.spawn("a1", DeviceType.GPU, bid_value=0.8, fitness=0.8)
        budget.tick()
        assert budget.total_current == 1

        budget.reset()
        assert budget.total_current == 0
        assert budget.last_auction_results == {}

    def test_multiple_device_types_independent_auctions(self):
        """Bids for GPU and CPU are auctioned independently."""
        budget = ThermalBudget(
            budgets={DeviceType.GPU: 1, DeviceType.CPU: 1},
            use_auction=True,
            auction_interval=1,
        )
        budget.spawn("g1", DeviceType.GPU, bid_value=0.9, fitness=0.9)
        budget.spawn("g2", DeviceType.GPU, bid_value=0.5, fitness=0.5)
        budget.spawn("c1", DeviceType.CPU, bid_value=0.8, fitness=0.8)
        budget.spawn("c2", DeviceType.CPU, bid_value=0.4, fitness=0.4)

        results = budget.tick()

        assert budget.get_device("g1") == DeviceType.GPU
        assert budget.get_device("c1") == DeviceType.CPU
        assert budget.get_device("g2") is None
        assert budget.get_device("c2") is None
        assert budget.total_current == 2

    def test_reused_agent_id_returns_false_on_direct_spawn(self):
        """Direct spawn returns False for duplicate agent_id (graceful, no crash)."""
        budget = ThermalBudget(
            budgets={DeviceType.GPU: 2},
            use_auction=False,
        )
        budget.spawn("dup", DeviceType.GPU)
        success, device = budget.spawn("dup", DeviceType.GPU)
        assert success is False
        assert device == DeviceType.GPU

    def test_last_auction_results_property(self):
        """last_auction_results exposes the most recent allocations."""
        budget = ThermalBudget(
            budgets={DeviceType.GPU: 1},
            use_auction=True,
            auction_interval=1,
        )
        budget.spawn("a1", DeviceType.GPU, bid_value=0.8, fitness=0.8)
        budget.tick()

        results = budget.last_auction_results
        assert "a1" in results
        assert isinstance(results["a1"], Allocation)
