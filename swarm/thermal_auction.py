"""VCG combinatorial auction for thermal slot allocation.

Agents submit bids (fitness × expected_value) for device slots.
VCG ensures truthful bidding is the dominant strategy.

For each device type with S identical slots:
- Top S bidders win
- Each winner pays the (S+1)th highest bid (or 0 if ≤ S bids)
- This is the externality each winner imposes on the losing bidders.
"""

from __future__ import annotations

__all__ = [
    "Bid",
    "Allocation",
    "VCGAuction",
]

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from swarm.thermal import DeviceType, ThermalBudget


@dataclass(frozen=True)
class Bid:
    """Agent bid for a device slot.

    Attributes:
        agent_id: Unique agent identifier.
        device_type: Target compute device.
        value: Agent's private value for this slot (already scaled by fitness).
        fitness: Current fitness score (used for audit / tie-breaking).
    """
    agent_id: str
    device_type: "DeviceType"
    value: float
    fitness: float

    def __repr__(self) -> str:
        return (
            f"Bid({self.agent_id!r}, {self.device_type.value}, "
            f"value={self.value:.3f}, fitness={self.fitness:.3f})"
        )


@dataclass(frozen=True)
class Allocation:
    """Result of the VCG auction for a single agent.

    Attributes:
        agent_id: Winning agent.
        device_type: Allocated device.
        price_paid: VCG price (not necessarily equal to bid value).
    """
    agent_id: str
    device_type: "DeviceType"
    price_paid: float

    def __repr__(self) -> str:
        return (
            f"Allocation({self.agent_id!r}, {self.device_type.value}, "
            f"price={self.price_paid:.3f})"
        )


class VCGAuction:
    """VCG combinatorial auction for GPU/CPU/iGPU/NPU slot allocation.

    Runs a separate multi-unit VCG auction per device type.
    Since each bid targets exactly one device type, the global optimum
    decomposes into independent per-device auctions.
    """

    def __init__(self, budget: "ThermalBudget") -> None:
        self.budget = budget

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def run_auction(self, bids: list[Bid]) -> dict[str, Allocation]:
        """Run VCG auction. Returns allocations with prices.

        bids: list of Bid(agent_id, device_type, value, fitness)
        returns: {agent_id: Allocation(device_type, price_paid)}
        """
        if not bids:
            return {}

        # Group bids by device type
        bids_by_device: dict["DeviceType", list[Bid]] = {}
        for bid in bids:
            bids_by_device.setdefault(bid.device_type, []).append(bid)

        allocations: dict[str, Allocation] = {}

        for device_type, device_bids in bids_by_device.items():
            max_slots = self.budget.device_budget(device_type).max_agents
            device_allocs = self._run_device_auction(device_type, device_bids, max_slots)
            allocations.update(device_allocs)

        return allocations

    def vcg_price(
        self,
        winner: Bid,
        all_bids: list[Bid],
        allocations_without_winner: dict[str, Allocation],
    ) -> float:
        """VCG price = social welfare without winner − social welfare of others with winner.

        For multi-unit identical slots this simplifies to the (S+1)th highest
        bid on the same device type, but we compute it from first principles
        so the formula is explicit.
        """
        # Social welfare on winner's device without winner
        device_bids = [b for b in all_bids if b.device_type == winner.device_type]
        max_slots = self.budget.device_budget(winner.device_type).max_agents

        # SW without winner = sum of top max_slots bids excluding winner
        other_bids = [b for b in device_bids if b.agent_id != winner.agent_id]
        sw_without_winner = self._top_k_value_sum(other_bids, max_slots)

        # SW of others with winner = sum of top max_slots bids minus winner's value
        all_device_values = sorted(
            (b.value for b in device_bids), reverse=True
        )
        top_k_sum = sum(all_device_values[:max_slots])
        sw_others_with_winner = top_k_sum - winner.value

        price = sw_without_winner - sw_others_with_winner
        return max(0.0, price)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _run_device_auction(
        self,
        device_type: "DeviceType",
        bids: list[Bid],
        max_slots: int,
    ) -> dict[str, Allocation]:
        """Run a single multi-unit VCG auction for one device type."""
        # Sort by value descending; tie-break by fitness descending for determinism
        sorted_bids = sorted(bids, key=lambda b: (b.value, b.fitness), reverse=True)

        winners = sorted_bids[:max_slots]
        losers = sorted_bids[max_slots:]

        # VCG price for each winner = highest losing bid's value (or 0 if no losers)
        # In multi-unit identical-item VCG, all winners pay the same price:
        # the value of the highest bid that did NOT win.
        # This is exactly the externality each winner imposes.
        highest_losing_value = losers[0].value if losers else 0.0

        allocations: dict[str, Allocation] = {}
        for winner in winners:
            # Verify via first-principles formula (should equal highest_losing_value)
            price = self.vcg_price(
                winner,
                bids,
                {},  # allocations_without_winner not needed for identical slots
            )
            # Clamp to highest_losing_value for numerical safety
            price = min(price, winner.value)
            price = max(0.0, price)
            allocations[winner.agent_id] = Allocation(
                agent_id=winner.agent_id,
                device_type=device_type,
                price_paid=price,
            )

        return allocations

    @staticmethod
    def _top_k_value_sum(bids: list[Bid], k: int) -> float:
        """Sum of top k bid values."""
        values = sorted((b.value for b in bids), reverse=True)
        return sum(values[:k])
