"""fleet/ — Fleet infrastructure: notifications, config, decks, CLI, health checks."""

from fleet.config import FleetConfig, get_config
from fleet.health_check import FleetHealthChecker, ServiceDef, CheckResult
from fleet.notifier import FleetNotifier, BreedingAlert
from fleet.deck import Deck, Slide, breeding_report, fleet_status, flux_gate_decision

__all__ = [
    "FleetConfig",
    "get_config",
    "FleetHealthChecker",
    "ServiceDef",
    "CheckResult",
    "FleetNotifier",
    "BreedingAlert",
    "Deck",
    "Slide",
    "breeding_report",
    "fleet_status",
    "flux_gate_decision",
]
