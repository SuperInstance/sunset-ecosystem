"""Federated Nexus — Fleet-wide agent registration and discovery.

Agents register with a central nexus to discover peers, share load,
and propagate seed-bank archives across the fleet.
"""

__version__ = "0.1.0"
__all__ = [
    "FederatedNexus",
    "RegistrationRecord",
    "FederationEndpoint",
    "NexusError",
    "ConnectionRefusedError",
]
