"""IP allowlist/denylist with CIDR support.

Manages IP allowlists and denylists with CIDR block support. Supports
individual IPs, CIDR ranges, and mixed lists. Used for fleet access
control, API rate limiting by IP, and network security.

Usage:
    acl = IPAllowlist()
    acl.allow("192.168.1.0/24")
    acl.deny("10.0.0.5")
    assert acl.is_allowed("192.168.1.100") is True
    assert acl.is_allowed("10.0.0.5") is False
"""
from __future__ import annotations

import ipaddress
from typing import Any, Dict, List, Optional, Union


class IPAllowlist:
    """
    IP allowlist/denylist with CIDR support.
    """

    def __init__(self, default_allow: bool = False):
        self._default_allow = default_allow
        self._allowed: List[Union[ipaddress.IPv4Network, ipaddress.IPv6Network]] = []
        self._denied: List[Union[ipaddress.IPv4Network, ipaddress.IPv6Network]] = []

    # ------------------------------------------------------------------
    # Allow / Deny
    # ------------------------------------------------------------------

    def allow(self, ip_or_cidr: str) -> bool:
        """
        Add an IP or CIDR to the allowlist.

        :param ip_or_cidr: IP address or CIDR string.
        :returns: True if added.
        """
        try:
            network = ipaddress.ip_network(ip_or_cidr, strict=False)
            self._allowed.append(network)
            return True
        except ValueError:
            return False

    def deny(self, ip_or_cidr: str) -> bool:
        """
        Add an IP or CIDR to the denylist.

        :param ip_or_cidr: IP address or CIDR string.
        :returns: True if added.
        """
        try:
            network = ipaddress.ip_network(ip_or_cidr, strict=False)
            self._denied.append(network)
            return True
        except ValueError:
            return False

    def remove_allow(self, ip_or_cidr: str) -> bool:
        """Remove from allowlist."""
        try:
            network = ipaddress.ip_network(ip_or_cidr, strict=False)
            if network in self._allowed:
                self._allowed.remove(network)
                return True
            return False
        except ValueError:
            return False

    def remove_deny(self, ip_or_cidr: str) -> bool:
        """Remove from denylist."""
        try:
            network = ipaddress.ip_network(ip_or_cidr, strict=False)
            if network in self._denied:
                self._denied.remove(network)
                return True
            return False
        except ValueError:
            return False

    # ------------------------------------------------------------------
    # Check
    # ------------------------------------------------------------------

    def is_allowed(self, ip_str: str) -> bool:
        """
        Check if IP is allowed.

        :param ip_str: IP address string.
        :returns: True if allowed.
        """
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            return False

        # Check denylist first (deny takes precedence)
        for network in self._denied:
            if ip in network:
                return False

        # Check allowlist
        if self._allowed:
            for network in self._allowed:
                if ip in network:
                    return True
            return False

        return self._default_allow

    def is_denied(self, ip_str: str) -> bool:
        """Check if IP is denied."""
        return not self.is_allowed(ip_str)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def allowed_list(self) -> List[str]:
        """List allowed networks as strings."""
        return [str(n) for n in self._allowed]

    def denied_list(self) -> List[str]:
        """List denied networks as strings."""
        return [str(n) for n in self._denied]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "allowed": len(self._allowed),
            "denied": len(self._denied),
            "default_allow": self._default_allow,
        }

    def __repr__(self) -> str:
        return f"<IPAllowlist allowed={len(self._allowed)} denied={len(self._denied)}>"
