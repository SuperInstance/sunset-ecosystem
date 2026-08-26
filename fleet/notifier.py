"""fleet/notifier.py — Multi-channel notification system for breeding alerts.

Cross-pollinated from ccc-os/notifier.py.  Extended for fleet-specific
scenarios: thermal emergencies, FLUX gate blocks, breeding failures,
and proof certificate generation.

Usage
-----
    from fleet.notifier import FleetNotifier, BreedingAlert

    n = FleetNotifier()
    n.add_discord(os.environ["FLEET_DISCORD_WEBHOOK"])
    n.add_file("./data/alerts.jsonl")

    alert = BreedingAlert.thermal_critical(pressure=0.95, source="gpu-0")
    n.send(alert)
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BreedingAlert:
    """A fleet-specific alert notification."""

    title: str
    body: str
    severity: str  # info, warning, critical
    category: str  # thermal, flux_gate, breeding, proof, health
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "body": self.body,
            "severity": self.severity,
            "category": self.category,
            "timestamp": self.timestamp,
            **self.metadata,
        }

    # ── Factory methods for common fleet scenarios ─────────────────

    @classmethod
    def thermal_critical(cls, pressure: float, source: str) -> "BreedingAlert":
        return cls(
            title="🔥 THERMAL CRITICAL",
            body=f"Thermal pressure {pressure:.2%} from {source}. Emergency breeding policies active.",
            severity="critical",
            category="thermal",
            metadata={"pressure": pressure, "source": source},
        )

    @classmethod
    def flux_gate_block(
        cls, candidate_id: str, violations: dict[str, float]
    ) -> "BreedingAlert":
        vstr = ", ".join(f"{k}={v:.3f}" for k, v in violations.items())
        return cls(
            title="🛡️ FLUX Gate Block",
            body=f"Candidate {candidate_id} rejected. Violations: {vstr}",
            severity="warning",
            category="flux_gate",
            metadata={"candidate_id": candidate_id, "violations": violations},
        )

    @classmethod
    def proof_generated(
        cls, candidate_id: str, proof_hash: str, cycles: int
    ) -> "BreedingAlert":
        return cls(
            title="✅ Proof Certificate Generated",
            body=f"Candidate {candidate_id} passed VM gating. Cycles: {cycles}, Hash: {proof_hash[:16]}...",
            severity="info",
            category="proof",
            metadata={
                "candidate_id": candidate_id,
                "proof_hash": proof_hash,
                "cycles": cycles,
            },
        )

    @classmethod
    def breeding_failure(cls, error: str, generation: int) -> "BreedingAlert":
        return cls(
            title="❌ Breeding Failure",
            body=f"Generation {generation} failed: {error}",
            severity="critical",
            category="breeding",
            metadata={"generation": generation, "error": error},
        )

    @classmethod
    def service_down(cls, service: str, host: str, port: int) -> "BreedingAlert":
        return cls(
            title=f"⚠️ Service Down: {service}",
            body=f"{service} at {host}:{port} is unreachable.",
            severity="warning",
            category="health",
            metadata={"service": service, "host": host, "port": port},
        )


# ═══════════════════════════════════════════════════════════════
# Channels
# ═══════════════════════════════════════════════════════════════


class Channel:
    """Base notification channel."""

    def __init__(self, name: str):
        self.name = name

    def send(self, alert: BreedingAlert) -> bool:
        raise NotImplementedError


class DiscordChannel(Channel):
    """Discord webhook notifications with severity-colored embeds."""

    def __init__(self, webhook_url: str):
        super().__init__("discord")
        self.webhook_url = webhook_url

    def send(self, alert: BreedingAlert) -> bool:
        if not self.webhook_url:
            return False
        try:
            color = {
                "critical": 15158332,  # red
                "warning": 16776960,  # yellow
                "info": 3447003,  # blue
            }.get(alert.severity, 3447003)

            payload = {
                "embeds": [
                    {
                        "title": alert.title,
                        "description": alert.body[:2000],
                        "color": color,
                        "timestamp": alert.timestamp,
                        "footer": {"text": f"Cocapn Fleet | {alert.category}"},
                    }
                ]
            }
            data = json.dumps(payload).encode()
            req = urllib.request.Request(
                self.webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            resp = urllib.request.urlopen(req, timeout=10)
            return resp.status in (200, 204)
        except Exception as e:
            logger.warning("Discord notification failed: %s", e)
            return False


class TelegramChannel(Channel):
    """Telegram bot API notifications."""

    def __init__(self, bot_token: str, chat_id: str):
        super().__init__("telegram")
        self.bot_token = bot_token
        self.chat_id = chat_id

    def send(self, alert: BreedingAlert) -> bool:
        if not self.bot_token or not self.chat_id:
            return False
        try:
            emoji = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(
                alert.severity, "ℹ️"
            )
            text = f"{emoji} *{alert.title}*\n\n{alert.body[:4000]}"
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {"chat_id": self.chat_id, "text": text, "parse_mode": "Markdown"}
            data = json.dumps(payload).encode()
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            resp = urllib.request.urlopen(req, timeout=10)
            return resp.status == 200
        except Exception as e:
            logger.warning("Telegram notification failed: %s", e)
            return False


class FileChannel(Channel):
    """Append alerts to a JSONL file."""

    def __init__(self, file_path: str | Path):
        super().__init__("file")
        self.file_path = Path(file_path)

    def send(self, alert: BreedingAlert) -> bool:
        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.file_path, "a") as f:
                f.write(json.dumps(alert.to_dict(), default=str) + "\n")
            return True
        except Exception as e:
            logger.warning("File notification failed: %s", e)
            return False


class WebhookChannel(Channel):
    """Generic webhook notifications."""

    def __init__(self, url: str):
        super().__init__("webhook")
        self.url = url

    def send(self, alert: BreedingAlert) -> bool:
        if not self.url:
            return False
        try:
            data = json.dumps(alert.to_dict(), default=str).encode()
            req = urllib.request.Request(
                self.url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            resp = urllib.request.urlopen(req, timeout=10)
            return 200 <= resp.status < 300
        except Exception as e:
            logger.warning("Webhook notification failed: %s", e)
            return False


class SSEChannel(Channel):
    """Push alerts to the SSE Stream Dashboard (in-memory buffer)."""

    def __init__(self, dashboard: Any | None = None):
        super().__init__("sse")
        self._dashboard = dashboard

    def send(self, alert: BreedingAlert) -> bool:
        if self._dashboard is None:
            return False
        try:
            if hasattr(self._dashboard, "push_event"):
                self._dashboard.push_event("ALERT", alert.to_dict())
                return True
        except Exception as e:
            logger.warning("SSE notification failed: %s", e)
        return False


# ═══════════════════════════════════════════════════════════════
# Fleet Notifier — Multi-channel dispatcher
# ═══════════════════════════════════════════════════════════════


class FleetNotifier:
    """Multi-channel notification dispatcher with fleet-specific routing.

    Routes alerts by category:
    - thermal → discord + file (always)
    - flux_gate → discord + file
    - breeding → all channels
    - proof → file only (verbose, not urgent)
    - health → discord + file
    """

    def __init__(self):
        self.channels: list[Channel] = []
        self._category_routes: dict[str, list[str]] = {
            "thermal": ["discord", "file", "sse"],
            "flux_gate": ["discord", "file", "sse"],
            "breeding": ["discord", "telegram", "file", "webhook", "sse"],
            "proof": ["file"],
            "health": ["discord", "file", "sse"],
        }

    def add_channel(self, channel: Channel) -> None:
        self.channels.append(channel)

    def add_discord(self, webhook_url: str) -> None:
        self.add_channel(DiscordChannel(webhook_url))

    def add_telegram(self, bot_token: str, chat_id: str) -> None:
        self.add_channel(TelegramChannel(bot_token, chat_id))

    def add_file(self, file_path: str | Path) -> None:
        self.add_channel(FileChannel(file_path))

    def add_webhook(self, url: str) -> None:
        self.add_channel(WebhookChannel(url))

    def add_sse(self, dashboard: Any | None = None) -> None:
        self.add_channel(SSEChannel(dashboard))

    def send(self, alert: BreedingAlert) -> dict[str, bool]:
        """Send alert to all matching channels. Returns per-channel results."""
        allowed = self._category_routes.get(
            alert.category, [c.name for c in self.channels]
        )
        results = {}
        for channel in self.channels:
            if channel.name not in allowed:
                continue
            try:
                results[channel.name] = channel.send(alert)
            except Exception as e:
                logger.warning("Channel %s failed: %s", channel.name, e)
                results[channel.name] = False
        return results

    def send_simple(
        self, title: str, body: str, severity: str = "info", category: str = "breeding"
    ) -> dict[str, bool]:
        """Convenience method for simple alerts."""
        return self.send(BreedingAlert(title, body, severity, category))

    @classmethod
    def from_config(
        cls, config: dict[str, str], data_dir: Path | None = None
    ) -> "FleetNotifier":
        """Create a FleetNotifier from config dict.

        Config keys: discord_webhook, telegram_bot_token, telegram_chat_id,
                     webhook_url, alert_file
        """
        notifier = cls()

        if config.get("discord_webhook"):
            notifier.add_discord(config["discord_webhook"])

        if config.get("telegram_bot_token") and config.get("telegram_chat_id"):
            notifier.add_telegram(
                config["telegram_bot_token"], config["telegram_chat_id"]
            )

        if config.get("webhook_url"):
            notifier.add_webhook(config["webhook_url"])

        alert_file = config.get("alert_file")
        if alert_file:
            notifier.add_file(alert_file)
        elif data_dir:
            notifier.add_file(data_dir / "alerts.jsonl")

        return notifier
