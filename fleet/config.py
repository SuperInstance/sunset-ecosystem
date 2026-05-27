"""fleet/config.py — YAML configuration with environment variable overrides.

Cross-pollinated from ccc-os/config.py.  Extended for fleet-specific
breeding parameters, service endpoints, and notification channels.

Usage
-----
    from fleet.config import FleetConfig

    cfg = FleetConfig("config/sunset.yaml")
    print(cfg.breeding_pool_size)  # 50
    print(cfg.notification_channels)  # {"discord_webhook": "..."}
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG = {
    "fleet": {
        "data_dir": "./data",
        "log_level": "INFO",
        "breeding": {
            "pool_size": 50,
            "generation_limit": 1000,
            "mutation_rate": 0.1,
            "crossover_rate": 0.7,
            "elitism": 0.05,
            "latent_dim": 8,
        },
        "flux": {
            "pass_threshold": 0.35,
            "weight_bounds": [-5.0, 5.0],
            "max_l2_norm": 100.0,
            "max_variance": 10.0,
            "max_chaos": 1.0,
            "thermal_budget_gate": 0.8,
            "vm_so_path": "${FLUX_VM_SO}",
            "vm_scale": 1000,
            "vm_max_cycles": 4096,
        },
        "thermal": {
            "normal_threshold": 0.5,
            "elevated_threshold": 0.7,
            "critical_threshold": 0.9,
            "emergency_policy": "throttle",
        },
        "notifications": {
            "discord_webhook": "",
            "telegram_bot_token": "",
            "telegram_chat_id": "",
            "webhook_url": "",
            "alert_file": "",
        },
        "services": [
            {"name": "PLATO Gate", "host": "147.224.38.131", "port": 8847, "path": "/status"},
            {"name": "PLATO Shell", "host": "147.224.38.131", "port": 8848, "path": "/"},
            {"name": "MUD", "host": "147.224.38.131", "port": 4042, "path": "/status"},
            {"name": "Arena", "host": "147.224.38.131", "port": 4044, "path": "/status"},
            {"name": "Grammar", "host": "147.224.38.131", "port": 4045, "path": "/status"},
            {"name": "Skill Forge", "host": "147.224.38.131", "port": 4057, "path": "/status"},
        ],
        "api": {
            "host": "0.0.0.0",
            "port": 14002,
        },
        "mesh": {
            "gossip_interval": 30,
            "vector_sync_interval": 60,
            "metronome_drift_tolerance_ms": 500,
        },
    }
}

_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _resolve_env_vars(value: str) -> str:
    """Replace ${VAR} patterns with environment variable values."""
    def _replace(match):
        var_name = match.group(1)
        return os.environ.get(var_name, "")
    return _ENV_PATTERN.sub(_replace, value)


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base, returning new dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _resolve_strings(obj: Any) -> Any:
    """Recursively resolve ${ENV} patterns in string values."""
    if isinstance(obj, str):
        return _resolve_env_vars(obj)
    elif isinstance(obj, dict):
        return {k: _resolve_strings(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_resolve_strings(item) for item in obj]
    return obj


class FleetConfig:
    """Fleet configuration with YAML loading and environment overrides."""

    def __init__(self, config_path: str | Path | None = None):
        self._data = _DEFAULT_CONFIG.copy()
        self._config_dir = Path.cwd()

        if config_path:
            config_path = Path(config_path)
            if config_path.exists():
                self._config_dir = config_path.parent
                with open(config_path) as f:
                    loaded = yaml.safe_load(f)
                if loaded and isinstance(loaded, dict):
                    self._data = _deep_merge(self._data, loaded)

        # Apply environment variable overrides
        self._data = _resolve_strings(self._data)

    def get(self, *keys: str, default: Any = None) -> Any:
        """Get a nested config value by key path."""
        obj = self._data
        for key in keys:
            if isinstance(obj, dict) and key in obj:
                obj = obj[key]
            else:
                return default
        return obj

    # ── Paths ────────────────────────────────────────────────────

    @property
    def data_dir(self) -> Path:
        raw = self.get("fleet", "data_dir") or "./data"
        p = Path(raw)
        if not p.is_absolute():
            return (self._config_dir / p).resolve()
        return p

    @property
    def output_dir(self) -> Path:
        return self.data_dir / "output"

    @property
    def alert_file(self) -> Path:
        return self.data_dir / "alerts.jsonl"

    # ── Breeding ─────────────────────────────────────────────────

    @property
    def breeding_pool_size(self) -> int:
        return int(self.get("fleet", "breeding", "pool_size") or 50)

    @property
    def generation_limit(self) -> int:
        return int(self.get("fleet", "breeding", "generation_limit") or 1000)

    @property
    def mutation_rate(self) -> float:
        return float(self.get("fleet", "breeding", "mutation_rate") or 0.1)

    @property
    def crossover_rate(self) -> float:
        return float(self.get("fleet", "breeding", "crossover_rate") or 0.7)

    @property
    def elitism(self) -> float:
        return float(self.get("fleet", "breeding", "elitism") or 0.05)

    @property
    def latent_dim(self) -> int:
        return int(self.get("fleet", "breeding", "latent_dim") or 8)

    # ── FLUX ─────────────────────────────────────────────────────

    @property
    def flux_pass_threshold(self) -> float:
        return float(self.get("fleet", "flux", "pass_threshold") or 0.35)

    @property
    def flux_weight_bounds(self) -> tuple[float, float]:
        bounds = self.get("fleet", "flux", "weight_bounds") or [-5.0, 5.0]
        return (float(bounds[0]), float(bounds[1]))

    @property
    def flux_max_l2_norm(self) -> float:
        return float(self.get("fleet", "flux", "max_l2_norm") or 100.0)

    @property
    def flux_max_variance(self) -> float:
        return float(self.get("fleet", "flux", "max_variance") or 10.0)

    @property
    def flux_max_chaos(self) -> float:
        return float(self.get("fleet", "flux", "max_chaos") or 1.0)

    @property
    def flux_thermal_budget_gate(self) -> float:
        return float(self.get("fleet", "flux", "thermal_budget_gate") or 0.8)

    @property
    def flux_vm_so_path(self) -> str | None:
        path = self.get("fleet", "flux", "vm_so_path")
        return path if path else None

    @property
    def flux_vm_scale(self) -> int:
        return int(self.get("fleet", "flux", "vm_scale") or 1000)

    @property
    def flux_vm_max_cycles(self) -> int:
        return int(self.get("fleet", "flux", "vm_max_cycles") or 4096)

    # ── Thermal ──────────────────────────────────────────────────

    @property
    def thermal_normal(self) -> float:
        return float(self.get("fleet", "thermal", "normal_threshold") or 0.5)

    @property
    def thermal_elevated(self) -> float:
        return float(self.get("fleet", "thermal", "elevated_threshold") or 0.7)

    @property
    def thermal_critical(self) -> float:
        return float(self.get("fleet", "thermal", "critical_threshold") or 0.9)

    @property
    def thermal_emergency_policy(self) -> str:
        return self.get("fleet", "thermal", "emergency_policy") or "throttle"

    # ── Notifications ────────────────────────────────────────────

    @property
    def notification_channels(self) -> dict[str, str]:
        return self.get("fleet", "notifications") or {}

    # ── Services ─────────────────────────────────────────────────

    def health_services(self) -> list[dict]:
        return self.get("fleet", "services") or []

    # ── API ──────────────────────────────────────────────────────

    @property
    def api_host(self) -> str:
        return self.get("fleet", "api", "host") or "0.0.0.0"

    @property
    def api_port(self) -> int:
        return int(self.get("fleet", "api", "port") or 14002)

    # ── Mesh ─────────────────────────────────────────────────────

    @property
    def mesh_gossip_interval(self) -> int:
        return int(self.get("fleet", "mesh", "gossip_interval") or 30)

    @property
    def mesh_vector_sync_interval(self) -> int:
        return int(self.get("fleet", "mesh", "vector_sync_interval") or 60)

    @property
    def mesh_metronome_drift_tolerance_ms(self) -> int:
        return int(self.get("fleet", "mesh", "metronome_drift_tolerance_ms") or 500)

    # ── Serialization ────────────────────────────────────────────

    def as_dict(self) -> dict:
        return self._data.copy()

    @classmethod
    def from_dict(cls, data: dict) -> "FleetConfig":
        """Create a FleetConfig from a dict (for testing)."""
        cfg = cls.__new__(cls)
        cfg._data = _deep_merge(_DEFAULT_CONFIG, data)
        cfg._config_dir = Path.cwd()
        return cfg


# Module-level singleton
_default_config: FleetConfig | None = None


def get_config(config_path: str | Path | None = None) -> FleetConfig:
    """Get or create the default config singleton."""
    global _default_config
    if _default_config is None or config_path is not None:
        _default_config = FleetConfig(config_path)
    return _default_config
