"""Tests for FleetConfig — YAML configuration with environment overrides.

Covers default config, YAML loading, deep merge, env var substitution,
property access, and nested get().
"""

import os
from pathlib import Path

import pytest
import yaml

from fleet.config import FleetConfig, _deep_merge, _resolve_env_vars, _resolve_strings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_resolve_env_vars(self, monkeypatch):
        monkeypatch.setenv("TEST_VAR", "hello")
        assert _resolve_env_vars("${TEST_VAR}") == "hello"
        assert _resolve_env_vars("prefix_${TEST_VAR}_suffix") == "prefix_hello_suffix"

    def test_resolve_env_vars_missing(self):
        assert _resolve_env_vars("${MISSING_VAR}") == ""

    def test_deep_merge(self):
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"c": 99}}
        result = _deep_merge(base, override)
        assert result["a"] == 1
        assert result["b"]["c"] == 99
        assert result["b"]["d"] == 3

    def test_resolve_strings_nested(self, monkeypatch):
        monkeypatch.setenv("FOO", "bar")
        obj = {"a": "${FOO}", "b": ["${FOO}", 1], "c": {"d": "${FOO}"}}
        result = _resolve_strings(obj)
        assert result["a"] == "bar"
        assert result["b"][0] == "bar"
        assert result["c"]["d"] == "bar"


# ---------------------------------------------------------------------------
# FleetConfig defaults
# ---------------------------------------------------------------------------


class TestFleetConfigDefaults:
    def test_no_path(self):
        cfg = FleetConfig()
        assert cfg.breeding_pool_size == 50
        assert cfg.generation_limit == 1000
        assert cfg.mutation_rate == pytest.approx(0.1)
        assert cfg.crossover_rate == pytest.approx(0.7)
        assert cfg.elitism == pytest.approx(0.05)
        assert cfg.latent_dim == 8

    def test_flux_defaults(self):
        cfg = FleetConfig()
        assert cfg.flux_pass_threshold == pytest.approx(0.35)
        assert cfg.flux_weight_bounds == (-5.0, 5.0)
        assert cfg.flux_max_l2_norm == 100.0
        assert cfg.flux_max_variance == 10.0
        assert cfg.flux_max_chaos == 1.0
        assert cfg.flux_thermal_budget_gate == 0.8

    def test_thermal_defaults(self):
        cfg = FleetConfig()
        assert cfg.thermal_normal == pytest.approx(0.5)
        assert cfg.thermal_elevated == pytest.approx(0.7)
        assert cfg.thermal_critical == pytest.approx(0.9)
        assert cfg.thermal_emergency_policy == "throttle"

    def test_notification_channels(self):
        cfg = FleetConfig()
        channels = cfg.notification_channels
        assert isinstance(channels, dict)

    def test_health_services(self):
        cfg = FleetConfig()
        services = cfg.health_services()
        assert len(services) >= 1
        assert all("name" in s and "host" in s for s in services)

    def test_api_defaults(self):
        cfg = FleetConfig()
        assert cfg.api_host == "0.0.0.0"
        assert cfg.api_port == 14002

    def test_data_dir(self):
        cfg = FleetConfig()
        assert isinstance(cfg.data_dir, Path)
        assert cfg.output_dir == cfg.data_dir / "output"
        assert cfg.alert_file == cfg.data_dir / "alerts.jsonl"


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------


class TestFleetConfigYAML:
    def test_yaml_override(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"fleet": {"breeding": {"pool_size": 99}}}))
        cfg = FleetConfig(config_file)
        assert cfg.breeding_pool_size == 99

    def test_yaml_deep_merge(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"fleet": {"flux": {"pass_threshold": 0.99}}}))
        cfg = FleetConfig(config_file)
        assert cfg.flux_pass_threshold == pytest.approx(0.99)
        # Other flux defaults still present
        assert cfg.flux_max_l2_norm == 100.0

    def test_yaml_env_substitution(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MY_PORT", "9999")
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"fleet": {"api": {"port": "${MY_PORT}"}}}))
        cfg = FleetConfig(config_file)
        assert cfg.api_port == 9999

    def test_missing_file(self):
        cfg = FleetConfig("/nonexistent/config.yaml")
        assert cfg.breeding_pool_size == 50  # falls back to defaults


# ---------------------------------------------------------------------------
# get() method
# ---------------------------------------------------------------------------


class TestFleetConfigGet:
    def test_get_nested(self):
        cfg = FleetConfig()
        assert cfg.get("fleet", "breeding", "pool_size") == 50

    def test_get_missing(self):
        cfg = FleetConfig()
        assert cfg.get("fleet", "missing", "key") is None

    def test_get_with_default(self):
        cfg = FleetConfig()
        assert cfg.get("fleet", "missing", "key", default="fallback") == "fallback"

    def test_get_shallow(self):
        cfg = FleetConfig()
        assert isinstance(cfg.get("fleet"), dict)
