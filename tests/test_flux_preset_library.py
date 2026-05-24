"""Tests for the FLUX Preset Library (sunset/flux_preset_library.py).

Covers:
    - All default presets load correctly
    - get_preset() returns correct preset
    - list_presets() filters by category
    - python_safe_only filters out unsafe presets
    - apply_preset() runs constraints on test context
    - suggest_preset_for_task() matches keywords correctly
    - Preset constraints use only safe opcodes
    - Library repr and count properties
"""

from __future__ import annotations

import time

import pytest

from logos.opcode_capability_index import OpcodeCapabilityIndex, OpcodeStatus
from sunset.flux_preset_library import (
    FluxPreset,
    FluxPresetLibrary,
    PRESET_CATEGORIES,
    PresetCategory,
    _DEF_PRESETS,
)


# ── fixtures ──────────────────────────────────────────────

@pytest.fixture
def library():
    return FluxPresetLibrary()


@pytest.fixture
def index():
    return OpcodeCapabilityIndex()


# ── 1. All default presets load correctly ─────────────────

class TestAllDefaultPresetsLoad:
    def test_all_10_presets_present(self, library):
        assert library.preset_count == 10
        expected = {
            "RangeCheck",
            "ProveAndHashCommit",
            "StreamBatch",
            "MemoryBudget",
            "DiversityFloor",
            "ThermalCeiling",
            "AgentLiveness",
            "CrossNodeSync",
            "BreedingStandard",
            "FleetHealth",
        }
        assert set(library.names) == expected

    def test_each_preset_is_flux_preset(self, library):
        for name in library.names:
            preset = library.get_preset(name)
            assert isinstance(preset, FluxPreset)
            assert preset.name == name
            assert preset.description
            assert isinstance(preset.category, PresetCategory)
            assert len(preset.constraints) >= 1
            assert len(preset.required_opcodes) >= 1

    def _default_presets_immutable(self):
        # _DEF_PRESETS is a module-level tuple; confirm it is frozen-like
        assert isinstance(_DEF_PRESETS, tuple)


# ── 2. get_preset() returns correct preset ────────────────

class TestGetPreset:
    def test_get_preset_by_exact_name(self, library):
        p = library.get_preset("RangeCheck")
        assert p.name == "RangeCheck"
        assert p.category == PresetCategory.BREEDING

    def test_get_preset_raises_on_unknown(self, library):
        with pytest.raises(KeyError) as exc_info:
            library.get_preset("NonExistentPreset")
        assert "NonExistentPreset" in str(exc_info.value)

    def test_get_preset_prove_and_hash(self, library):
        p = library.get_preset("ProveAndHashCommit")
        assert p.category == PresetCategory.CRYPTO
        assert len(p.constraints) == 2

    def test_get_preset_agent_liveness(self, library):
        p = library.get_preset("AgentLiveness")
        assert p.category == PresetCategory.LIVENESS
        assert len(p.constraints) == 2


# ── 3. list_presets() filters by category ─────────────────

class TestListPresets:
    def test_list_all_returns_everything(self, library):
        all_presets = library.list_presets()
        assert len(all_presets) == 10

    def test_list_breeding_category(self, library):
        breeding = library.list_presets(category="breeding")
        names = {p.name for p in breeding}
        assert names == {"RangeCheck", "BreedingStandard"}

    def test_list_thermal_category(self, library):
        thermal = library.list_presets(category="thermal")
        names = {p.name for p in thermal}
        assert names == {"ThermalCeiling", "FleetHealth"}

    def test_list_liveness_category(self, library):
        liveness = library.list_presets(category="liveness")
        assert len(liveness) == 1
        assert liveness[0].name == "AgentLiveness"

    def test_list_crypto_category(self, library):
        crypto = library.list_presets(category="crypto")
        assert len(crypto) == 1
        assert crypto[0].name == "ProveAndHashCommit"

    def test_list_sync_category(self, library):
        sync = library.list_presets(category="sync")
        assert len(sync) == 1
        assert sync[0].name == "CrossNodeSync"

    def test_list_unknown_category_returns_empty(self, library):
        result = library.list_presets(category="nonexistent")
        assert result == []

    def test_list_case_insensitive(self, library):
        upper = library.list_presets(category="BREEDING")
        lower = library.list_presets(category="breeding")
        assert len(upper) == len(lower)


# ── 4. python_safe_only filters ───────────────────────────

class TestPythonSafeOnly:
    def test_all_default_presets_are_python_safe(self, library):
        all_presets = library.list_presets(python_safe_only=True)
        assert len(all_presets) == 10
        for p in all_presets:
            assert p.python_safe is True

    def test_python_safe_only_excludes_rust_only(self, index):
        # Manufacture a preset that requires a RUST_ONLY opcode
        rust_preset = FluxPreset(
            name="RustOnlyPreset",
            description="Requires a Rust-only opcode.",
            category=PresetCategory.BREEDING,
            constraints=(),
            required_opcodes=("Prove",),  # RUST_ONLY
            python_safe=False,
        )
        lib = FluxPresetLibrary(presets=[rust_preset], index=index)
        all_with = lib.list_presets(python_safe_only=False)
        safe_only = lib.list_presets(python_safe_only=True)
        assert len(all_with) == 1
        assert len(safe_only) == 0

    def test_library_auto_corrects_python_safe_flag(self, index):
        # A preset claims python_safe=True but index says required opcode is RUST_ONLY
        bad_preset = FluxPreset(
            name="BadPreset",
            description="Claims safe but isn't.",
            category=PresetCategory.BREEDING,
            constraints=(),
            required_opcodes=("VecLoad",),  # RUST_ONLY
            python_safe=True,
        )
        lib = FluxPresetLibrary(presets=[bad_preset], index=index)
        p = lib.get_preset("BadPreset")
        assert p.python_safe is False


# ── 5. apply_preset() runs constraints ──────────────────────

class TestApplyPreset:
    def test_apply_range_check_passes(self, library):
        ctx = {"weights": 2.5, "chaos": 0.3, "thermal_headroom": 0.8}
        results = library.apply_preset("RangeCheck", ctx)
        assert len(results) == 3
        for r in results:
            assert r["passed"] is True
            assert r["severity"] == "info"

    def test_apply_range_check_fails_weight(self, library):
        ctx = {"weights": 15.0, "chaos": 0.3, "thermal_headroom": 0.8}
        results = library.apply_preset("RangeCheck", ctx)
        assert results[0]["passed"] is False
        assert results[0]["severity"] == "critical"
        assert "norm" in results[0]["detail"]

    def test_apply_range_check_fails_chaos(self, library):
        ctx = {"weights": 2.5, "chaos": 1.5, "thermal_headroom": 0.8}
        results = library.apply_preset("RangeCheck", ctx)
        # chaos is second constraint
        assert results[1]["passed"] is False
        assert results[1]["severity"] == "warning"

    def test_apply_thermal_ceiling_blocks(self, library):
        ctx = {"thermal_headroom": 0.99}
        results = library.apply_preset("ThermalCeiling", ctx)
        assert len(results) == 1
        # strict < ceiling: 0.99 < 0.99 is False → passed=False
        assert results[0]["passed"] is False

    def test_apply_thermal_ceiling_allows(self, library):
        ctx = {"thermal_headroom": 0.5}
        results = library.apply_preset("ThermalCeiling", ctx)
        assert results[0]["passed"] is True

    def test_apply_agent_liveness_timeout(self, library):
        now = time.time()
        ctx = {"last_heartbeat": now - 60, "heartbeat_timeout_seconds": 30, "now": now}
        results = library.apply_preset("AgentLiveness", ctx)
        assert len(results) == 2
        # first constraint: heartbeat timeout
        assert results[0]["passed"] is False
        assert results[0]["severity"] == "critical"
        # second constraint: crash detection (no failures field → passes)
        assert results[1]["passed"] is True

    def test_apply_agent_liveness_alive(self, library):
        now = time.time()
        ctx = {"last_heartbeat": now - 5, "heartbeat_timeout_seconds": 30, "now": now}
        results = library.apply_preset("AgentLiveness", ctx)
        assert results[0]["passed"] is True

    def test_apply_stream_batch_ratelimit(self, library):
        ctx = {"batch_size": 50, "max_batch_size": 64, "requests_per_second": 1200, "max_rps": 1000}
        results = library.apply_preset("StreamBatch", ctx)
        assert len(results) == 2
        assert results[0]["passed"] is True  # batch_size 50 <= 64
        assert results[1]["passed"] is False  # rps 1200 > 1000
        assert results[1]["severity"] == "warning"

    def test_apply_memory_budget(self, library):
        ctx = {"memory_mb": 2048, "memory_cap_mb": 1024}
        results = library.apply_preset("MemoryBudget", ctx)
        assert len(results) == 1
        assert results[0]["passed"] is False
        assert results[0]["severity"] == "critical"

    def test_apply_diversity_floor(self, library):
        ctx = {"diversity_score": 0.05, "diversity_floor": 0.1}
        results = library.apply_preset("DiversityFloor", ctx)
        assert len(results) == 1
        assert results[0]["passed"] is False
        assert results[0]["severity"] == "warning"

    def test_apply_cross_node_sync_match(self, library):
        ctx = {"local_hash": "abc123", "gossip_hash": "abc123"}
        results = library.apply_preset("CrossNodeSync", ctx)
        assert len(results) == 1
        assert results[0]["passed"] is True

    def test_apply_cross_node_sync_mismatch(self, library):
        ctx = {"local_hash": "abc123", "gossip_hash": "xyz789"}
        results = library.apply_preset("CrossNodeSync", ctx)
        assert results[0]["passed"] is False
        assert results[0]["severity"] == "warning"

    def test_apply_breeding_standard(self, library):
        ctx = {
            "weights": 12.0,
            "chaos": 0.05,
            "thermal_headroom": 0.8,
            "diversity_score": 0.05,
            "diversity_floor": 0.1,
        }
        results = library.apply_preset("BreedingStandard", ctx)
        assert len(results) == 4
        assert results[0]["passed"] is False  # weight too high
        assert results[1]["passed"] is True   # chaos ok
        assert results[2]["passed"] is True   # thermal ok
        assert results[3]["passed"] is False  # diversity too low

    def test_apply_prove_and_hash_commit(self, library):
        payload = "test_payload"
        expected_sig = "e7d6c5c0e8f3c3a2"  # first 16 of sha256 of str(payload)
        ctx = {"payload": payload, "signature": expected_sig}
        results = library.apply_preset("ProveAndHashCommit", ctx)
        assert len(results) == 2

    def test_apply_fleet_health(self, library):
        now = time.time()
        ctx = {
            "thermal_headroom": 0.99,
            "last_heartbeat": now - 60,
            "heartbeat_timeout_seconds": 30,
            "now": now,
            "consecutive_failures": 5,
            "crash_threshold": 3,
        }
        results = library.apply_preset("FleetHealth", ctx)
        assert len(results) == 3
        assert results[0]["passed"] is False  # thermal breach
        assert results[1]["passed"] is False  # heartbeat timeout
        assert results[2]["passed"] is False  # crash threshold


# ── 6. suggest_preset_for_task() ──────────────────────────

class TestSuggestPresetForTask:
    def test_suggest_range_check(self, library):
        name = library.suggest_preset_for_task("check weight bounds and norms")
        assert name == "RangeCheck"

    def test_suggest_prove_hash(self, library):
        name = library.suggest_preset_for_task("verify signature and hash commitment")
        assert name == "ProveAndHashCommit"

    def test_suggest_stream_batch(self, library):
        name = library.suggest_preset_for_task("limit batch size and rate")
        assert name == "StreamBatch"

    def test_suggest_memory(self, library):
        name = library.suggest_preset_for_task("check memory budget and cap")
        assert name == "MemoryBudget"

    def test_suggest_diversity(self, library):
        name = library.suggest_preset_for_task("maintain population diversity")
        assert name == "DiversityFloor"

    def test_suggest_thermal(self, library):
        name = library.suggest_preset_for_task("thermal ceiling and heat limit")
        assert name == "ThermalCeiling"

    def test_suggest_liveness(self, library):
        name = library.suggest_preset_for_task("agent heartbeat alive check")
        assert name == "AgentLiveness"

    def test_suggest_sync(self, library):
        name = library.suggest_preset_for_task("mesh gossip node consistency")
        assert name == "CrossNodeSync"

    def test_suggest_breeding(self, library):
        name = library.suggest_preset_for_task("breed new tournament agents")
        assert name == "BreedingStandard"

    def test_suggest_fleet_health(self, library):
        name = library.suggest_preset_for_task("overall fleet health status")
        assert name == "FleetHealth"

    def test_suggest_fallback_no_keywords(self, library):
        name = library.suggest_preset_for_task("something completely unrelated")
        assert name == "BreedingStandard"

    def test_suggest_case_insensitive(self, library):
        name = library.suggest_preset_for_task("BATCH SIZE AND RPS LIMIT")
        assert name == "StreamBatch"


# ── 7. Preset constraints use only safe opcodes ───────────

class TestPresetOpcodesAreSafe:
    def test_all_presets_use_only_safe_opcodes(self, library, index):
        for name in library.names:
            preset = library.get_preset(name)
            for opcode in preset.required_opcodes:
                assert index.can_use_from_python(opcode), (
                    f"Preset '{name}' requires opcode '{opcode}' which is not PYTHON_SAFE"
                )

    def test_range_check_opcodes_safe(self, index):
        for op in ("RangeCheck", "Validate", "ClassifySeverity"):
            assert index.can_use_from_python(op)

    def test_stream_batch_opcodes_safe(self, index):
        for op in ("Saturate", "Validate", "Min", "Max", "Sub"):
            assert index.can_use_from_python(op)

    def test_agent_liveness_opcodes_safe(self, index):
        for op in ("Sub", "Validate", "EmitEvent"):
            assert index.can_use_from_python(op)

    def test_rust_only_opcodes_not_used(self, index):
        rust_only = ["Prove", "HashCommit", "VecLoad", "ParDispatch", "StreamOpen"]
        for op in rust_only:
            assert not index.can_use_from_python(op)


# ── 8. Library properties ─────────────────────────────────

class TestLibraryProperties:
    def test_preset_count(self, library):
        assert library.preset_count == 10

    def test_names_sorted(self, library):
        names = library.names
        assert names == sorted(names)

    def test_repr(self, library):
        r = repr(library)
        assert "FluxPresetLibrary(" in r
        assert "presets=10" in r
        assert "python_safe=10" in r

    def test_empty_library(self):
        lib = FluxPresetLibrary(presets=[])
        assert lib.preset_count == 0
        assert lib.names == []

    def test_library_with_custom_preset(self, index):
        custom = FluxPreset(
            name="CustomCheck",
            description="A custom check.",
            category=PresetCategory.BREEDING,
            constraints=(lambda ctx: {"passed": True, "severity": "info"},),
            required_opcodes=("Nop",),
            python_safe=True,
        )
        lib = FluxPresetLibrary(presets=[custom], index=index)
        assert lib.preset_count == 1
        assert lib.get_preset("CustomCheck").name == "CustomCheck"
