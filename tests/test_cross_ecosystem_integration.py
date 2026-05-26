"""Cross-ecosystem integration smoke tests.

Verifies interactions between sunset-ecosystem and its sibling packages:
- constraint-theory-py
- fleet-math-c (via superinstance-ffi)
- PLATO tile store
- tensor-spline
- holonomy-consensus
- zerolang
- flux-check
- superinstance-runtime

All external packages are optional — tests skip gracefully when siblings
are not installed.
"""

from __future__ import annotations

import ctypes
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

# ── optional sibling packages ────────────────────────────
try:
    from constraint_theory import check_constraint
except ImportError:
    check_constraint = None  # type: ignore[assignment]

try:
    from flux_check import ConstraintEngine, get_preset
except ImportError:
    ConstraintEngine = None  # type: ignore[misc]
    get_preset = None  # type: ignore[assignment]

try:
    from nexus.holonomy_bridge import HolonomyBridge
except ImportError:
    HolonomyBridge = None  # type: ignore[misc]

try:
    from plato_core.types import TileType
except ImportError:
    TileType = None  # type: ignore[misc]

try:
    from sunset.agent import Agent, AgentPhase
except ImportError:
    Agent = None  # type: ignore[misc]
    AgentPhase = None  # type: ignore[misc]

try:
    from sunset.plato_bridge import PlatoBridge
except ImportError:
    PlatoBridge = None  # type: ignore[misc]

try:
    from sunset.seed_bank import SeedBank
except ImportError:
    SeedBank = None  # type: ignore[misc]

try:
    from sunset.sunset_documents import Epilogue, Onboarding
except ImportError:
    Epilogue = None  # type: ignore[misc]
    Onboarding = None  # type: ignore[misc]

try:
    from superinstance.runtime import EventBus
except ImportError:
    EventBus = None  # type: ignore[misc]

try:
    from superinstance.plugins.constraint import ConstraintCollector, ConstraintSelector, ConstraintCompiler
except ImportError:
    ConstraintCollector = None  # type: ignore[misc]
    ConstraintSelector = None  # type: ignore[misc]
    ConstraintCompiler = None  # type: ignore[misc]

try:
    from tensor_spline import SplineLinear, compression_ratio
except ImportError:
    SplineLinear = None  # type: ignore[misc]
    compression_ratio = None  # type: ignore[assignment]

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


# ── helpers ────────────────────────────────────────────────

def _load_superinstance_ffi() -> ctypes.CDLL:
    so_path = Path(__file__).parent.parent / "superinstance-ffi" / "target" / "release" / "libsuperinstance_ffi.so"
    lib = ctypes.CDLL(str(so_path))
    lib.laman_is_rigid.argtypes = [ctypes.c_uint, ctypes.c_uint]
    lib.laman_is_rigid.restype = ctypes.c_int
    lib.eisenstein_norm.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.eisenstein_norm.restype = ctypes.c_int
    return lib


# ── pytest skip helpers ─────────────────────────────────

_skip_no_plato = pytest.mark.skipif(PlatoBridge is None, reason="sunset.plato_bridge not installed")
_skip_no_tensor_spline = pytest.mark.skipif(SplineLinear is None, reason="tensor_spline not installed")
_skip_no_holonomy = pytest.mark.skipif(HolonomyBridge is None, reason="nexus.holonomy_bridge not installed")
_skip_no_sunset_agent = pytest.mark.skipif(Agent is None, reason="sunset.agent not installed")
_skip_no_constraint_theory = pytest.mark.skipif(check_constraint is None, reason="constraint_theory not installed")
_skip_no_flux_check = pytest.mark.skipif(ConstraintEngine is None, reason="flux_check not installed")
_skip_no_seed_bank = pytest.mark.skipif(SeedBank is None, reason="sunset.seed_bank not installed")
_skip_no_zerolang = pytest.mark.skipif(
    not Path("/home/phoenix/.openclaw/workspace/zerolang/bin/zero").exists(),
    reason="zerolang not installed",
)
_skip_no_superinstance = pytest.mark.skipif(EventBus is None, reason="superinstance.runtime not installed")


class TestCrossEcosystem:

    @_skip_no_plato
    def test_plato_tile_round_trip(self):
        """Write a trinity score tile and read it back intact."""
        bridge = PlatoBridge()
        tile = bridge.write_trinity_score("agent-42", 0.9, 0.8, 0.7)
        fetched = bridge.get_tile(tile.tile_id)
        assert fetched is not None
        desc = json.loads(fetched.description)
        assert desc["ethos"] == pytest.approx(0.9)
        assert desc["fitness"] == pytest.approx(0.9 * 0.8 * 0.7)

    @_skip_no_tensor_spline
    def test_tensor_spline_compress_decompress(self):
        """SplineLinear should have fewer params than dense and forward pass should work."""
        import torch as _torch
        dense = _torch.nn.Linear(64, 64)
        dense_params = sum(p.numel() for p in dense.parameters())
        spline = SplineLinear(64, 64, n_control_points=16)
        spline_params = sum(p.numel() for p in spline.parameters())
        assert spline_params < dense_params
        x = _torch.randn(4, 64)
        y = spline(x)
        assert y.shape == (4, 64)
        ratio = compression_ratio(spline)
        assert ratio["ratio"] > 1.0

    @_skip_no_holonomy
    def test_holonomy_identity_cycle(self):
        """A cycle with identical states must be consistent."""
        bridge = HolonomyBridge()
        for name in ["a", "b", "c"]:
            bridge.add_fleet_node(name, state=0.0)
        bridge.link("a", "b")
        bridge.link("b", "c")
        bridge.link("c", "a")
        report = bridge.verify_cycle(["a", "b", "c"])
        assert report.consistent
        assert report.holonomy_error == pytest.approx(0.0)

    @_skip_no_sunset_agent
    def test_sunset_lifecycle_transition(self):
        """Agent should advance through lifecycle phases."""
        agent = Agent(id="test", phase=AgentPhase.INCUBATING)
        agent.advance(AgentPhase.COMPETING)
        assert agent.phase == AgentPhase.COMPETING
        agent.advance(AgentPhase.BREEDING)
        assert agent.phase == AgentPhase.BREEDING
        agent.advance(AgentPhase.SUNSETTING)
        assert agent.phase == AgentPhase.SUNSETTING

    @_skip_no_constraint_theory
    def test_constraint_theory_py_satisfiable(self):
        """constraint-theory-py should recognize a lattice point as satisfiable."""
        assert check_constraint(0.0, 0.0) is True
        assert check_constraint(10.0, 10.0) is False

    @_skip_no_flux_check
    def test_flux_check_violations(self):
        """flux-check should detect out-of-bounds sensor readings."""
        engine = ConstraintEngine(get_preset("iot_mqtt"))
        # Normal readings
        normal = [20.0, 50.0, 1013.0, 400.0, 12.0, 300.0, 80.0, -50.0]
        result_normal = engine.check_vector(normal)
        assert result_normal.passed

        # Violate temperature (upper bound typically 60)
        hot = [100.0, 50.0, 1013.0, 400.0, 12.0, 300.0, 80.0, -50.0]
        result_hot = engine.check_vector(hot)
        assert not result_hot.passed
        assert any(v.name == "ambient_temp_c" for v in result_hot.violations)

    @_skip_no_seed_bank
    def test_seed_bank_cross_generation(self):
        """SeedBank should store and select across generations."""
        bank = SeedBank()
        for gen in range(3):
            onboarding = Onboarding(agent_id=f"a{gen}", generation=gen, letter_to_children=f"gen{gen}")
            bank.store(onboarding, relevance=0.5 + gen * 0.1, novelty=0.5)
        selected = bank.select(n=2, generation=1)
        assert len(selected) == 1
        assert selected[0].generation == 1

    def test_laman_rust_ffi_vs_python(self):
        """superinstance-ffi Laman result should match Python edge-count logic."""
        try:
            lib = _load_superinstance_ffi()
        except OSError:
            pytest.skip("superinstance-ffi shared library not built")
        for n in range(3, 20):
            need = 2 * n - 3
            # Exact count -> rigid
            assert lib.laman_is_rigid(n, need) == 1
            # One too few -> not rigid
            assert lib.laman_is_rigid(n, need - 1) == 0
            # One too many -> not rigid (minimal rigidity)
            assert lib.laman_is_rigid(n, need + 1) == 0

    @_skip_no_zerolang
    def test_zerolang_laman_output(self):
        """zerolang laman-rigidity package should print success."""
        zero_bin = Path("/home/phoenix/.openclaw/workspace/zerolang/bin/zero")
        main_file = Path("/home/phoenix/.openclaw/workspace/zerolang/packages/laman-rigidity/main.0")
        result = subprocess.run(
            [str(zero_bin), "run", str(main_file)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "rigid" in result.stdout.lower()

    @_skip_no_superinstance
    def test_superinstance_runtime_pipeline(self):
        """EventBus COLLECT → SELECT → COMPILE should run end-to-end."""
        bus = EventBus()
        bus.register_collector(ConstraintCollector())
        bus.register_selector(ConstraintSelector())
        bus.register_compiler(ConstraintCompiler())
        result = bus.run(
            {
                "constraints": [
                    {"field": "x", "value": 0.5, "lower_bound": 0, "upper_bound": 1},
                    {"field": "y", "value": 2.0, "lower_bound": 0, "upper_bound": 1},
                ]
            }
        )
        assert len(result.collected) == 2
        assert len(result.selected) == 1
        assert result.selected[0].field == "y"
        assert len(result.compiled) == 1
        assert result.compiled[0]["action"] == "decrease"
