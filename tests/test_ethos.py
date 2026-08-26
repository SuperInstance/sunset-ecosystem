"""Tests for the ETHOS room."""

from __future__ import annotations

import pytest

from ethos.hardware_survey import (
    HardwareProfile,
    CPUInfo,
    MemoryInfo,
    CudaGPU,
    ThermalZone,
    survey_hardware,
)
from ethos.stress_test import StressReport, DeviceBenchmark, run_stress_test
from ethos.agent_allocator import AllocationPlan, build_allocation_plan
from ethos.trinity_connection import score_ethos_connection, EthosConnectionScore


# ── Fixtures ──────────────────────────────────────────────────────────


def _make_profile(
    gpu_count: int = 0,
    ram_mb: float = 16384.0,
    cpu_cores: int = 8,
) -> HardwareProfile:
    gpus = []
    for i in range(gpu_count):
        gpus.append(
            CudaGPU(
                index=i,
                name=f"Test GPU {i}",
                total_memory_mb=8192,
                free_memory_mb=6144,
                compute_capability="8.6",
                multiprocessor_count=42,
                temperature_c=55.0,
            )
        )
    return HardwareProfile(
        hostname="test-host",
        platform="Linux-Test",
        cpu=CPUInfo(
            model="Test CPU",
            cores_physical=cpu_cores,
            cores_logical=cpu_cores * 2,
        ),
        memory=MemoryInfo(
            total_ram_mb=ram_mb,
            available_ram_mb=ram_mb * 0.7,
            total_swap_mb=8192,
            available_swap_mb=4096,
        ),
        cuda_gpus=gpus,
        thermal_zones=[ThermalZone("tz0", "cpu", 45.0)],
    )


def _make_stress(gpu_gflops: float = 0.0, cpu_gflops: float = 50.0) -> StressReport:
    from ethos.stress_test import MatrixBenchmark, TokenThroughput

    benches = []
    # CPU bench
    cpu_b = DeviceBenchmark(
        device_name="Test CPU",
        device_type="cpu",
        matrix_benchmarks=[MatrixBenchmark(size=1024, avg_ms=100.0, gflops=cpu_gflops)],
        token_throughput=TokenThroughput(tokens_per_second=10.0, avg_latency_ms=100.0),
    )
    benches.append(cpu_b)

    if gpu_gflops > 0:
        gpu_b = DeviceBenchmark(
            device_name="Test GPU",
            device_type="cuda",
            device_index=0,
            matrix_benchmarks=[
                MatrixBenchmark(size=1024, avg_ms=5.0, gflops=gpu_gflops)
            ],
            token_throughput=TokenThroughput(
                tokens_per_second=200.0, avg_latency_ms=5.0
            ),
        )
        benches.append(gpu_b)

    return StressReport(benchmarks=benches, max_parallel_agents=4, total_duration_s=1.0)


# ── Hardware Survey ──────────────────────────────────────────────────


class TestHardwareProfile:
    def test_repr(self):
        p = _make_profile()
        r = repr(p)
        assert "test-host" in r
        assert "0 GPU" in r

    def test_repr_with_gpu(self):
        p = _make_profile(gpu_count=1)
        r = repr(p)
        assert "1 GPU" in r

    def test_cuda_gpu_repr(self):
        gpu = CudaGPU(0, "RTX 4090", 24576, 20000, "8.9", 128, 60.0)
        assert "RTX 4090" in repr(gpu)
        assert "24576" in repr(gpu)

    def test_cpu_info_repr(self):
        cpu = CPUInfo("AMD 5950X", 16, 32)
        r = repr(cpu)
        assert "AMD 5950X" in r
        assert "16P/32L" in r

    def test_memory_info_repr(self):
        mem = MemoryInfo(32768, 20000, 8192, 4096)
        r = repr(mem)
        assert "32768" in r

    def test_survey_hardware(self):
        """survey_hardware runs without error on any host."""
        profile = survey_hardware()
        assert isinstance(profile, HardwareProfile)
        assert profile.hostname
        assert isinstance(profile.cpu, CPUInfo)
        assert profile.cpu.cores_logical >= 1


# ── Stress Test ──────────────────────────────────────────────────────


class TestStressTest:
    def test_stress_report_repr(self):
        r = _make_stress()
        assert "StressReport" in repr(r)

    def test_device_benchmark_repr(self):
        b = DeviceBenchmark("CPU", "cpu")
        assert "CPU" in repr(b)

    def test_run_stress_quick(self):
        """Quick stress test should complete on any machine."""
        report = run_stress_test(quick=True)
        assert isinstance(report, StressReport)
        assert len(report.benchmarks) >= 1  # at least CPU
        assert report.total_duration_s > 0

    def test_best_gpu_gflops_none(self):
        report = _make_stress(gpu_gflops=0)
        assert report.best_gpu_gflops() is None

    def test_best_gpu_gflops(self):
        report = _make_stress(gpu_gflops=500.0)
        assert report.best_gpu_gflops() == 500.0

    def test_cpu_gflops(self):
        report = _make_stress(cpu_gflops=42.0)
        assert report.cpu_gflops() == 42.0


# ── Agent Allocator ──────────────────────────────────────────────────


class TestAgentAllocator:
    def test_allocation_plan_repr(self):
        plan = AllocationPlan()
        assert "AllocationPlan" in repr(plan)

    def test_build_plan_cpu_only(self):
        profile = _make_profile(gpu_count=0)
        stress = _make_stress(gpu_gflops=0, cpu_gflops=50.0)
        plan = build_allocation_plan(profile, stress)
        assert isinstance(plan, AllocationPlan)
        assert plan.total_agents >= 1
        assert len(plan.allocations) >= 1

    def test_build_plan_with_gpu(self):
        profile = _make_profile(gpu_count=1)
        stress = _make_stress(gpu_gflops=300.0, cpu_gflops=50.0)
        plan = build_allocation_plan(profile, stress)
        assert plan.total_agents >= 1
        gpu_allocs = plan.for_device("cuda:0")
        assert len(gpu_allocs) >= 1

    def test_build_plan_specific_types(self):
        profile = _make_profile()
        stress = _make_stress()
        plan = build_allocation_plan(profile, stress, agent_types=["routing", "memory"])
        types = {a.agent_type for a in plan.allocations}
        assert types == {"routing", "memory"}

    def test_build_plan_unknown_type(self):
        profile = _make_profile()
        stress = _make_stress()
        plan = build_allocation_plan(profile, stress, agent_types=["nonexistent"])
        assert any("Unknown" in n for n in plan.notes)

    def test_summary(self):
        profile = _make_profile()
        stress = _make_stress()
        plan = build_allocation_plan(profile, stress)
        s = plan.summary()
        assert "Allocation Plan" in s
        assert "agents" in s

    def test_agent_allocation_repr(self):
        from ethos.agent_allocator import AgentAllocation

        a = AgentAllocation("inference", 2, "cuda:0", 0.5, 4096)
        r = repr(a)
        assert "inference" in r
        assert "cuda:0" in r


# ── Trinity Connection ───────────────────────────────────────────────


class TestTrinityConnection:
    def test_score_good(self):
        profile = _make_profile(ram_mb=32768)
        stress = _make_stress(cpu_gflops=100.0)
        score = score_ethos_connection(
            profile,
            stress,
            agent_compute_utilization=0.5,
            agent_latency_ms=50.0,
            agent_memory_mb=2048.0,
            agent_thermal_impact=0.2,
        )
        assert isinstance(score, EthosConnectionScore)
        assert 0.0 <= score.total <= 1.0
        assert score.total > 0.5

    def test_score_poor_thermal(self):
        profile = _make_profile()
        profile.cuda_gpus = [
            CudaGPU(0, "Hot GPU", 8192, 4096, "8.6", 42, temperature_c=82.0)
        ]
        stress = _make_stress(gpu_gflops=300.0)
        score = score_ethos_connection(profile, stress, agent_thermal_impact=0.9)
        assert score.thermal_fit < 0.5

    def test_score_low_memory(self):
        profile = _make_profile(ram_mb=2048)
        stress = _make_stress()
        score = score_ethos_connection(profile, stress, agent_memory_mb=1800.0)
        assert score.memory_fit < 0.5

    def test_score_repr(self):
        score = EthosConnectionScore(0.8, 0.7, 0.9, 0.8, 0.75)
        r = repr(score)
        assert "0.8" in r
        assert "eff=" in r

    def test_score_clamped(self):
        """Total should always be 0-1."""
        profile = _make_profile()
        stress = _make_stress()
        score = score_ethos_connection(profile, stress)
        assert 0.0 <= score.total <= 1.0
        for dim in [
            score.hardware_efficiency,
            score.latency_fit,
            score.thermal_fit,
            score.memory_fit,
        ]:
            assert 0.0 <= dim <= 1.0


# ── Exports ──────────────────────────────────────────────────────────


class TestExports:
    def test_all_exports(self):
        import ethos

        for name in ethos.__all__:
            assert hasattr(ethos, name), f"Missing export: {name}"

    def test_version(self):
        import ethos

        assert ethos.__version__

    def test_py_typed_exists(self):
        import ethos
        import pathlib

        pkg = pathlib.Path(ethos.__file__).parent / "py.typed"
        assert pkg.exists()
