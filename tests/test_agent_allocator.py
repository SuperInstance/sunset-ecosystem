"""Tests for ethos.agent_allocator — agent-to-hardware allocation planning."""

import pytest

from ethos.agent_allocator import (
    AgentAllocation,
    AllocationPlan,
    build_allocation_plan,
)
from ethos.hardware_survey import CPUInfo, CudaGPU, HardwareProfile, MemoryInfo, ThermalZone
from ethos.stress_test import DeviceBenchmark, MatrixBenchmark, StressReport


def _make_profile(
    cuda_gpus=None,
    cpu_cores=8,
    available_ram_mb=32000,
    thermal_zones=None,
) -> HardwareProfile:
    return HardwareProfile(
        hostname="test-host",
        platform="linux",
        cpu=CPUInfo(
            model="TestCPU",
            cores_physical=cpu_cores // 2,
            cores_logical=cpu_cores,
            frequency_mhz=3000.0,
        ),
        memory=MemoryInfo(
            total_ram_mb=64000,
            available_ram_mb=available_ram_mb,
            total_swap_mb=0,
            available_swap_mb=0,
        ),
        cuda_gpus=cuda_gpus or [],
        thermal_zones=thermal_zones or [],
    )


def _make_stress(gpu_gflops=None, cpu_gflops=100.0) -> StressReport:
    benchmarks = []
    if gpu_gflops is not None:
        benchmarks.append(
            DeviceBenchmark(
                device_name="cuda:0",
                device_type="cuda",
                device_index=0,
                matrix_benchmarks=[
                    MatrixBenchmark(size=1024, avg_ms=10.0, gflops=gpu_gflops)
                ],
            )
        )
    benchmarks.append(
        DeviceBenchmark(
            device_name="cpu",
            device_type="cpu",
            device_index=0,
            matrix_benchmarks=[
                MatrixBenchmark(size=512, avg_ms=100.0, gflops=cpu_gflops)
            ],
        )
    )
    return StressReport(benchmarks=benchmarks, max_parallel_agents=4, total_duration_s=1.0)


class TestAgentAllocation:
    def test_repr(self):
        alloc = AgentAllocation(
            agent_type="inference",
            count=2,
            device="cuda:0",
            compute_budget_pct=0.5,
            memory_budget_mb=4096,
        )
        assert "inference" in repr(alloc)
        assert "cuda:0" in repr(alloc)


class TestAllocationPlan:
    def test_repr(self):
        plan = AllocationPlan(total_agents=4, thermal_budget_pct=0.6)
        assert "4 agents" in repr(plan)

    def test_for_device(self):
        plan = AllocationPlan(
            allocations=[
                AgentAllocation("inference", 2, "cuda:0", 0.5, 4096),
                AgentAllocation("routing", 1, "cpu", 0.1, 512),
            ],
            total_agents=3,
        )
        assert len(plan.for_device("cuda:0")) == 1
        assert len(plan.for_device("cpu")) == 1

    def test_summary(self):
        plan = AllocationPlan(
            allocations=[
                AgentAllocation("inference", 2, "cuda:0", 0.5, 4096),
            ],
            total_agents=2,
            notes=["test note"],
        )
        summary = plan.summary()
        assert "2 total agents" in summary
        assert "test note" in summary


class TestBuildAllocationPlan:
    def test_cpu_only(self):
        profile = _make_profile(cuda_gpus=[], cpu_cores=8)
        stress = _make_stress(gpu_gflops=None, cpu_gflops=100.0)
        plan = build_allocation_plan(profile, stress)
        assert plan.total_agents > 0
        # All allocations should be on CPU
        assert all(a.device == "cpu" for a in plan.allocations)
        assert len(plan.notes) >= 1

    def test_with_gpu(self):
        gpus = [
            CudaGPU(
                index=0,
                name="TestGPU",
                total_memory_mb=16000,
                free_memory_mb=8000,
                compute_capability="8.6",
                multiprocessor_count=80,
                temperature_c=50.0,
                power_limit_w=250.0,
            )
        ]
        profile = _make_profile(cuda_gpus=gpus, cpu_cores=8)
        stress = _make_stress(gpu_gflops=1000.0, cpu_gflops=100.0)
        plan = build_allocation_plan(profile, stress)
        assert plan.total_agents > 0
        # GPU-preferred types should be on GPU
        gpu_allocs = [a for a in plan.allocations if a.device == "cuda:0"]
        assert len(gpu_allocs) > 0

    def test_thermal_headroom(self):
        zones = [ThermalZone(name="test", type="x86_pkg_temp", temperature_c=70.0)]
        profile = _make_profile(cuda_gpus=[], cpu_cores=8, thermal_zones=zones)
        stress = _make_stress(gpu_gflops=None, cpu_gflops=100.0)
        plan = build_allocation_plan(profile, stress)
        assert plan.thermal_budget_pct > 0
        assert any("Thermal headroom" in n for n in plan.notes)

    def test_unknown_agent_type(self):
        profile = _make_profile(cuda_gpus=[], cpu_cores=8)
        stress = _make_stress(gpu_gflops=None, cpu_gflops=100.0)
        plan = build_allocation_plan(profile, stress, agent_types=["inference", "unknown_type_xyz"])
        assert any("Unknown agent type" in n for n in plan.notes)

    def test_subset_agent_types(self):
        gpus = [
            CudaGPU(
                index=0,
                name="TestGPU",
                total_memory_mb=16000,
                free_memory_mb=8000,
                compute_capability="8.6",
                multiprocessor_count=80,
            )
        ]
        profile = _make_profile(cuda_gpus=gpus, cpu_cores=8)
        stress = _make_stress(gpu_gflops=1000.0, cpu_gflops=100.0)
        plan = build_allocation_plan(profile, stress, agent_types=["inference"])
        assert len(plan.allocations) == 1
        assert plan.allocations[0].agent_type == "inference"

    def test_allocation_bounds(self):
        gpus = [
            CudaGPU(
                index=0,
                name="TestGPU",
                total_memory_mb=16000,
                free_memory_mb=8000,
                compute_capability="8.6",
                multiprocessor_count=80,
            )
        ]
        profile = _make_profile(cuda_gpus=gpus, cpu_cores=8)
        stress = _make_stress(gpu_gflops=1000.0, cpu_gflops=100.0)
        plan = build_allocation_plan(profile, stress)
        for alloc in plan.allocations:
            assert alloc.count >= 1
            assert alloc.count <= 4  # capped
            assert 0 < alloc.compute_budget_pct <= 1.0
            assert alloc.memory_budget_mb > 0

    def test_empty_gpu_list(self):
        profile = _make_profile(cuda_gpus=[], cpu_cores=4)
        stress = _make_stress(gpu_gflops=None, cpu_gflops=50.0)
        plan = build_allocation_plan(profile, stress, agent_types=["inference"])
        # inference prefers GPU but falls back to CPU
        assert plan.allocations[0].device == "cpu"
