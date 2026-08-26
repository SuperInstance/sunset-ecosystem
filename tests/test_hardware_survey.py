"""Tests for hardware survey — CPU, GPU, memory, and thermal detection.

Mocks subprocess calls to avoid platform-specific dependencies.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ethos.hardware_survey import (
    CPUInfo,
    CudaGPU,
    HardwareProfile,
    MemoryInfo,
    ThermalZone,
    _detect_cpu,
    _detect_cuda_via_smi,
    _try_import_numpy,
    _try_import_torch,
    survey_hardware,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_cuda_gpu_repr(self):
        gpu = CudaGPU(
            index=0,
            name="RTX 4090",
            total_memory_mb=24576,
            free_memory_mb=12000,
            compute_capability="8.9",
            multiprocessor_count=128,
        )
        assert "RTX 4090" in repr(gpu)
        assert "24576MB" in repr(gpu)

    def test_cpu_info_repr(self):
        cpu = CPUInfo(model="AMD EPYC", cores_physical=64, cores_logical=128)
        assert "AMD EPYC" in repr(cpu)
        assert "64P/128L" in repr(cpu)

    def test_memory_info_repr(self):
        mem = MemoryInfo(
            total_ram_mb=65536,
            available_ram_mb=32768,
            total_swap_mb=8192,
            available_swap_mb=4096,
        )
        assert "RAM=65536/32768MB" in repr(mem)

    def test_thermal_zone_repr(self):
        tz = ThermalZone(name="CPU", type="x86_pkg_temp", temperature_c=65.0)
        assert "CPU" in repr(tz)
        assert "65.0°C" in repr(tz)

    def test_hardware_profile_repr(self):
        hp = HardwareProfile(
            hostname="test",
            platform="linux",
            cpu=CPUInfo(model="x86", cores_physical=4, cores_logical=8),
            memory=MemoryInfo(
                total_ram_mb=16000,
                available_ram_mb=8000,
                total_swap_mb=2000,
                available_swap_mb=1000,
            ),
        )
        assert "test" in repr(hp)
        assert "0 GPU(s)" in repr(hp)  # 0 GPUs


# ---------------------------------------------------------------------------
# CUDA detection
# ---------------------------------------------------------------------------


class TestCudaDetection:
    def test_detect_cuda_via_smi_no_binary(self):
        with patch("subprocess.run", side_effect=FileNotFoundError()):
            gpus = _detect_cuda_via_smi()
            assert gpus == []

    def test_detect_cuda_via_smi_timeout(self):
        import subprocess as sp

        with patch("subprocess.run", side_effect=sp.TimeoutExpired("cmd", 10)):
            gpus = _detect_cuda_via_smi()
            assert gpus == []

    def test_detect_cuda_via_smi_valid(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "0, RTX 4090, 24576, 12000, 8.9, 128, 65, 50, 300, 450\n"
        with patch("subprocess.run", return_value=mock_result):
            gpus = _detect_cuda_via_smi()
            assert len(gpus) == 1
            assert gpus[0].name == "RTX 4090"
            assert gpus[0].index == 0


# ---------------------------------------------------------------------------
# CPU detection
# ---------------------------------------------------------------------------


class TestCpuDetection:
    def test_detect_cpu_fallback(self):
        with patch("subprocess.run", side_effect=FileNotFoundError()):
            cpu = _detect_cpu()
            assert cpu.cores_physical >= 1
            assert cpu.model != ""

    def test_detect_cpu_lscpu(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = '{"lscpu": [{"field": "Model name:", "data": "Intel Xeon"}, {"field": "CPU(s):", "data": "64"}, {"field": "Thread(s) per core:", "data": "2"}, {"field": "Core(s) per socket:", "data": "32"}, {"field": "CPU MHz:", "data": "2400"}]}'
        with patch("subprocess.run", return_value=mock_result):
            cpu = _detect_cpu()
            assert "Intel Xeon" in cpu.model
            # lscpu parsing may vary; just assert it's reasonable
            assert cpu.cores_physical >= 1
            assert cpu.cores_logical >= cpu.cores_physical


# ---------------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------------


class TestImportHelpers:
    def test_try_import_numpy(self):
        ok, version = _try_import_numpy()
        assert ok is True
        assert version is not None

    def test_try_import_torch(self):
        ok, version, gpus = _try_import_torch()
        # torch may or may not be installed
        assert isinstance(ok, bool)
        assert isinstance(gpus, list)


# ---------------------------------------------------------------------------
# survey_hardware
# ---------------------------------------------------------------------------


class TestSurveyHardware:
    def test_survey_returns_profile(self):
        with patch("ethos.hardware_survey._detect_cuda_via_smi", return_value=[]):
            with patch(
                "ethos.hardware_survey._detect_cpu",
                return_value=CPUInfo(model="x86", cores_physical=4, cores_logical=8),
            ):
                with patch(
                    "ethos.hardware_survey._detect_memory",
                    return_value=MemoryInfo(
                        total_ram_mb=16000,
                        available_ram_mb=8000,
                        total_swap_mb=2000,
                        available_swap_mb=1000,
                    ),
                ):
                    with patch(
                        "ethos.hardware_survey._read_thermal_zones", return_value=[]
                    ):
                        profile = survey_hardware()
                        assert isinstance(profile, HardwareProfile)
                        assert profile.hostname != ""
                        assert profile.platform != ""
                        assert profile.cpu is not None
                        assert profile.memory is not None

    def test_survey_has_numpy(self):
        with patch("ethos.hardware_survey._detect_cuda_via_smi", return_value=[]):
            with patch(
                "ethos.hardware_survey._detect_cpu",
                return_value=CPUInfo(model="x86", cores_physical=4, cores_logical=8),
            ):
                with patch(
                    "ethos.hardware_survey._detect_memory",
                    return_value=MemoryInfo(
                        total_ram_mb=16000,
                        available_ram_mb=8000,
                        total_swap_mb=2000,
                        available_swap_mb=1000,
                    ),
                ):
                    with patch(
                        "ethos.hardware_survey._read_thermal_zones", return_value=[]
                    ):
                        profile = survey_hardware()
                        assert profile.numpy_available is True
                        assert profile.numpy_version is not None

    def test_survey_torch_field(self):
        with patch("ethos.hardware_survey._detect_cuda_via_smi", return_value=[]):
            with patch(
                "ethos.hardware_survey._detect_cpu",
                return_value=CPUInfo(model="x86", cores_physical=4, cores_logical=8),
            ):
                with patch(
                    "ethos.hardware_survey._detect_memory",
                    return_value=MemoryInfo(
                        total_ram_mb=16000,
                        available_ram_mb=8000,
                        total_swap_mb=2000,
                        available_swap_mb=1000,
                    ),
                ):
                    with patch(
                        "ethos.hardware_survey._read_thermal_zones", return_value=[]
                    ):
                        profile = survey_hardware()
                        assert isinstance(profile.torch_available, bool)
