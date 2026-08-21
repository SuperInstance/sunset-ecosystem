"""Survey all available compute resources on the current host."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = ["HardwareProfile", "survey_hardware"]


@dataclass
class CudaGPU:
    """Single CUDA-capable GPU."""

    index: int
    name: str
    total_memory_mb: float
    free_memory_mb: float
    compute_capability: str
    multiprocessor_count: int
    temperature_c: Optional[float] = None
    utilization_pct: Optional[float] = None
    power_draw_w: Optional[float] = None
    power_limit_w: Optional[float] = None

    def __repr__(self) -> str:
        return (
            f"CudaGPU({self.index}, {self.name!r}, "
            f"{self.total_memory_mb:.0f}MB, cc={self.compute_capability})"
        )


@dataclass
class CPUInfo:
    """CPU details."""

    model: str
    cores_physical: int
    cores_logical: int
    frequency_mhz: Optional[float] = None
    l1_cache_kb: Optional[int] = None
    l2_cache_kb: Optional[int] = None
    l3_cache_kb: Optional[int] = None

    def __repr__(self) -> str:
        return f"CPUInfo({self.model!r}, {self.cores_physical}P/{self.cores_logical}L)"


@dataclass
class MemoryInfo:
    """System memory."""

    total_ram_mb: float
    available_ram_mb: float
    total_swap_mb: float
    available_swap_mb: float

    def __repr__(self) -> str:
        return (
            f"MemoryInfo(RAM={self.total_ram_mb:.0f}/{self.available_ram_mb:.0f}MB, "
            f"Swap={self.total_swap_mb:.0f}/{self.available_swap_mb:.0f}MB)"
        )


@dataclass
class ThermalZone:
    """A thermal zone reading."""

    name: str
    type: str
    temperature_c: Optional[float] = None

    def __repr__(self) -> str:
        return f"ThermalZone({self.name!r}, {self.temperature_c}°C)"


@dataclass
class HardwareProfile:
    """Complete hardware profile for the host."""

    hostname: str
    platform: str
    cpu: CPUInfo
    memory: MemoryInfo
    cuda_gpus: List[CudaGPU] = field(default_factory=list)
    igpu_available: bool = False
    igpu_name: Optional[str] = None
    npu_available: bool = False
    npu_name: Optional[str] = None
    thermal_zones: List[ThermalZone] = field(default_factory=list)
    python_version: str = ""
    torch_available: bool = False
    torch_version: Optional[str] = None
    numpy_available: bool = False
    numpy_version: Optional[str] = None
    directml_available: bool = False
    extras: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        gpu_count = len(self.cuda_gpus)
        return (
            f"HardwareProfile({self.hostname!r}, {self.cpu}, {self.memory}, "
            f"{gpu_count} GPU(s), igpu={self.igpu_available}, npu={self.npu_available})"
        )


def _try_import_torch() -> tuple[bool, Optional[str], List[CudaGPU]]:
    """Try to import torch and enumerate CUDA GPUs."""
    try:
        import torch  # type: ignore[import-untyped]

        version = torch.__version__
        gpus: List[CudaGPU] = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                free_mem, _total = torch.cuda.mem_get_info(i)
                gpus.append(
                    CudaGPU(
                        index=i,
                        name=props.name,
                        total_memory_mb=props.total_mem / (1024 * 1024),
                        free_memory_mb=free_mem / (1024 * 1024),
                        compute_capability=f"{props.major}.{props.minor}",
                        multiprocessor_count=props.multi_processor_count,
                    )
                )
        return True, version, gpus
    except Exception:
        return False, None, []


def _try_import_numpy() -> tuple[bool, Optional[str]]:
    try:
        import numpy  # type: ignore[import-untyped]

        return True, numpy.__version__
    except Exception:
        return False, None


def _detect_cuda_via_smi() -> List[CudaGPU]:
    """Fallback: detect GPUs via nvidia-smi."""
    gpus: List[CudaGPU] = []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free,"
                "compute_cap,multiprocessor_count,temperature.gpu,"
                "utilization.gpu,power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 5:
                    gpu = CudaGPU(
                        index=int(parts[0]),
                        name=parts[1],
                        total_memory_mb=float(parts[2]),
                        free_memory_mb=float(parts[3]),
                        compute_capability=parts[4],
                        multiprocessor_count=int(parts[5]) if len(parts) > 5 else 0,
                        temperature_c=float(parts[6])
                        if len(parts) > 6 and parts[6]
                        else None,
                        utilization_pct=float(parts[7])
                        if len(parts) > 7 and parts[7]
                        else None,
                        power_draw_w=float(parts[8])
                        if len(parts) > 8 and parts[8]
                        else None,
                        power_limit_w=float(parts[9])
                        if len(parts) > 9 and parts[9]
                        else None,
                    )
                    gpus.append(gpu)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return gpus


def _detect_cpu() -> CPUInfo:
    """Gather CPU info from the system."""
    physical = os.cpu_count() or 1
    logical = physical
    frequency = None
    model = platform.processor() or "Unknown CPU"
    l1 = l2 = l3 = None

    # Try lscpu on Linux
    try:
        result = subprocess.run(
            ["lscpu", "-J"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            for entry in data.get("lscpu", []):
                field_val = entry.get("field", "")
                val = entry.get("data", "")
                if "Model name" in field_val:
                    model = val
                elif "CPU(s):" == field_val or field_val == "CPU(s):":
                    logical = int(val)
                elif "Core(s) per socket" in field_val:
                    physical = logical // int(val) if int(val) > 0 else logical
                elif "Thread(s) per core" in field_val:
                    pass  # already have logical
                elif "CPU MHz" in field_val:
                    try:
                        frequency = float(val)
                    except ValueError:
                        pass
                elif "L1d cache" in field_val:
                    l1 = _parse_cache_size(val)
                elif "L2 cache" in field_val:
                    l2 = _parse_cache_size(val)
                elif "L3 cache" in field_val:
                    l3 = _parse_cache_size(val)
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        pass

    return CPUInfo(
        model=model,
        cores_physical=physical,
        cores_logical=logical,
        frequency_mhz=frequency,
        l1_cache_kb=l1,
        l2_cache_kb=l2,
        l3_cache_kb=l3,
    )


def _parse_cache_size(val: str) -> Optional[int]:
    """Parse cache size string like '32768K' or '16M' to KB."""
    val = val.strip().upper()
    try:
        if val.endswith("K"):
            return int(float(val[:-1]))
        elif val.endswith("M"):
            return int(float(val[:-1]) * 1024)
        elif val.endswith("G"):
            return int(float(val[:-1]) * 1024 * 1024)
        else:
            return int(float(val))
    except (ValueError, TypeError):
        return None


def _detect_memory() -> MemoryInfo:
    """Get RAM and swap info."""
    try:
        import resource as _res

        # WSL / Linux: read from /proc/meminfo
        total_ram = available = total_swap = avail_swap = 0.0
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        key, val = parts[0], parts[1]
                        if key == "MemTotal:":
                            total_ram = float(val)
                        elif key == "MemAvailable:":
                            available = float(val)
                        elif key == "SwapTotal:":
                            total_swap = float(val)
                        elif key == "SwapFree:":
                            avail_swap = float(val)
            # Values are in kB
            return MemoryInfo(
                total_ram_mb=total_ram / 1024,
                available_ram_mb=available / 1024,
                total_swap_mb=total_swap / 1024,
                available_swap_mb=avail_swap / 1024,
            )
        except FileNotFoundError:
            pass
    except Exception:
        pass

    # Fallback
    return MemoryInfo(
        total_ram_mb=0,
        available_ram_mb=0,
        total_swap_mb=0,
        available_swap_mb=0,
    )


def _detect_directml() -> tuple[bool, Optional[str]]:
    """Check if DirectML is available (iGPU support)."""
    try:
        import torch_directml  # type: ignore[import-untyped]

        return True, torch_directml.device_name(0)
    except Exception:
        pass
    return False, None


def _detect_npu() -> tuple[bool, Optional[str]]:
    """Check for NPU (Ry AI SDK or similar)."""
    # Check for Ryzen AI / VAEP / Qualcomm NPU
    try:
        result = subprocess.run(
            ["ls", "/dev/accel/accel0"],
            capture_output=True,
            timeout=2,
        )
        if result.returncode == 0:
            return True, "NPU via /dev/accel"
    except Exception:
        pass

    try:
        import ryzenai  # type: ignore[import-untyped]

        return True, "Ryzen AI"
    except Exception:
        pass

    return False, None


def _read_thermal_zones() -> List[ThermalZone]:
    """Read Linux thermal zones."""
    zones: List[ThermalZone] = []
    thermal_path = "/sys/class/thermal"
    try:
        entries = sorted(os.listdir(thermal_path))
        for entry in entries:
            if entry.startswith("thermal_zone"):
                zone_path = os.path.join(thermal_path, entry)
                try:
                    ztype = ""
                    temp = None
                    with open(os.path.join(zone_path, "type")) as f:
                        ztype = f.read().strip()
                    with open(os.path.join(zone_path, "temp")) as f:
                        raw = int(f.read().strip())
                        temp = raw / 1000.0
                    zones.append(
                        ThermalZone(name=entry, type=ztype, temperature_c=temp)
                    )
                except (FileNotFoundError, PermissionError, ValueError):
                    zones.append(ThermalZone(name=entry, type="unknown"))
    except (FileNotFoundError, PermissionError):
        pass
    return zones


def survey_hardware() -> HardwareProfile:
    """Survey all available compute on the current host.

    Probes CUDA GPUs (via torch or nvidia-smi fallback), CPU, RAM, swap,
    iGPU (DirectML), NPU, and thermal state. Returns a complete
    :class:`HardwareProfile`.

    Returns:
        HardwareProfile with all discovered hardware specs.
    """
    torch_ok, torch_ver, torch_gpus = _try_import_torch()
    np_ok, np_ver = _try_import_numpy()

    # If torch didn't find GPUs, try nvidia-smi
    gpus = torch_gpus or _detect_cuda_via_smi()

    igpu_ok, igpu_name = _detect_directml()
    npu_ok, npu_name = _detect_npu()
    thermal = _read_thermal_zones()

    # Enrich GPU thermal/power from nvidia-smi if torch didn't provide it
    if gpus and all(g.temperature_c is None for g in gpus):
        smi_gpus = _detect_cuda_via_smi()
        for gpu in gpus:
            for sg in smi_gpus:
                if sg.index == gpu.index:
                    gpu.temperature_c = sg.temperature_c
                    gpu.utilization_pct = sg.utilization_pct
                    gpu.power_draw_w = sg.power_draw_w
                    gpu.power_limit_w = sg.power_limit_w
                    break

    return HardwareProfile(
        hostname=platform.node(),
        platform=platform.platform(),
        cpu=_detect_cpu(),
        memory=_detect_memory(),
        cuda_gpus=gpus,
        igpu_available=igpu_ok,
        igpu_name=igpu_name,
        npu_available=npu_ok,
        npu_name=npu_name,
        thermal_zones=thermal,
        python_version=sys.version,
        torch_available=torch_ok,
        torch_version=torch_ver,
        numpy_available=np_ok,
        numpy_version=np_ver,
        directml_available=igpu_ok,
    )
