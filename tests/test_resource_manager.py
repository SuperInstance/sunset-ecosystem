import pytest
from fleet.resource_manager import ResourceAllocation, ResourceManager


class TestResourceAllocation:
    def test_to_dict(self):
        a = ResourceAllocation("t1", 2.0, 1024.0, 1, 0.0)
        d = a.to_dict()
        assert d["task_id"] == "t1"
        assert d["cpu_cores"] == 2.0


class TestResourceManager:
    def test_init(self):
        rm = ResourceManager()
        assert rm.fleet_node_id == "default"
        assert rm.total_cpu == 8.0

    def test_allocate(self):
        rm = ResourceManager()
        alloc = rm.allocate("task1", cpu_cores=2.0, memory_mb=1024.0)
        assert alloc is not None
        assert alloc.task_id == "task1"

    def test_allocate_exceed_cpu(self):
        rm = ResourceManager(total_cpu=1.0)
        alloc = rm.allocate("task1", cpu_cores=2.0)
        assert alloc is None

    def test_allocate_exceed_memory(self):
        rm = ResourceManager(total_memory_mb=512.0)
        alloc = rm.allocate("task1", memory_mb=1024.0)
        assert alloc is None

    def test_allocate_exceed_gpu(self):
        rm = ResourceManager(total_gpu=1)
        alloc = rm.allocate("task1", gpu_devices=2)
        assert alloc is None

    def test_release(self):
        rm = ResourceManager()
        rm.allocate("task1", cpu_cores=2.0)
        assert rm.release("task1") is True
        assert rm.release("task1") is False

    def test_get_usage(self):
        rm = ResourceManager()
        rm.allocate("task1", cpu_cores=2.0, memory_mb=1024.0)
        usage = rm.get_usage()
        assert usage["cpu_cores"] == 2.0
        assert usage["memory_mb"] == 1024.0

    def test_get_available(self):
        rm = ResourceManager(total_cpu=8.0)
        rm.allocate("task1", cpu_cores=2.0)
        avail = rm.get_available()
        assert avail["cpu_cores"] == 6.0

    def test_get_utilization(self):
        rm = ResourceManager(total_cpu=4.0)
        rm.allocate("task1", cpu_cores=2.0)
        util = rm.get_utilization()
        assert util["cpu"] == 0.5

    def test_get_allocations(self):
        rm = ResourceManager()
        rm.allocate("task1", cpu_cores=2.0)
        allocs = rm.get_allocations()
        assert len(allocs) == 1

    def test_to_dict(self):
        rm = ResourceManager()
        rm.allocate("task1", cpu_cores=2.0)
        d = rm.to_dict()
        assert d["active_allocations"] == 1
        assert d["usage"]["cpu_cores"] == 2.0
