"""CPU RNG and mocked accelerator tests; zero models and optimizer updates."""
from contextlib import contextmanager
import os
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from hydra_directflow.devices import (CudaQualificationLimits, DeviceSelection, RuntimePolicy, available_device,
    capture_rng, configure_runtime, memory_snapshot, plan_cuda_qualification, require_compatible_runtime,
    restore_rng, runtime_profile, seed_rng, synchronize, validate_funding_record)


class FakeCuda:
    def __init__(self):
        self.available, self.count, self.current = True, 2, 0
        self.calls = []
        self.states = {0: torch.tensor([1, 2, 3], dtype=torch.uint8), 1: torch.tensor([4, 5, 6], dtype=torch.uint8)}
    def is_available(self):
        return self.available
    def device_count(self):
        return self.count
    @contextmanager
    def device(self, index):
        previous, self.current = self.current, index
        self.calls.append(("context_enter", index))
        try:
            yield
        finally:
            self.current = previous
            self.calls.append(("context_exit", previous))
    def get_rng_state(self, index):
        self.calls.append(("get_rng", index))
        return self.states[index].clone()
    def manual_seed(self, seed):
        self.calls.append(("seed", self.current, seed))
        self.states[self.current] = torch.tensor([seed % 255, 2, 3], dtype=torch.uint8)
    def set_rng_state(self, state, device):
        self.calls.append(("set_rng", device))
        self.states[device] = state.clone()
    def synchronize(self, index):
        self.calls.append(("synchronize", index))
    def get_device_properties(self, index):
        self.calls.append(("properties", index))
        return SimpleNamespace(name="MOCK_CUDA_GPU", total_memory=8 * 1024**3)
    def set_device(self, index):
        self.calls.append(("set_device", index))
        self.current = index
    def set_per_process_memory_fraction(self, fraction, device):
        self.calls.append(("fraction", fraction, device))
    def get_device_capability(self, index):
        return (8, 9)
    def memory_allocated(self, index):
        self.calls.append(("memory_allocated", index))
        return 100 + index
    def memory_reserved(self, index):
        return 200 + index
    def max_memory_allocated(self, index):
        return 300 + index
    def max_memory_reserved(self, index):
        return 400 + index


class FakeMps:
    def __init__(self):
        self.calls = []
        self.state = torch.tensor([7, 8, 9], dtype=torch.uint8)
    def synchronize(self):
        self.calls.append("synchronize")
    def manual_seed(self, seed):
        self.calls.append(("seed", seed))
        self.state = torch.tensor([seed % 255, 8, 9], dtype=torch.uint8)
    def get_rng_state(self):
        return self.state.clone()
    def set_rng_state(self, state):
        self.state = state.clone()
    def recommended_max_memory(self):
        return 8 * 1024**3
    def current_allocated_memory(self):
        return 128
    def driver_allocated_memory(self):
        return 256
    def set_per_process_memory_fraction(self, value):
        self.calls.append(("fraction", value))


class FakeTorch:
    def __init__(self):
        self.cuda, self.mps = FakeCuda(), FakeMps()
        self.mps_available, self.deterministic, self.precision = True, False, "highest"
        self.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: self.mps_available),
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
            cudnn=SimpleNamespace(allow_tf32=True, benchmark=True, deterministic=False, version=lambda: 90100))
        self.version = SimpleNamespace(cuda="MOCK_CUDA_TOOLKIT")
    def __getattr__(self, name):
        return getattr(torch, name)
    def use_deterministic_algorithms(self, enabled, warn_only=False):
        self.deterministic = enabled
    def are_deterministic_algorithms_enabled(self):
        return self.deterministic
    def is_deterministic_algorithms_warn_only_enabled(self):
        return False
    def set_float32_matmul_precision(self, precision):
        self.precision = precision
    def get_float32_matmul_precision(self):
        return self.precision


def funding():
    return dict(schema="hydra-cuda-pilot-funding/v1", status="APPROVED_WITHIN_COMBINED_ENVELOPE",
                approval_sha256="a" * 64, budget_ledger_sha256="b" * 64, price_quote_sha256="c" * 64,
                combined_cap_usd="150", other_committed_or_reserved_usd="140", pilot_cap_usd="10", pilot_upper_estimate_usd="5")


class DeviceTests(unittest.TestCase):
    def setUp(self):
        self.cpu_rng = torch.get_rng_state().clone()
        self.fake = FakeTorch()
    def tearDown(self):
        torch.set_rng_state(self.cpu_rng)

    def test_device_names_require_explicit_backend_and_cuda_index(self):
        self.assertEqual(DeviceSelection.parse("cuda:1").index, 1)
        self.assertEqual(DeviceSelection.parse("cpu").name, "cpu")
        for value in ("cuda", "cuda:01", "cuda:-1", "cuda:128", "cpu:0", "mps:0", "auto", True):
            with self.assertRaises(ValueError):
                DeviceSelection.parse(value)

    def test_unavailable_or_invalid_index_never_falls_back(self):
        self.fake.cuda.available = False
        with self.assertRaisesRegex(ValueError, "fallback"):
            available_device("cuda:0", torch_module=self.fake)
        self.fake.cuda.available = True
        with self.assertRaisesRegex(ValueError, "index"):
            available_device("cuda:2", torch_module=self.fake)
        self.fake.mps_available = False
        with self.assertRaisesRegex(ValueError, "fallback"):
            available_device("mps", torch_module=self.fake)
        self.assertEqual(self.fake.cuda.calls, [])

    def test_cpu_rng_exact_replay_without_accelerator_calls(self):
        seed_rng("cpu", 17, torch_module=self.fake)
        state = capture_rng("cpu", torch_module=self.fake)
        expected = torch.rand(12)
        torch.rand(7)
        restore_rng(state, "cpu", torch_module=self.fake)
        self.assertTrue(torch.equal(torch.rand(12), expected))
        self.assertEqual(self.fake.cuda.calls, [])
        self.assertEqual(self.fake.mps.calls, [])

    def test_cuda_seed_initializes_inside_selected_context_not_all_devices(self):
        original = self.fake.cuda.states[0].clone()
        seed_rng("cuda:1", 42, torch_module=self.fake)
        self.assertEqual(self.fake.cuda.calls[:3], [("context_enter", 1), ("get_rng", 1), ("seed", 1, 42)])
        self.assertEqual(self.fake.cuda.current, 0)
        self.assertTrue(torch.equal(self.fake.cuda.states[0], original))
        state = capture_rng("cuda:1", torch_module=self.fake)
        self.fake.cuda.states[1].zero_()
        restore_rng(state, "cuda:1", torch_module=self.fake)
        self.assertTrue(torch.equal(self.fake.cuda.states[1], state["accelerator"]))
        self.assertIn(("synchronize", 1), self.fake.cuda.calls)
        self.assertIn(("set_rng", 1), self.fake.cuda.calls)
        self.assertNotIn(("set_rng", 0), self.fake.cuda.calls)

    def test_rng_backend_dtype_and_length_mismatches_precede_cpu_mutation(self):
        state = capture_rng("cuda:1", torch_module=self.fake)
        for changed in ({**state, "device": "cuda:0"},
                        {**state, "accelerator": torch.zeros(3)},
                        {**state, "accelerator": torch.zeros(1, dtype=torch.uint8)},
                        {**state, "cpu": torch.zeros(1, dtype=torch.uint8)}):
            seed_rng("cpu", 33, torch_module=self.fake)
            before = torch.get_rng_state().clone()
            with self.assertRaises(ValueError):
                restore_rng(changed, "cuda:1", torch_module=self.fake)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_mps_rng_and_allocator_are_isolated_from_cuda(self):
        seed_rng("mps", 9, torch_module=self.fake)
        state = capture_rng("mps", torch_module=self.fake)
        self.fake.mps.state.zero_()
        restore_rng(state, "mps", torch_module=self.fake)
        self.assertTrue(torch.equal(state["accelerator"], self.fake.mps.state))
        result = configure_runtime(RuntimePolicy("mps", allocator_limit_bytes=2 * 1024**3), torch_module=self.fake)
        self.assertEqual(result["allocator_fraction"], .25)
        self.assertIn(("fraction", .25), self.fake.mps.calls)
        self.assertEqual(self.fake.cuda.calls, [])

    def test_cuda_configuration_requires_cap_and_launch_determinism_environment(self):
        with self.assertRaisesRegex(ValueError, "allocator cap"):
            configure_runtime(RuntimePolicy("cuda:0"), torch_module=self.fake)
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ValueError, "CUBLAS"):
            configure_runtime(RuntimePolicy("cuda:0", allocator_limit_bytes=1024, deterministic_algorithms=True), torch_module=self.fake)
        self.assertFalse(any(call[0] == "set_device" for call in self.fake.cuda.calls))

    def test_cuda_policy_uses_selected_index_fp32_and_fraction(self):
        with patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}):
            result = configure_runtime(RuntimePolicy("cuda:1", allocator_limit_bytes=4 * 1024**3, deterministic_algorithms=True), torch_module=self.fake)
        self.assertEqual(result["allocator_fraction"], .5)
        self.assertIn(("fraction", .5, 1), self.fake.cuda.calls)
        self.assertFalse(self.fake.backends.cuda.matmul.allow_tf32)
        self.assertFalse(self.fake.backends.cudnn.allow_tf32)
        self.assertFalse(self.fake.backends.cudnn.benchmark)
        self.assertTrue(result["runtime"]["deterministic_algorithms"])
        self.assertEqual(result["runtime"]["float32_matmul_precision"], "highest")
        with self.assertRaisesRegex(ValueError, "exceeds"):
            configure_runtime(RuntimePolicy("cuda:0", allocator_limit_bytes=16 * 1024**3), torch_module=self.fake)

    def test_memory_scope_does_not_invent_cpu_or_total_vram_measurement(self):
        cpu = memory_snapshot("cpu", torch_module=self.fake)
        self.assertIsNone(cpu["allocator_allocated_bytes"])
        cuda = memory_snapshot("cuda:1", torch_module=self.fake)
        self.assertEqual(cuda["allocator_allocated_bytes"], 101)
        self.assertEqual(cuda["allocator_peak_reserved_bytes"], 401)
        self.assertIn("excludes", cuda["scope"])
        mps = memory_snapshot("mps", torch_module=self.fake)
        self.assertEqual(mps["driver_allocated_bytes"], 256)

    def test_runtime_change_or_unknown_cuda_driver_rejects_exact_restore(self):
        profile = runtime_profile("cuda:0", torch_module=self.fake)
        self.assertNotIn("hostname", profile)
        self.assertNotIn("uuid", profile)
        with self.assertRaisesRegex(ValueError, "driver"):
            require_compatible_runtime(profile, profile)
        known = runtime_profile("cuda:0", torch_module=self.fake, cuda_driver_version="MOCK_DRIVER")
        require_compatible_runtime(known, dict(known))
        for key, value in (("cuda_driver_version", "changed"), ("device", "cuda:1"), ("cuda_matmul_allow_tf32", not known["cuda_matmul_allow_tf32"])):
            with self.assertRaises(ValueError):
                require_compatible_runtime(known, {**known, key: value})

    def test_default_cuda_plan_is_blocked_and_never_imports_torch(self):
        result = plan_cuda_qualification()
        self.assertEqual(result["status"], "BLOCKED_PLAN_ONLY")
        self.assertEqual(len(result["blockers"]), 3)
        self.assertFalse(result["execution_enabled"])
        self.assertEqual(sum(case["optimizer_updates"] for case in result["cases"]), 24)
        code = "import sys; from hydra_directflow.devices import plan_cuda_qualification; assert 'torch' not in sys.modules; p=plan_cuda_qualification(); assert 'torch' not in sys.modules and not p['execution_enabled']"
        child = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=5)
        self.assertEqual(child.returncode, 0, child.stderr)

    def test_valid_gates_still_cannot_execute_or_enable_current_trainer(self):
        result = plan_cuda_qualification(funding=funding(), data_protocol_ready=True, local_escalation_required=True,
                                        protocol_sha256="d" * 64, cohort_sha256="e" * 64, local_feasibility_sha256="f" * 64)
        self.assertEqual(result["status"], "GATES_RECORDED_PLAN_ONLY")
        self.assertFalse(result["execution_enabled"])
        self.assertFalse(result["trainer_cuda_enabled"])
        self.assertEqual((result["provider_operations"], result["optimizer_updates_executed"]), (0, 0))

    def test_funding_and_limit_gates_cannot_overrun_existing_allocations(self):
        validate_funding_record(funding())
        for update in ({"pilot_cap_usd": "11"}, {"combined_cap_usd": "151"}, {"other_committed_or_reserved_usd": "141"},
                       {"pilot_upper_estimate_usd": "10.01"}, {"approval_sha256": None}, {"pilot_cap_usd": True},
                       {"pilot_upper_estimate_usd": "NaN"}):
            with self.assertRaises(ValueError):
                validate_funding_record({**funding(), **update})
        for limits in (dict(max_optimizer_updates=25), dict(max_wall_seconds=901), dict(max_host_rss_bytes=5 * 1024**3)):
            with self.assertRaises(ValueError):
                CudaQualificationLimits(**limits)


if __name__ == "__main__":
    unittest.main()
