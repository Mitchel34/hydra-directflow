"""Synthetic mechanics only: no research data, downloads, or fitted artifacts.

Standard unittest discovery makes no optimizer updates. The separately bounded
qualification runner owns checkpoint/update tests and their explicit ledger.
"""
import copy
import io
import unittest

import torch

from hydra_directflow.models.directflow import (
    ARCHITECTURES, DirectFlowModel, ModelConfig, masked_mse,
)
from hydra_directflow.models.mamba2_math import mixer_math, sequential, ssd


def configure_threads():
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)


def config(architecture, history_hours=24, dropout=0, weather_features=None):
    # These feature names and lookbacks are synthetic geometry, not a data policy.
    history = ("discharge_cms", "age_hours") if weather_features else ("flow", "flow_age_hours")
    state = ("soil_water_fraction", "soil_age_hours") if weather_features else ("soil_state", "soil_age_hours")
    return ModelConfig(architecture, history,
        weather_features or ("temperature", "precipitation"), state,
        history_hours=history_hours, horizon_hours=tuple(range(1, 19)), dropout=dropout)


def batch(history_hours=24, device="cpu", weather_count=2, batch_size=1):
    generator = torch.Generator().manual_seed(104)
    values = [torch.randn(shape, generator=generator).to(device) for shape in
              ((batch_size, history_hours, 2), (batch_size, 18, weather_count), (batch_size, 2))]
    result = []
    for value in values:
        mask = torch.ones_like(value, dtype=torch.bool)
        mask.reshape(-1)[1] = False
        value.reshape(-1)[1] = 0
        result.extend((value, mask))
    return result


def forward_gradient_case(architecture, history_hours, device, weather_features=None):
    torch.manual_seed(71)
    model_config = config(architecture, history_hours, weather_features=weather_features)
    model = DirectFlowModel(model_config).to(device).eval()
    batch_size = 3 if weather_features is not None else 1
    inputs = batch(history_hours, device, len(model_config.weather_features), batch_size)
    prediction = model(*inputs)
    assert tuple(prediction.shape) == (batch_size, 18)
    assert bool(torch.isfinite(prediction).all())
    target = torch.linspace(-0.3, 0.7, 18, device=device)[None].expand(batch_size, -1).clone()
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[:, 4] = False
    target[:, 4] = float("nan")
    loss = masked_mse(prediction, target, mask)
    loss.backward()
    missing, nonfinite = [], []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            missing.append(name)
        elif not bool(torch.isfinite(parameter.grad).all()):
            nonfinite.append(name)
    assert not missing, f"Missing gradients: {missing}"
    assert not nonfinite, f"Nonfinite gradients: {nonfinite}"
    if architecture == "Mamba2_Transformer":
        assert type(model.mamba).__name__ == "Mamba2Mixer"
        assert type(model.mamba_norm).__module__.startswith("transformers.")
    return {"output_shape": list(prediction.shape), "parameter_count":
            sum(p.numel() for p in model.parameters()), "finite_all_parameter_gradients": True,
            "loss": float(loss.detach().cpu()), "missing_input_masks_exercised": True,
            "history_features": list(model_config.history_features),
            "weather_features": list(model_config.weather_features),
            "state_features": list(model_config.state_features)}


def checkpoint_case(architecture, history_hours, device):
    """Save after one synthetic update; compare the next update after restore.

    Three total optimizer steps: pre-save, uninterrupted, restored branch.
    This tests model/optimizer/RNG state, including nonzero-dropout training mode.
    """
    torch.manual_seed(272)
    model = DirectFlowModel(config(architecture, history_hours, dropout=0.1)).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False)
    inputs = batch(history_hours, device)
    target = torch.linspace(-0.5, 0.5, 18, device=device)[None]
    mask = torch.ones_like(target, dtype=torch.bool)

    def update():
        optimizer.zero_grad(set_to_none=True)
        loss = masked_mse(model(*inputs), target, mask)
        loss.backward()
        optimizer.step()
        return loss.detach().cpu()

    update()
    checkpoint = {"model": copy.deepcopy(model.state_dict()),
                  "optimizer": copy.deepcopy(optimizer.state_dict()),
                  "cpu_rng": torch.get_rng_state()}
    if device == "mps":
        checkpoint["device_rng"] = torch.mps.get_rng_state()
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    expected_loss = update()
    expected = copy.deepcopy(model.state_dict())
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    buffer.seek(0)
    # Keep Adam's non-capturable step counter and RNG bytes on CPU. Loading the
    # whole archive onto MPS moves CPU bookkeeping too; load_state_dict moves
    # model parameters and optimizer moments to their parameter devices.
    restored = torch.load(buffer, map_location="cpu", weights_only=True)
    model.load_state_dict(restored["model"])
    optimizer.load_state_dict(restored["optimizer"])
    torch.set_rng_state(restored["cpu_rng"].cpu())
    if device == "mps":
        torch.mps.set_rng_state(restored["device_rng"].cpu())
    actual_loss = update()
    # Same-device replay must be exact; never silently relax or switch devices.
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    actual_optimizer = optimizer.state_dict()
    for key, state in expected_optimizer["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(actual_optimizer["state"][key][name], value, rtol=0, atol=0)
    return {"next_update_parameter_parity": "exact", "next_update_optimizer_parity": "exact",
            "dropout": 0.1, "optimizer_updates": 3, "checkpoint_bytes": buffer.getbuffer().nbytes}


def ssd_oracle_case(device="cpu"):
    """Compare forward, final state and gradients against float64 CPU recurrence."""
    results = []
    for length in (24, 168):
        generator = torch.Generator().manual_seed(12 + length)
        base = [torch.randn(1, length, 2, 3, generator=generator, dtype=torch.float64),
                torch.rand(1, length, 2, generator=generator, dtype=torch.float64) * .1,
                -torch.rand(2, generator=generator, dtype=torch.float64),
                torch.randn(1, length, 2, 4, generator=generator, dtype=torch.float64),
                torch.randn(1, length, 2, 4, generator=generator, dtype=torch.float64),
                torch.randn(2, generator=generator, dtype=torch.float64)]
        expected_inputs = [v.clone().requires_grad_() for v in base]
        expected = sequential(*expected_inputs)
        expected_grads = torch.autograd.grad(sum(v.square().mean() for v in expected), expected_inputs)
        dtype = torch.float64 if device == "cpu" else torch.float32
        inputs = [v.to(device=device, dtype=dtype).requires_grad_() for v in base]
        actual = ssd(*inputs, chunk=32)
        gradients = torch.autograd.grad(sum(v.square().mean() for v in actual), inputs)
        atol, rtol = (1e-10, 1e-10) if device == "cpu" else (3e-5, 3e-4)
        for a, b in zip((*actual, *gradients), (*expected, *expected_grads)):
            torch.testing.assert_close(a.detach().cpu().double(), b.detach(), atol=atol, rtol=rtol)
        results.append({"length": length, "padding_steps": (-length) % 32,
                        "forward_state_gradient_oracle": "PASS", "atol": atol, "rtol": rtol})
    # Also exercise the full HF-initialized mixer projection/convolution/gate/norm.
    torch.manual_seed(52)
    model = DirectFlowModel(config("Mamba2_Transformer"))
    u_cpu = torch.randn(1, 24, 64)
    expected = mixer_math(model.mamba, u_cpu, reference=True)
    model = model.to(device)
    actual = mixer_math(model.mamba, u_cpu.to(device))
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a.detach().cpu().double(), b.detach(), atol=3e-5, rtol=3e-4)
    return {"ssd_cases": results, "initialized_hf_mixer_float64_oracle": "PASS",
            "cpu_reference_only": True, "device_under_test": device}


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure_threads()

    def test_all_four_models_h18_l24_and_l168(self):
        for architecture in ARCHITECTURES:
            for length in (24, 168):
                with self.subTest(architecture=architecture, history_hours=length):
                    forward_gradient_case(architecture, length, "cpu")

    def test_ssd_forward_state_and_gradient_float64_oracle(self):
        ssd_oracle_case()

    def test_masks_reject_hidden_nonzero_nan_and_wrong_dtype(self):
        model = DirectFlowModel(config("LSTM"))
        for value in (123., float("nan")):
            inputs = batch()
            inputs[2].reshape(-1)[1] = value
            with self.assertRaises(ValueError):
                model(*inputs)
        inputs = batch()
        inputs[1] = inputs[1].float()
        with self.assertRaises(ValueError):
            model(*inputs)

    def test_targets_not_forward_inputs_and_noncontract_sources_rejected(self):
        with self.assertRaises(TypeError):
            DirectFlowModel(config("LSTM"))(*batch(), target=torch.zeros(1, 18))
        with self.assertRaises(ValueError):
            ModelConfig("LSTM", ("flow",), ("NWM_forecast",), (), 24, (1, 2))

    def test_masked_loss_selects_before_nan_arithmetic(self):
        prediction = torch.tensor([[1., 8., 3.]], requires_grad=True)
        target = torch.tensor([[2., float("nan"), 5.]])
        mask = torch.tensor([[True, False, True]])
        loss = masked_mse(prediction, target, mask)
        self.assertEqual(loss.item(), 2.5)
        loss.backward()
        torch.testing.assert_close(prediction.grad, torch.tensor([[-1., 0., -2.]]))
        with self.assertRaises(ValueError):
            masked_mse(prediction, target, torch.zeros_like(mask))


if __name__ == "__main__":
    unittest.main(verbosity=2)
