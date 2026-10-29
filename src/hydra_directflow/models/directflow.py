"""Configurable direct-flow candidates, with explicit input masks.

Adapted from the user's resident hydra_issued/models.py architecture layout.
No legacy NWM/ERA5 data contract, fitted parameters or checkpoints are reused.
This module does not fetch data, create optimizers or start training.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn


ARCHITECTURES = ("LSTM", "Transformer", "GRU_Transformer", "Mamba2_Transformer")


@dataclass(frozen=True)
class ModelConfig:
    architecture: str
    history_features: tuple[str, ...]
    weather_features: tuple[str, ...]
    state_features: tuple[str, ...]
    history_hours: int
    horizon_hours: tuple[int, ...]
    width: int = 64
    dropout: float = 0.1
    # These are model geometry choices, not a frozen experiment/data policy.
    site_mode: str = "per_gauge"

    def __post_init__(self):
        if self.architecture not in ARCHITECTURES:
            raise ValueError("Unknown architecture")
        if self.site_mode != "per_gauge":
            raise ValueError("Only the explicit per-gauge adapter is prepared")
        if (type(self.history_hours) is not int or self.history_hours < 1
                or not self.horizon_hours
                or any(type(x) is not int or x < 1 for x in self.horizon_hours)
                or tuple(sorted(set(self.horizon_hours))) != self.horizon_hours):
            raise ValueError("Positive history length and increasing unique horizons required")
        for names in (self.history_features, self.weather_features, self.state_features):
            if (not isinstance(names, tuple) or len(set(names)) != len(names)
                    or any(not isinstance(x, str) or not x for x in names)):
                raise ValueError("Unique explicit feature-name tuples required")
            if any(word in name.lower() for name in names
                   for word in ("nwm", "era5", "radiation", "dswrf")):
                raise ValueError("Feature source is outside the HRRR direct-flow contract")
        if not self.history_features or not self.weather_features:
            raise ValueError("History and issued-weather schemas must be nonempty")
        if (type(self.width) is not int or self.width < 4 or self.width % 4
                or not 0 <= self.dropout < 1):
            raise ValueError("Width divisible by four and dropout in [0,1) required")
        if self.architecture == "Mamba2_Transformer" and self.width != 64:
            raise ValueError("Prepared Mamba2 parameter layout requires width64")


def positional(coordinates, width):
    """Use the actual horizon offsets, including for nonconsecutive horizons."""
    position = torch.tensor(coordinates, dtype=torch.float32)[:, None]
    rates = torch.exp(torch.arange(0, width, 2) * (-math.log(10000) / width))
    result = torch.zeros(len(coordinates), width)
    result[:, 0::2] = torch.sin(position * rates)
    result[:, 1::2] = torch.cos(position * rates)
    return result


class DirectFlowModel(nn.Module):
    """Consume already transformed values plus masks; predict [batch,horizon].

    Missing transformed inputs must carry an explicit false mask and a zero
    placeholder. That convention must be selected in the preprocessing policy
    before real data use. It is not physical zero flow/weather. Masks themselves
    are model channels. Targets and target masks are never forward arguments.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        width = config.width
        self.history_projection = nn.Linear(2 * len(config.history_features), width)
        self.weather_projection = nn.Linear(
            2 * (len(config.weather_features) + len(config.state_features)), width)
        self.register_buffer("history_position", positional(
            tuple(range(-config.history_hours, 0)), width))
        self.register_buffer("horizon_position", positional(config.horizon_hours, width))
        self.dropout = nn.Dropout(config.dropout)
        if config.architecture == "LSTM":
            self.history_encoder = nn.LSTM(width, width, batch_first=True)
            self.horizon_decoder = nn.LSTM(width, width, batch_first=True)
        elif config.architecture == "Transformer":
            layer = nn.TransformerEncoderLayer(width, 4, 2 * width,
                dropout=config.dropout, activation="gelu", batch_first=True, norm_first=True)
            self.history_encoder = nn.TransformerEncoder(
                layer, 1, enable_nested_tensor=False)
        elif config.architecture == "GRU_Transformer":
            self.history_encoder = nn.GRU(width, width, batch_first=True)
        else:
            # Initialized architecture only: no from_pretrained or downloads.
            from transformers import Mamba2Config, Mamba2Model
            initialized = Mamba2Model(Mamba2Config(vocab_size=1, hidden_size=64,
                num_heads=4, head_dim=32, state_size=64, num_hidden_layers=1,
                expand=2, n_groups=1, conv_kernel=4, chunk_size=32,
                use_bias=False, use_conv_bias=True, rms_norm=True,
                residual_in_fp32=True))
            self.mamba_norm = initialized.layers[0].norm
            self.mamba = initialized.layers[0].mixer
        if config.architecture != "LSTM":
            layer = nn.TransformerDecoderLayer(width, 4, 2 * width,
                dropout=config.dropout, activation="gelu", batch_first=True, norm_first=True)
            self.horizon_decoder = nn.TransformerDecoder(layer, 1)
        self.final_norm = nn.LayerNorm(width)
        self.flow_head = nn.Linear(width, 1)

    @staticmethod
    def _masked_channels(values, mask, shape, name):
        if (tuple(values.shape) != shape or tuple(mask.shape) != shape
                or mask.dtype != torch.bool or not values.is_floating_point()
                or mask.device != values.device):
            raise ValueError(name + " values/mask geometry, dtype or device differs")
        if not bool(torch.isfinite(values).all()):
            raise ValueError(name + " contains nonfinite transformed inputs")
        if bool((values.masked_select(~mask) != 0).any()):
            raise ValueError(name + " unobserved channels require explicit zero placeholders")
        return torch.cat((values, mask.to(values.dtype)), dim=-1)

    def forward(self, history, history_mask, issued_weather, issued_mask,
                origin_state, state_mask):
        c = self.config
        if history.ndim != 3:
            raise ValueError("History must have three dimensions")
        batch = history.shape[0]
        horizon = len(c.horizon_hours)
        h = self._masked_channels(history, history_mask,
            (batch, c.history_hours, len(c.history_features)), "history")
        f = self._masked_channels(issued_weather, issued_mask,
            (batch, horizon, len(c.weather_features)), "issued_weather")
        s = self._masked_channels(origin_state, state_mask,
            (batch, len(c.state_features)), "origin_state")
        if (issued_weather.dtype != history.dtype or origin_state.dtype != history.dtype
                or issued_weather.device != history.device or origin_state.device != history.device):
            raise ValueError("All inputs must share dtype and device")
        f = torch.cat((f, s[:, None, :].expand(-1, horizon, -1)), dim=-1)
        h = self.dropout(self.history_projection(h) + self.history_position[None])
        f = self.dropout(self.weather_projection(f) + self.horizon_position[None])
        if c.architecture == "LSTM":
            _, state = self.history_encoder(h)
            output, _ = self.horizon_decoder(f, state)
        else:
            if c.architecture == "Transformer":
                memory = self.history_encoder(h)
            elif c.architecture == "GRU_Transformer":
                memory, _ = self.history_encoder(h)
            else:
                from .mamba2_math import mixer_math
                mixed, _ = mixer_math(self.mamba, self.mamba_norm(h))
                memory = h + mixed
            # Issued-weather tokens must all be available by the information
            # cutoff. Full horizon interaction never includes future targets.
            output = self.horizon_decoder(f, memory)
        # Output is in the declared target transform's space. Nonnegativity and
        # inverse transformation belong to the separately selected data policy.
        return self.flow_head(self.final_norm(output)).squeeze(-1)


def masked_mse(prediction, target, target_mask):
    """Example masked objective, pending study loss selection.

    Select observed targets before arithmetic, so NaN missing labels cannot
    contaminate the loss. Do not use forecast support masks as target evidence.
    """
    if (prediction.shape != target.shape or target_mask.shape != target.shape
            or target_mask.dtype != torch.bool or not bool(target_mask.any())):
        raise ValueError("Matching tensors and at least one admitted target required")
    p, y = prediction[target_mask], target[target_mask]
    if not bool(torch.isfinite(p).all() and torch.isfinite(y).all()):
        raise ValueError("Admitted prediction/target contains nonfinite values")
    return (p - y).square().mean()
