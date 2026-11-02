"""Train-only transformations and restartable training mechanics.

This module does not admit a scientific campaign. Its runner consumes an already
admitted development bundle and an explicit bounded update allowance. Protocol,
cohort, source and execution approval remain external prerequisites. Protected
coordinates are rejected before numerical training data are inspected.
"""
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import re

import numpy as np

from .contract import ARCHITECTURES, require
from .reference import ASSESSMENT_START, digest

INPUTS = ("history", "weather", "state")
FEATURES = {"history": ("discharge_cms", "age_hours"),
            "weather": ("precip_mm", "temp_k", "specific_humidity", "swe_kg_m2"),
            "state": ("soil_water_fraction", "soil_age_hours")}


def implementation_fingerprint():
    root = Path(__file__).resolve().parent
    return dict(code={name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                      for name in ("training.py", "contract.py", "reference.py", "models/directflow.py", "models/mamba2_math.py")},
                dependencies={name: importlib.metadata.version(name) for name in ("numpy", "torch", "transformers")})


def array_identity(arrays):
    result = hashlib.sha256()
    for name in sorted(arrays):
        value = np.asarray(arrays[name])
        require(value.dtype.kind in "fbiuUS", "Only non-object arrays are admitted")
        result.update(json.dumps([name, value.dtype.str, value.shape], separators=(",", ":")).encode())
        result.update(value.tobytes(order="C"))
    return result.hexdigest()


def validate_coordinates(arrays, history_hours):
    """Coordinate checks precede value access; histories may cross backward only."""
    origins, history, target = (np.asarray(arrays[k]) for k in ("origin_time", "history_time", "target_time"))
    n = len(origins)
    require(origins.dtype.kind in "iu" and history.dtype.kind in "iu" and target.dtype.kind in "iu", "Integer UTC clocks required")
    require(origins.shape == (n,) and 0 < n <= 200000 and np.all(origins % 3600 == 0), "Bounded exact-hour origins required")
    require(history.shape == (n, history_hours) and target.shape == (n, 18), "Clock geometry differs")
    require(np.array_equal(history, origins[:, None] - np.arange(history_hours + 1, 1, -1)[None, :] * 3600), "History must end at C minus two hours")
    require(np.array_equal(target, origins[:, None] + np.arange(1, 19)[None, :] * 3600), "H18 targets must follow C")
    require(np.all(target < ASSESSMENT_START), "Protected assessment or crossing targets are not admitted")
    case_ids = np.asarray(arrays["case_id"])
    require(case_ids.shape == (n,) and case_ids.dtype.kind in "US" and len(set(case_ids.tolist())) == n, "Unique case IDs required")
    site = np.asarray(arrays["site_id"])
    require(site.shape == (n,) and site.dtype.kind in "iu" and len(set(site.tolist())) == 1 and int(site[0]) in range(3), "One explicit gauge per fit required")
    return n


def validate_development_bundle(arrays, history_hours):
    n = validate_coordinates(arrays, history_hours)
    shapes = {"history": (n, history_hours, 2), "weather": (n, 18, 4), "state": (n, 2), "target": (n, 18)}
    for name, shape in shapes.items():
        values, mask = np.asarray(arrays[name]), np.asarray(arrays[name + "_mask"])
        require(values.shape == mask.shape == shape and values.dtype.kind == "f" and mask.dtype == np.dtype(bool), "Values/mask shape or type differs: " + name)
        require(np.isfinite(values[mask]).all(), "Nonfinite admitted value: " + name)
    return n


def validate_split(arrays, train_indices, validation_indices, *, boundary_epoch):
    """Return immutable indices after label-window purging, never random splitting."""
    n = len(arrays["origin_time"])
    sets = []
    for indices in (train_indices, validation_indices):
        require(all(type(i) is int for i in indices), "Split indices must be integers")
        values = tuple(indices)
        require(values and len(set(values)) == len(values) and all(0 <= i < n for i in values), "Nonempty unique in-range split indices required")
        sets.append(values)
    train, validation = sets
    require(not set(train) & set(validation), "Training and validation cases overlap")
    require(type(boundary_epoch) is int and boundary_epoch % 3600 == 0 and boundary_epoch < ASSESSMENT_START, "Explicit development split boundary required")
    require(np.all(arrays["target_time"][list(train)] < boundary_epoch), "Training labels cross the validation boundary")
    require(np.all(arrays["origin_time"][list(validation)] >= boundary_epoch), "Validation origins precede their boundary")
    return train, validation


@dataclass(frozen=True)
class TrainOnlyTransform:
    """One declared candidate: affine standardization fitted on training cells.

    Constant training channels use scale1 and retain an explicit flag. Input
    missingness gets a zero network placeholder; missing labels remain NaN.
    This is implementation capability, not selection of the study transform.
    """
    parameters: dict
    training_case_ids_sha256: str
    training_values_sha256: str
    schema: str = "hydra-train-only-affine/v1"

    @classmethod
    def fit(cls, arrays, train_indices):
        indices = tuple(train_indices)
        require(indices and all(type(i) is int for i in indices) and len(set(indices)) == len(indices), "Explicit unique training indices required")
        require(all(0 <= i < len(arrays["case_id"]) for i in indices), "Training index outside bundle")
        parameters, selected = {}, {}
        for name in INPUTS + ("target",):
            values = np.asarray(arrays[name])[list(indices)]
            mask = np.asarray(arrays[name + "_mask"])[list(indices)]
            require(mask.dtype == np.dtype(bool) and values.shape == mask.shape and np.isfinite(values[mask]).all(), "Training scaler inputs invalid")
            selected[name], selected[name + "_mask"] = values, mask
            columns = 1 if name == "target" else values.shape[-1]
            channel = []
            for column in range(columns):
                observed = values[mask] if name == "target" else values[..., column][mask[..., column]]
                require(len(observed) > 0, "No observed training cells for " + name)
                mean, deviation = float(np.mean(observed, dtype=np.float64)), float(np.std(observed, dtype=np.float64))
                require(np.isfinite(mean) and np.isfinite(deviation), "Training scaler statistics overflowed")
                channel.append(dict(mean=mean, scale=deviation if deviation > 0 else 1., constant=deviation == 0, cells=len(observed)))
            parameters[name] = channel
        return cls(parameters, digest(np.asarray(arrays["case_id"])[list(indices)].tolist()), array_identity(selected))

    @classmethod
    def from_dict(cls, value):
        require(set(value) == {"parameters", "training_case_ids_sha256", "training_values_sha256", "schema"} and value["schema"] == "hydra-train-only-affine/v1", "Unknown transform state")
        result = cls(**value)
        for name, channels in result.parameters.items():
            require(name in INPUTS + ("target",) and len(channels) == (1 if name == "target" else len(FEATURES[name])), "Transform feature schema differs")
            for p in channels:
                require(np.isfinite(p["mean"]) and np.isfinite(p["scale"]) and p["scale"] > 0 and type(p["cells"]) is int and p["cells"] > 0, "Invalid transform statistics")
        require(set(result.parameters) == set(INPUTS + ("target",)), "Transform channels missing")
        return result

    def transform(self, arrays):
        mapped = {}
        for name in INPUTS + ("target",):
            source, mask = np.asarray(arrays[name]), np.asarray(arrays[name + "_mask"])
            require(source.shape == mask.shape and mask.dtype == np.dtype(bool) and np.isfinite(source[mask]).all(), "Transform values/masks differ")
            output = np.full(source.shape, np.nan if name == "target" else 0., dtype=np.float32)
            for column, p in enumerate(self.parameters[name]):
                if name == "target":
                    output[mask] = (source[mask] - p["mean"]) / p["scale"]
                else:
                    admitted = mask[..., column]
                    output[..., column][admitted] = (source[..., column][admitted] - p["mean"]) / p["scale"]
            require(np.isfinite(output[mask]).all(), "Transformed admitted values overflowed")
            mapped[name], mapped[name + "_mask"] = output, mask.copy()
        return mapped

    def inverse_target(self, value):
        p = self.parameters["target"][0]
        return value * p["scale"] + p["mean"]


class EpochSampler:
    """Within-training permutation only: split membership never changes."""
    def __init__(self, indices, *, seed, shuffle, epoch=0, cursor=0):
        self.indices = tuple(indices)
        require(self.indices and len(set(self.indices)) == len(self.indices) and all(type(i) is int and i >= 0 for i in self.indices), "Sampler needs unique training indices")
        require(type(seed) is int and 0 <= seed < 2**32 and type(shuffle) is bool, "Explicit sampler seed and order required")
        require(type(epoch) is int and epoch >= 0 and type(cursor) is int and 0 <= cursor <= len(self.indices), "Sampler position differs")
        self.seed, self.shuffle, self.epoch, self.cursor = seed, shuffle, epoch, cursor

    def _order(self):
        if not self.shuffle:
            return list(self.indices)
        generator = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch]))
        return generator.permutation(self.indices).tolist()

    def next(self, batch_size):
        require(type(batch_size) is int and batch_size > 0, "Positive batch size required")
        if self.cursor == len(self.indices):
            self.epoch += 1
            self.cursor = 0
        order = self._order()
        batch = order[self.cursor:self.cursor + batch_size]
        self.cursor += len(batch)
        return batch

    def state(self):
        return dict(indices=list(self.indices), seed=self.seed, shuffle=self.shuffle, epoch=self.epoch, cursor=self.cursor,
                    order_sha256=digest(self._order()))

    @classmethod
    def restore(cls, state):
        result = cls(state["indices"], seed=state["seed"], shuffle=state["shuffle"], epoch=state["epoch"], cursor=state["cursor"])
        require(result.state()["order_sha256"] == state["order_sha256"], "Sampler algorithm/order changed")
        return result


@dataclass(frozen=True)
class TrainerSpec:
    architecture: str
    device: str
    history_hours: int
    batch_size: int
    learning_rate: float
    seed: int
    shuffle_training: bool
    max_updates: int

    def __post_init__(self):
        require(self.architecture in ARCHITECTURES and self.device in ("cpu", "mps"), "Prepared local architecture/device required")
        require(self.history_hours in (24, 168) and type(self.batch_size) is int and 1 <= self.batch_size <= 64, "Bounded training geometry required")
        require(type(self.max_updates) is int and 0 <= self.max_updates <= 100000, "Explicit update ceiling required")
        require(type(self.seed) is int and 0 <= self.seed < 2**32 and type(self.shuffle_training) is bool, "Explicit seed/order required")
        require(type(self.learning_rate) in (float, int) and np.isfinite(self.learning_rate) and 0 < self.learning_rate <= .01, "Finite learning rate required")


@dataclass
class EarlyStopping:
    max_epochs: int
    min_epochs: int
    patience: int
    min_improvement: float
    completed_epochs: int = 0
    best_loss: float | None = None
    stale_epochs: int = 0

    def __post_init__(self):
        require(type(self.max_epochs) is int and type(self.min_epochs) is int and 1 <= self.min_epochs <= self.max_epochs <= 1000,
                "Explicit bounded epoch limits required")
        require(type(self.patience) is int and self.patience > 0 and np.isfinite(self.min_improvement) and self.min_improvement >= 0,
                "Explicit stopping patience and improvement required")
        require(0 <= self.completed_epochs <= self.max_epochs and self.stale_epochs >= 0
                and (self.best_loss is None or np.isfinite(self.best_loss)), "Invalid stopping state")

    def observe(self, loss):
        require(not self.finished and np.isfinite(loss) and loss >= 0, "Finite stopping loss and active allowance required")
        improved = self.best_loss is None or loss < self.best_loss - self.min_improvement
        self.completed_epochs += 1
        if improved:
            self.best_loss, self.stale_epochs = float(loss), 0
        else:
            self.stale_epochs += 1
        return improved

    @property
    def finished(self):
        return self.completed_epochs >= self.max_epochs or (self.completed_epochs >= self.min_epochs and self.stale_epochs >= self.patience)


class DevelopmentTrainer:
    """Explicit mechanics API, with no automatic scientific execution gate bypass."""
    def __init__(self, arrays, *, train_indices, validation_indices, boundary_epoch, spec, source_identity):
        require(isinstance(spec, TrainerSpec), "Explicit trainer specification required")
        validate_development_bundle(arrays, spec.history_hours)
        self.train, self.validation = validate_split(arrays, train_indices, validation_indices, boundary_epoch=boundary_epoch)
        require(isinstance(source_identity, str) and re.fullmatch(r"[0-9a-f]{64}", source_identity), "Admitted source manifest digest required")
        require(np.all(arrays["target_mask"][list(self.train)].any(axis=1)), "Training case has no observed label")
        self.spec, self.arrays, self.boundary = spec, arrays, boundary_epoch
        self.source_identity, self.array_sha256 = source_identity, array_identity(arrays)
        self.scaler = TrainOnlyTransform.fit(arrays, self.train)
        self.sampler = EpochSampler(self.train, seed=spec.seed, shuffle=spec.shuffle_training)
        self.updates = 0
        self.stopping, self.best_model_state, self.epoch_records = None, None, []
        import torch
        from .models.directflow import DirectFlowModel, ModelConfig
        self.torch = torch
        self.runtime = dict(torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
                            default_dtype=str(torch.get_default_dtype()), deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
        torch.manual_seed(spec.seed)
        if spec.device == "mps":
            require(torch.backends.mps.is_available(), "MPS unavailable; no fallback")
            torch.mps.manual_seed(spec.seed)
        config = ModelConfig(spec.architecture, FEATURES["history"], FEATURES["weather"], FEATURES["state"], spec.history_hours, tuple(range(1, 19)))
        self.model = DirectFlowModel(config).to(spec.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=spec.learning_rate, foreach=False)
        self.host = {name: torch.from_numpy(value) for name, value in self.scaler.transform(arrays).items()}

    def _batch(self, indices):
        return {name: value[list(indices)].to(self.spec.device) for name, value in self.host.items()}

    def update(self):
        require(self.updates < self.spec.max_updates, "Declared update allowance exhausted")
        from .models.directflow import masked_mse
        indices = self.sampler.next(self.spec.batch_size)
        batch = self._batch(indices)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        prediction = self.model(*(batch[name] for name in ("history", "history_mask", "weather", "weather_mask", "state", "state_mask")))
        loss = masked_mse(prediction, batch["target"], batch["target_mask"])
        loss.backward()
        require(all(p.grad is not None and bool(self.torch.isfinite(p.grad).all()) for p in self.model.parameters()), "Nonfinite or missing gradient")
        self.optimizer.step()
        self.updates += 1
        require(all(bool(self.torch.isfinite(p).all()) for p in self.model.parameters()), "Nonfinite parameter after update")
        return dict(indices=indices, case_ids=self.arrays["case_id"][indices].tolist(), loss=float(loss.detach().cpu()), updates=self.updates)

    def stopping_loss(self):
        """Masked standardized MSE on the supplied inner stopping subset only."""
        self.model.eval()
        total, count = 0., 0
        with self.torch.no_grad():
            for offset in range(0, len(self.validation), self.spec.batch_size):
                batch = self._batch(self.validation[offset:offset + self.spec.batch_size])
                prediction = self.model(*(batch[name] for name in ("history", "history_mask", "weather", "weather_mask", "state", "state_mask")))
                mask = batch["target_mask"]
                errors = prediction[mask] - batch["target"][mask]
                require(bool(self.torch.isfinite(errors).all()), "Nonfinite stopping error")
                total += float(errors.square().sum().cpu())
                count += int(mask.sum().cpu())
        require(count > 0, "Inner stopping subset has no observed labels")
        return total / count

    def fit_epochs(self, stopping, *, checkpoint_directory=None):
        """An explicit development fit; never uses outer-CV or assessment scores.

        The caller must establish that validation_indices identify an approved
        INNER stopping subset. This class is not source/cohort authorization.
        """
        from .engineering import _cpu_tree
        require(isinstance(stopping, EarlyStopping), "Explicit stopping configuration required")
        if self.stopping is None:
            self.stopping = stopping
        else:
            require(asdict(self.stopping) == asdict(stopping), "Resume stopping state differs")
        while not self.stopping.finished:
            epoch = self.stopping.completed_epochs
            require(self.sampler.epoch in (max(0, epoch - 1), epoch), "Sampler and stopping epochs differ")
            while self.sampler.epoch < epoch or self.sampler.cursor < len(self.train):
                self.update()
            loss = self.stopping_loss()
            improved = self.stopping.observe(loss)
            if improved:
                self.best_model_state = _cpu_tree(self.model.state_dict(), self.torch)
            self.epoch_records.append(dict(epoch=self.stopping.completed_epochs, inner_stopping_loss=loss,
                                           updates=self.updates, improved=improved))
            if checkpoint_directory is not None:
                directory = Path(checkpoint_directory)
                directory.mkdir(parents=True, exist_ok=True)
                self.checkpoint(directory / f"epoch-{self.stopping.completed_epochs:04d}-update-{self.updates:08d}.pt")
        return list(self.epoch_records)

    def checkpoint(self, path):
        from .engineering import _cpu_tree
        torch = self.torch
        value = dict(schema="hydra-development-trainer-checkpoint/v1", spec=asdict(self.spec), source_identity=self.source_identity,
                     implementation=implementation_fingerprint(), runtime=self.runtime,
                     array_sha256=self.array_sha256, train_indices=list(self.train), validation_indices=list(self.validation), boundary_epoch=self.boundary,
                     transform=asdict(self.scaler), sampler=self.sampler.state(), updates=self.updates,
                     stopping=asdict(self.stopping) if self.stopping is not None else None,
                     best_model=self.best_model_state, epoch_records=self.epoch_records,
                     model=_cpu_tree(self.model.state_dict(), torch), optimizer=_cpu_tree(self.optimizer.state_dict(), torch),
                     cpu_rng=torch.get_rng_state(), mps_rng=torch.mps.get_rng_state().cpu() if self.spec.device == "mps" else None)
        path = Path(path)
        stage = path.with_name(path.name + ".writing")
        require(not path.exists() and not path.is_symlink() and not stage.exists() and not stage.is_symlink(), "Checkpoint and staging path must be new")
        with stage.open("xb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        stage.rename(path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @classmethod
    def restore(cls, path, arrays, *, expected_sha256, expected_source_identity):
        import torch
        from .engineering import _local_bytes
        raw = _local_bytes(path, 16 * 1024**2)
        require(len(raw) <= 16 * 1024**2 and hashlib.sha256(raw).hexdigest() == expected_sha256, "Checkpoint byte identity differs")
        value = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        require(value["schema"] == "hydra-development-trainer-checkpoint/v1" and value["source_identity"] == expected_source_identity,
                "Checkpoint schema/source identity differs")
        require(value["implementation"] == implementation_fingerprint(), "Trainer/model code or dependencies changed")
        spec = TrainerSpec(**value["spec"])
        result = cls(arrays, train_indices=value["train_indices"], validation_indices=value["validation_indices"],
                     boundary_epoch=value["boundary_epoch"], spec=spec, source_identity=expected_source_identity)
        require(result.array_sha256 == value["array_sha256"] and asdict(result.scaler) == value["transform"], "Dataset or train-only transform changed")
        require(result.runtime == value["runtime"], "Training runtime settings changed")
        result.scaler = TrainOnlyTransform.from_dict(value["transform"])
        result.sampler = EpochSampler.restore(value["sampler"])
        require(result.sampler.indices == result.train and result.sampler.seed == spec.seed and result.sampler.shuffle == spec.shuffle_training,
                "Checkpoint sampler differs from training split/spec")
        result.updates = value["updates"]
        require(type(result.updates) is int and 0 <= result.updates <= spec.max_updates, "Invalid checkpoint update count")
        result.model.load_state_dict(value["model"])
        result.optimizer.load_state_dict(value["optimizer"])
        result.stopping = EarlyStopping(**value["stopping"]) if value["stopping"] is not None else None
        result.best_model_state = value["best_model"]
        result.epoch_records = value["epoch_records"]
        torch.set_rng_state(value["cpu_rng"])
        if spec.device == "mps":
            torch.mps.set_rng_state(value["mps_rng"])
        return result
