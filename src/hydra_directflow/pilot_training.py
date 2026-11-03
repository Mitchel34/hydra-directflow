"""Separate H4 LSTM/CPU pilot mechanics, not scientific execution authority.

Only callers with a separately bounded qualification/pilot allowance may invoke
updates. No CLI, real-data execution switch, data reader or cloud operation is
provided. Original H2 gradient statistics, explicit provenance, role clocks,
policies, sampler order and CPU RNG are bound to a distinct checkpoint schema.
"""
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
from types import MappingProxyType

import numpy as np

from .reference import ASSESSMENT_START, digest, require
from .contract import GAUGE_IDS
from .quality import PRIMARY_APPROVED, admit_usgs_quality
from .training import (FULL_FEATURE_SCHEMA, INPUTS, FEATURES, TrainOnlyTransform,
    EpochSampler, EarlyStopping, array_identity, validate_coordinates,
    validate_development_bundle, implementation_fingerprint)
from .trainer_bundle import _roles, _refit_policy, refit_label_mask
from .pilot_provenance import admit_history_provenance, ValidatedProvenanceBatch

CHECKPOINT_SCHEMA = "hydra-h4-pilot-checkpoint/v1"
H4_FEATURE_SCHEMA = "PILOT_HRRR_H4_W4_S2_V1"
H4_FEATURES = ("discharge_cms", "age_hours", "was_missing", "elapsed_since_real_hours")
COORDINATES = ("origin_time", "history_time", "target_time", "site_id", "case_id")
MODEL_KEYS = tuple(key for field in INPUTS for key in (field, field + "_mask"))


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _json_default(item):
    """Let the native JSON encoder traverse normal dictionaries/sequences."""
    if isinstance(item, Mapping): return dict(item)
    if isinstance(item, np.ndarray): return item.tolist()
    if isinstance(item, np.generic):
        value = item.item()
        # NumPy extended scalars can return their own type from item(). The
        # previous recursive converter rejected these with RecursionError.
        if isinstance(value, np.generic) and type(value) is type(item):
            raise RecursionError("NumPy scalar has no native JSON representation")
        return value
    raise TypeError("Object of type " + type(item).__name__ + " is not JSON serializable")


def _json_text(value):
    return json.dumps(value, default=_json_default, sort_keys=True,
                      separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _canonical(value):
    """Preserve the legacy plain JSON value contract without a Python tree copy."""
    return json.loads(_json_text(value))


def _needs_key_normalization(value):
    """Find rare JSON-key coercions without recursively copying scalar leaves.

    The legacy JSON roundtrip stringifies keys before the final sorted digest.
    Sorting integer keys directly can therefore produce a different digest.
    Normal provenance has string keys; keep the fallback for legacy supported
    key types. Retained references prevent ID reuse for temporary object-array
    lists during this bounded, per-call scan; there is no cross-call cache.
    """
    pending, seen = [value], {}
    primitive = (str, int, float, bool, type(None))
    while pending:
        item = pending.pop()
        if id(item) in seen: continue
        seen[id(item)] = item
        if isinstance(item, Mapping):
            if any(type(key) is not str for key in item): return True
            children = item.values()
        elif isinstance(item, np.ndarray):
            if type(item) is np.ndarray and not item.dtype.hasobject: continue
            children = (item.tolist(),)
        elif isinstance(item, np.generic):
            children = (_json_default(item),)
        elif isinstance(item, (tuple, list)):
            children = item
        else:
            continue
        pending.extend(child for child in children if type(child) not in primitive)
    return False


def _canonical_bytes(value):
    """Byte-equivalent to JSON encoding of the old _canonical(value) result."""
    if _needs_key_normalization(value):
        value = _canonical(value)
    return _json_text(value).encode("utf-8")


def _canonical_digest(value):
    """Hash normal provenance with one encoding and no JSON roundtrip."""
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _freeze_metadata(value):
    if isinstance(value,Mapping): return MappingProxyType({key:_freeze_metadata(child) for key,child in value.items()})
    if isinstance(value,(list,tuple)): return tuple(_freeze_metadata(child) for child in value)
    return value


def _ridge_identity(model):
    if model is None: return None
    require(_sha(model.identity), "Explicit serialized ridge model identity required")
    return model.identity


def _ridge_state_sha256(model):
    return None if model is None else _canonical_digest(model.to_dict())


def pilot_implementation_fingerprint():
    parent = Path(__file__).resolve().parent
    return {**implementation_fingerprint(), "pilot_code": {
        name: hashlib.sha256((parent / name).read_bytes()).hexdigest()
        for name in ("pilot_training.py", "pilot_inputs.py", "pilot_provenance.py", "gap_filling.py", "gap_policies.py", "quality.py", "trainer_bundle.py", "inference.py", "engineering.py")}}


@dataclass(frozen=True)
class PilotSpec:
    mode: str
    seed: int = 17
    batch_size: int = 64
    learning_rate: float = .001
    shuffle_training: bool = True
    max_updates: int = 0
    mask_seed: int = 1729
    architecture: str = "LSTM"
    device: str = "cpu"
    history_hours: int = 168
    width: int = 64
    feature_schema: str = H4_FEATURE_SCHEMA

    def __post_init__(self):
        require(self.mode in ("clean", "augmented") and self.architecture == "LSTM" and self.device == "cpu"
                and self.history_hours == 168 and self.width == 64 and self.feature_schema == H4_FEATURE_SCHEMA,
                "Separate fixed LSTM/CPU/H4/L168 pilot configuration required")
        require(type(self.batch_size) is int and 1 <= self.batch_size <= 64 and type(self.max_updates) is int
                and 0 <= self.max_updates <= 100000, "Explicit bounded batch/update allowance required")
        require(type(self.seed) is int and 0 <= self.seed < 2**32 and type(self.mask_seed) is int
                and 0 <= self.mask_seed < 2**32 and type(self.shuffle_training) is bool, "Explicit pilot seeds/order required")
        require(type(self.learning_rate) in (int, float) and np.isfinite(self.learning_rate)
                and 0 < self.learning_rate <= .01, "Finite bounded pilot learning rate required")


def validate_pilot_roles(arrays, *, train_indices, inner_indices, role_time_bounds, refit_at_utc):
    """All clocks/role/refit guards precede any numerical input or label access."""
    n = validate_coordinates(arrays, 168)
    origin = np.asarray(arrays["origin_time"])
    require(np.all((origin >= 169 * 3600) & (origin < ASSESSMENT_START)) and np.all(np.diff(origin.astype(np.int64)) > 0), "Chronological development pilot origins required")
    site = int(np.asarray(arrays["site_id"])[0])
    ids = np.asarray(arrays["case_id"]).tolist()
    expected = [GAUGE_IDS[site] + ":" + datetime.fromtimestamp(int(t), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") for t in origin]
    require(ids == expected, "Exact gauge/origin pilot case IDs required")
    roles = _roles(role_time_bounds)
    require({"fit", "inner_stop"} <= set(roles), "Explicit separate fit and clean-inner roles required")
    refit = _refit_policy(dict(refit_at_utc=refit_at_utc, simulated_label_delay_hours=2), roles)
    supplied = []
    for name, raw in (("fit", train_indices), ("inner_stop", inner_indices)):
        indices = tuple(raw)
        require(indices and all(type(i) is int and 0 <= i < n for i in indices)
                and indices == tuple(sorted(set(indices))), "Ordered unique explicit pilot role indices required")
        start, end = roles[name]
        future = np.asarray(arrays["target_time"])[list(indices)]
        require(np.all((origin[list(indices)] >= start) & (origin[list(indices)] < end))
                and np.all(future < end), "Pilot target clocks cross their explicit role window")
        require(refit_label_mask(future, refit_at_utc=refit_at_utc, simulated_label_delay_hours=2).all(),
                "Pilot gradient/inner labels were not available by simulated refit")
        supplied.append(indices)
    require(set(supplied[0]).isdisjoint(supplied[1]), "Pilot fit and inner cases overlap")
    return supplied[0], supplied[1], dict(role_time_bounds=_canonical(role_time_bounds), refit=refit,
        train_case_ids_sha256=digest([ids[i] for i in supplied[0]]), inner_case_ids_sha256=digest([ids[i] for i in supplied[1]]))


def _provenance(arrays, history_provenance):
    require(isinstance(history_provenance, Mapping) and set(history_provenance) == {"observation_times", "quality_flags", "source_ids"},
            "Explicit original slot observation clocks, qualifiers and source IDs required")
    n = len(arrays["origin_time"])
    result = _canonical(history_provenance)
    require(all(len(result[name]) == n and all(len(row) == 168 for row in result[name]) for name in result), "Slot provenance geometry differs")
    masks = np.asarray(arrays["history_mask"])
    require(masks.shape == (n,168,2) and masks.dtype == np.dtype(bool), "Original H2 source masks required")
    times, origins = np.asarray(arrays["history_time"]), np.asarray(arrays["origin_time"])
    for i in range(n):
        for j in range(168):
            stamp = result["observation_times"][i][j]
            require(stamp is None or type(stamp) is int and stamp == int(times[i,j]), "Original source timestamp differs from slot")
            if masks[i,j,0]:
                require(stamp is not None and stamp + 7200 <= int(origins[i])
                        and admit_usgs_quality(result["quality_flags"][i][j], policy_id=PRIMARY_APPROVED)
                        and isinstance(result["source_ids"][i][j], str) and result["source_ids"][i][j],
                        "Observed Q requires explicit approved source provenance and simulated availability")
    return result


def _training_identity(arrays, train):
    return array_identity({name: np.asarray(arrays[name])[list(train)] for field in INPUTS + ("target",) for name in (field, field + "_mask")})


def _runtime(torch):
    return dict(torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
                default_dtype=str(torch.get_default_dtype()), deterministic_algorithms=torch.are_deterministic_algorithms_enabled())


def _model(spec):
    from .models.directflow import DirectFlowModel, ModelConfig
    return DirectFlowModel(ModelConfig("LSTM", H4_FEATURES, FEATURES["weather"], FEATURES["state"], 168, tuple(range(1,19)), width=64))


def _state_valid(model, state, torch):
    expected = model.state_dict()
    require(isinstance(state, Mapping) and set(state) == set(expected), "Pilot model state keys differ")
    for key, value in state.items():
        require(torch.is_tensor(value) and value.shape == expected[key].shape and value.dtype == expected[key].dtype
                and bool(torch.isfinite(value).all()), "Invalid pilot model state: " + key)
        if key in ("history_position", "horizon_position"):
            require(torch.equal(value, expected[key]), "Fixed pilot positional coordinates changed")


def _prepare(arrays, provenance, scaler, source, *, scenario_id, epoch, mask_seed, fill_policy, recipe, ridge_model):
    from .pilot_inputs import prepare_h4_inputs
    return prepare_h4_inputs(arrays, history_provenance=provenance, saved_h2_transform=scaler,
        inference_source_identity=source, scenario_id=scenario_id, epoch=epoch, mask_seed=mask_seed,
        fill_policy=fill_policy, ridge_model=ridge_model, augmentation_recipe=recipe,
        expected_ridge_identity=None if ridge_model is None else ridge_model.identity)


def _check_provenance_view(provenance, arrays, source, cohort):
    return provenance.checked_native(original_mask=np.asarray(arrays["history_mask"])[...,0],
        times=arrays["history_time"],origins=arrays["origin_time"],forecast_times=arrays["target_time"],
        site_ids=arrays["site_id"],case_ids=arrays["case_id"],source_identity=source,cohort_digest=cohort)


def _inference_provenance_view(owner, arrays, raw_provenance, source, cohort):
    """At most one owned batch snapshot; actual content must match on reuse.

    The caller's protected/gauge/outer-coordinate gates run first. Only original
    metadata and coordinate/mask content are hashed here; Q/target values are
    neither read nor cached. Dynamic fills and input validation still run.
    """
    if type(raw_provenance) is ValidatedProvenanceBatch:
        _check_provenance_view(raw_provenance,arrays,source,cohort)
        owner._provenance_explicit_view_uses = getattr(owner,"_provenance_explicit_view_uses",0)+1
        return raw_provenance
    metadata_sha = _canonical_digest(raw_provenance)
    coordinates_sha = array_identity({name:np.asarray(arrays[name]) for name in
        COORDINATES+("history_mask","feature_schema","cohort_admission_digest")})
    key = (source,cohort,metadata_sha,coordinates_sha)
    cached = getattr(owner,"_provenance_cache",None)
    if cached is None or cached[0] != key:
        owner._provenance_cache_misses = getattr(owner,"_provenance_cache_misses",0)+1
        snapshot = admit_history_provenance(arrays,raw_provenance,source_identity=source,cohort_digest=cohort)
        require(snapshot.provenance_sha256 == metadata_sha,"Owned provenance differs from hashed input metadata")
        owner._provenance_cache = (key,snapshot)
        owner._provenance_cache_admissions = getattr(owner,"_provenance_cache_admissions",0)+1
    else:
        snapshot = cached[1]
        owner._provenance_cache_hits = getattr(owner,"_provenance_cache_hits",0)+1
    view = snapshot.select(tuple(range(len(arrays["origin_time"]))))
    _check_provenance_view(view,arrays,source,cohort)
    return view


class PilotTrainer:
    def __init__(self, arrays, *, train_indices, inner_indices, role_time_bounds, refit_at_utc,
                 spec, source_identity, cohort_digest, history_provenance, fill_policy,
                 augmentation_recipe, saved_h2_transform=None, ridge_model=None):
        require(isinstance(spec, PilotSpec) and _sha(source_identity) and _sha(cohort_digest), "Explicit pilot spec/source/cohort required")
        self.train, self.validation, self.roles = validate_pilot_roles(arrays, train_indices=train_indices, inner_indices=inner_indices,
            role_time_bounds=role_time_bounds, refit_at_utc=refit_at_utc)
        require(np.asarray(arrays["feature_schema"]).shape == () and np.asarray(arrays["feature_schema"]).item() == FULL_FEATURE_SCHEMA
                and np.asarray(arrays["cohort_admission_digest"]).shape == () and np.asarray(arrays["cohort_admission_digest"]).item() == cohort_digest,
                "Original full H2 source and comparison cohort required")
        self.history_provenance = admit_history_provenance(arrays,history_provenance,
            source_identity=source_identity,cohort_digest=cohort_digest)
        validate_development_bundle(arrays,168,feature_schema=FULL_FEATURE_SCHEMA)
        selected = list(self.train + self.validation)
        require(all(np.asarray(arrays[name])[selected].all() for name in ("history_mask", "weather_mask", "state_mask", "target_mask")),
                "Pilot training/inner roles require complete original inputs and all18 approved labels")
        require(np.all(np.asarray(arrays["history"])[selected,:,0] >= 0) and np.all(np.asarray(arrays["target"])[selected] >= 0),
                "Original admitted pilot Q/labels must be nonnegative")
        self.arrays = MappingProxyType({key: np.array(value, copy=True) for key,value in arrays.items()})
        for value in self.arrays.values(): value.setflags(write=False)
        self.spec, self.source_identity, self.cohort_admission_digest = spec, source_identity, cohort_digest
        self.gauge_id = GAUGE_IDS[int(self.arrays["site_id"][0])]
        self.array_sha256, self.provenance_sha256 = array_identity(self.arrays), self.history_provenance.provenance_sha256
        from .pilot_inputs import validate_fill_policy, validate_augmentation_recipe, H4_HISTORY_FEATURES
        require(H4_HISTORY_FEATURES == H4_FEATURES, "Pilot adapter/model H4 feature order differs")
        self.fill_policy = _canonical(validate_fill_policy(_canonical(fill_policy)))
        self.augmentation_recipe = _canonical(validate_augmentation_recipe(_canonical(augmentation_recipe)))
        require(self.fill_policy.get("trailing_method") == "carry", "Training recipe uses the declared carry fallback only")
        evidence = self.augmentation_recipe.get("evidence", {})
        require(evidence.get("scope") == "GRADIENT_ONLY" and evidence.get("source_identity") == source_identity
                and evidence.get("gradient_case_ids_sha256") == self.roles["train_case_ids_sha256"],
                "Augmentation recipe must bind original gradient-only evidence")
        require(all("ridge" not in key for key in self.augmentation_recipe.get("mixture", {})), "No learned-ridge training augmentation in this pilot")
        self.policy_identity = digest(dict(fill_policy=self.fill_policy, augmentation_recipe=self.augmentation_recipe))
        self.ridge_model, self.ridge_identity = ridge_model, _ridge_identity(ridge_model)
        self.ridge_state_sha256 = _ridge_state_sha256(ridge_model)
        if saved_h2_transform is None:
            self.scaler = TrainOnlyTransform.fit(self.arrays, self.train, feature_schema=FULL_FEATURE_SCHEMA)
        else:
            state = asdict(saved_h2_transform) if isinstance(saved_h2_transform,TrainOnlyTransform) else saved_h2_transform
            self.scaler = TrainOnlyTransform.from_dict(_canonical(state))
        require(self.scaler.feature_schema == FULL_FEATURE_SCHEMA
                and self.scaler.training_case_ids_sha256 == self.roles["train_case_ids_sha256"]
                and self.scaler.training_values_sha256 == _training_identity(self.arrays,self.train),
                "Saved H2 transform differs from original gradient rows/values")
        self.transform_sha256 = digest(asdict(self.scaler))
        self.sampler = EpochSampler(self.train,seed=spec.seed,shuffle=spec.shuffle_training)
        self.updates = self.updates_entered = 0
        self.update_records, self.epoch_records = [], []
        self.stopping, self.best_model_state = None, None
        import torch
        self.torch, self.runtime = torch, _runtime(torch)
        require(torch.get_default_dtype() == torch.float32, "Pilot requires explicit FP32 default dtype")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(spec.seed)
            self.model = _model(spec)
            self.cpu_rng = torch.get_rng_state().clone()
        self.optimizer = None

    def _view(self, indices):
        require(self.history_provenance.provenance_sha256 == self.provenance_sha256,"Owned source provenance changed")
        provenance = self.history_provenance.select(indices)
        result = {name:self.arrays[name][list(indices)] for name in COORDINATES+("history_mask",)}
        _check_provenance_view(provenance,result,self.source_identity,self.cohort_admission_digest)
        result.update({name:self.arrays[name][list(indices)] for name in MODEL_KEYS})
        for name in ("feature_schema", "cohort_admission_digest"): result[name] = self.arrays[name]
        return result, provenance

    def preview_batch(self, indices, *, epoch, augment=False):
        require(digest(dict(fill_policy=self.fill_policy,augmentation_recipe=self.augmentation_recipe)) == self.policy_identity
                and digest(asdict(self.scaler)) == self.transform_sha256, "Pilot policy/transform mutated after construction")
        indices = tuple(indices)
        require(indices and len(set(indices)) == len(indices) and len(indices) <= self.spec.batch_size
                and all(type(i) is int and 0 <= i < len(self.arrays["case_id"]) for i in indices), "Bounded explicit pilot batch required")
        require(type(epoch) is int and epoch >= 0 and type(augment) is bool, "Explicit epoch and augmentation choice required")
        if augment: require(set(indices) <= set(self.train) and self.spec.mode == "augmented", "Only gradient rows in augmented arm may be corrupted")
        arrays, provenance = self._view(indices)
        prepared = _prepare(arrays, provenance, self.scaler, self.source_identity,
            scenario_id="augmentation" if augment else "clean", epoch=epoch, mask_seed=self.spec.mask_seed,
            fill_policy=self.fill_policy, recipe=self.augmentation_recipe if augment else None, ridge_model=None)
        require(np.asarray(prepared["case_admissible"]).shape == (len(indices),) and np.asarray(prepared["case_admissible"]).all(),
                "Pilot batch contains failed/unsupported history fills; no dropping or forwarding missing Q")
        transformed = prepared["transformed_inputs"]
        batch = {name: np.asarray(transformed[name]) for name in MODEL_KEYS}
        for name in MODEL_KEYS:
            require(np.isfinite(batch[name]).all(), "Nonfinite transformed pilot input")
        target, target_mask = self.arrays["target"][list(indices)], self.arrays["target_mask"][list(indices)]
        p = self.scaler.parameters["target"][0]
        batch["target"] = ((target-p["mean"])/p["scale"]).astype(np.float32)
        batch["target_mask"] = np.array(target_mask,copy=True)
        mask_sha = array_identity({name: np.asarray(prepared["raw_inputs"][name]) for name in ("history", "history_mask")})
        receipt = dict(indices=list(indices), case_ids=self.arrays["case_id"][list(indices)].tolist(), epoch=epoch,
            augmented=augment, mask_digest=mask_sha, adapter_receipt_sha256=digest(prepared["receipt"]),
            provenance_sha256=_canonical_digest(prepared["provenance"]), batch_identity=array_identity(batch))
        return batch, receipt

    def preview_next_batch(self):
        sampler = EpochSampler.restore(self.sampler.state())
        indices = sampler.next(self.spec.batch_size)
        return self.preview_batch(indices,epoch=sampler.epoch,augment=self.spec.mode == "augmented")

    def update(self):
        require(self.updates_entered == self.updates and self.updates < self.spec.max_updates, "Pilot update allowance exhausted or previous update incomplete")
        if self.stopping is not None:
            require(not self.stopping.finished, "Pilot stopping allowance exhausted")
            require(not (self.sampler.epoch == self.stopping.completed_epochs and self.sampler.cursor == len(self.train)),
                    "Finish the current epoch before another update")
        next_sampler = EpochSampler.restore(self.sampler.state())
        indices = next_sampler.next(self.spec.batch_size)
        arrays, receipt = self.preview_batch(indices,epoch=next_sampler.epoch,augment=self.spec.mode == "augmented")
        batch = {name:self.torch.from_numpy(np.array(value,copy=True)) for name,value in arrays.items()}
        from .models.directflow import masked_mse
        if self.optimizer is None:
            self.optimizer = self.torch.optim.AdamW(self.model.parameters(),lr=self.spec.learning_rate,foreach=False)
        self.sampler = next_sampler
        self.updates_entered += 1
        with self.torch.random.fork_rng(devices=[]):
            self.torch.set_rng_state(self.cpu_rng)
            self.model.train(); self.optimizer.zero_grad(set_to_none=True)
            prediction = self.model(*(batch[name] for name in MODEL_KEYS))
            loss = masked_mse(prediction,batch["target"],batch["target_mask"])
            loss.backward()
            require(all(p.grad is not None and bool(self.torch.isfinite(p.grad).all()) for p in self.model.parameters()), "Invalid pilot gradient")
            self.optimizer.step()
            require(all(bool(self.torch.isfinite(p).all()) for p in self.model.parameters()), "Invalid pilot parameter after update")
            self.cpu_rng = self.torch.get_rng_state().clone()
        self.updates += 1
        record = {**receipt,"loss":float(loss.detach()),"updates":self.updates,"updates_entered":self.updates_entered}
        self.update_records.append({key:value for key,value in record.items() if key not in ("indices", "case_ids")})
        return record

    def stopping_loss(self):
        self.model.eval(); total, count = 0., 0
        with self.torch.inference_mode():
            for offset in range(0,len(self.validation),self.spec.batch_size):
                arrays,_ = self.preview_batch(self.validation[offset:offset+self.spec.batch_size],epoch=0,augment=False)
                batch = {name:self.torch.from_numpy(np.array(value,copy=True)) for name,value in arrays.items()}
                prediction = self.model(*(batch[name] for name in MODEL_KEYS))
                errors = prediction - batch["target"]
                require(bool(self.torch.isfinite(errors).all()), "Nonfinite clean-inner error")
                total += float(errors.square().sum()); count += errors.numel()
        require(count > 0, "Empty pilot clean-inner support")
        return total/count

    def finish_epoch(self):
        require(self.stopping is not None and not self.stopping.finished
                and self.sampler.epoch == self.stopping.completed_epochs and self.sampler.cursor == len(self.train),
                "A complete gradient epoch and active stopping policy are required")
        from .engineering import _cpu_tree
        loss = self.stopping_loss(); improved = self.stopping.observe(loss)
        if improved: self.best_model_state = _cpu_tree(self.model.state_dict(),self.torch)
        row = dict(epoch=self.stopping.completed_epochs,inner_stopping_loss=loss,updates=self.updates,improved=improved,
            monitor="CLEAN_INNER_STANDARDIZED_Q_MSE",augmentation_used_for_stopping=False)
        self.epoch_records.append(row)
        return row

    def run_epoch(self, stopping=None):
        if self.stopping is None:
            require(isinstance(stopping,EarlyStopping) and stopping.completed_epochs == 0 and stopping.max_epochs <= 50,
                    "Fresh explicit stopping policy with at most50 epochs required")
            self.stopping = EarlyStopping(**asdict(stopping))
        elif stopping is not None:
            require(asdict(stopping) == asdict(self.stopping), "Resume stopping state/policy differs")
        require(not self.stopping.finished, "Pilot stopping policy completed")
        epoch = self.stopping.completed_epochs
        require(self.sampler.epoch in (max(0,epoch-1),epoch), "Pilot sampler and stopping epoch differ")
        while self.sampler.epoch < epoch or self.sampler.cursor < len(self.train): self.update()
        return self.finish_epoch()

    def fit_epochs(self, stopping):
        while self.stopping is None or not self.stopping.finished:
            self.run_epoch(stopping if self.stopping is None else self.stopping)
        return list(self.epoch_records)

    def checkpoint(self, path):
        require(self.updates_entered == self.updates, "Cannot checkpoint an incompletely entered update")
        require(digest(dict(fill_policy=self.fill_policy,augmentation_recipe=self.augmentation_recipe)) == self.policy_identity
                and digest(asdict(self.scaler)) == self.transform_sha256 and _ridge_identity(self.ridge_model) == self.ridge_identity
                and _ridge_state_sha256(self.ridge_model) == self.ridge_state_sha256, "Pilot checkpoint policies/transform/ridge mutated")
        from .engineering import _cpu_tree
        value = dict(schema=CHECKPOINT_SCHEMA,spec=asdict(self.spec),source_identity=self.source_identity,
            gauge_id=self.gauge_id,cohort_admission_digest=self.cohort_admission_digest,array_sha256=self.array_sha256,provenance_sha256=self.provenance_sha256,
            roles=self.roles,train_indices=list(self.train),inner_indices=list(self.validation),
            fill_policy=self.fill_policy,augmentation_recipe=self.augmentation_recipe,policy_identity=self.policy_identity,
            ridge_identity=self.ridge_identity,ridge_state_sha256=self.ridge_state_sha256,transform=asdict(self.scaler),transform_sha256=self.transform_sha256,
            sampler=self.sampler.state(),cpu_rng=self.cpu_rng,updates=self.updates,updates_entered=self.updates_entered,
            update_records=self.update_records,stopping=asdict(self.stopping) if self.stopping else None,
            epoch_records=self.epoch_records,best_model=self.best_model_state,model=_cpu_tree(self.model.state_dict(),self.torch),
            optimizer=_cpu_tree(self.optimizer.state_dict(),self.torch) if self.optimizer is not None else None,
            runtime=self.runtime,implementation=pilot_implementation_fingerprint(),scientific_execution_authorized=False)
        path = Path(path); stage = path.with_name(path.name + ".writing")
        require(not path.exists() and not stage.exists() and not path.is_symlink() and not stage.is_symlink(), "Fresh pilot checkpoint path required")
        with stage.open("xb") as stream:
            self.torch.save(value,stream); stream.flush(); os.fsync(stream.fileno())
        stage.rename(path); fd = os.open(path.parent,os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @classmethod
    def restore(cls,path,arrays,*,history_provenance,expected_sha256,expected_source_identity,expected_cohort_digest,
                expected_mode,expected_fill_policy,expected_augmentation_recipe,ridge_model=None):
        value,torch = _load_checkpoint(path,expected_sha256=expected_sha256,expected_source_identity=expected_source_identity,
            expected_cohort_digest=expected_cohort_digest,expected_mode=expected_mode,
            expected_fill_policy=expected_fill_policy,expected_augmentation_recipe=expected_augmentation_recipe,ridge_model=ridge_model)
        result = cls(arrays,train_indices=tuple(value["train_indices"]),inner_indices=tuple(value["inner_indices"]),
            role_time_bounds=value["roles"]["role_time_bounds"],refit_at_utc=value["roles"]["refit"]["refit_at_utc"],
            spec=PilotSpec(**value["spec"]),source_identity=expected_source_identity,cohort_digest=expected_cohort_digest,
            history_provenance=history_provenance,fill_policy=expected_fill_policy,augmentation_recipe=expected_augmentation_recipe,
            saved_h2_transform=value["transform"],ridge_model=ridge_model)
        require(result.array_sha256 == value["array_sha256"] and result.provenance_sha256 == value["provenance_sha256"]
                and result.roles == value["roles"] and result.runtime == value["runtime"] and result.gauge_id == value["gauge_id"], "Pilot data/provenance/roles/runtime/gauge changed")
        sampler = EpochSampler.restore(value["sampler"])
        require(sampler.indices == result.train and sampler.seed == result.spec.seed and sampler.shuffle == result.spec.shuffle_training,
                "Pilot sampler identity differs")
        _state_valid(result.model,value["model"],torch); result.model.load_state_dict(value["model"])
        if value["optimizer"] is not None:
            result.optimizer = torch.optim.AdamW(result.model.parameters(),lr=result.spec.learning_rate,foreach=False)
            result.optimizer.load_state_dict(value["optimizer"])
        result.sampler, result.cpu_rng = sampler, value["cpu_rng"].clone()
        result.updates = result.updates_entered = value["updates"]
        result.stopping = EarlyStopping(**value["stopping"]) if value["stopping"] else None
        result.best_model_state, result.epoch_records, result.update_records = value["best_model"], value["epoch_records"], value["update_records"]
        if result.best_model_state is not None: _state_valid(result.model,result.best_model_state,torch)
        return result


def _load_checkpoint(path,*,expected_sha256,expected_source_identity,expected_cohort_digest,expected_mode,
                     expected_fill_policy,expected_augmentation_recipe,ridge_model=None):
    import torch
    from .engineering import _local_bytes
    require(all(_sha(x) for x in (expected_sha256,expected_source_identity,expected_cohort_digest)), "Explicit pilot checkpoint/source/cohort identities required")
    raw = _local_bytes(path,64*1024**2)
    require(hashlib.sha256(raw).hexdigest() == expected_sha256, "Pilot checkpoint bytes differ")
    value = torch.load(io.BytesIO(raw),map_location="cpu",weights_only=True)
    require(value.get("schema") == CHECKPOINT_SCHEMA and value.get("source_identity") == expected_source_identity
            and value.get("cohort_admission_digest") == expected_cohort_digest and value.get("spec",{}).get("mode") == expected_mode,
            "Pilot checkpoint schema/source/cohort/mode differs")
    require(value.get("gauge_id") in GAUGE_IDS, "Pilot checkpoint training gauge missing")
    spec = PilotSpec(**value["spec"])
    require(value["fill_policy"] == _canonical(expected_fill_policy) and value["augmentation_recipe"] == _canonical(expected_augmentation_recipe)
            and value["policy_identity"] == digest(dict(fill_policy=value["fill_policy"],augmentation_recipe=value["augmentation_recipe"]))
            and value["ridge_identity"] == _ridge_identity(ridge_model) and value["ridge_state_sha256"] == _ridge_state_sha256(ridge_model),
            "Pilot fill/augmentation/ridge policy changed")
    require(value["implementation"] == pilot_implementation_fingerprint(), "Pilot code/dependencies changed")
    require(type(value["updates"]) is int and 0 <= value["updates"] <= spec.max_updates and value["updates_entered"] == value["updates"]
            and len(value["update_records"]) == value["updates"], "Pilot update accounting differs")
    require((value["optimizer"] is None) == (value["updates"] == 0), "Pilot optimizer/update state differs")
    sampler = EpochSampler.restore(value["sampler"])
    require(sampler.indices == tuple(value["train_indices"]) and sampler.seed == spec.seed and sampler.shuffle == spec.shuffle_training,
            "Pilot checkpoint sampler identity differs")
    batches = (len(sampler.indices)+spec.batch_size-1)//spec.batch_size
    require((sampler.cursor == len(sampler.indices) or sampler.cursor % spec.batch_size == 0)
            and value["updates"] == sampler.epoch*batches+(sampler.cursor+spec.batch_size-1)//spec.batch_size,
            "Pilot sampler/update accounting differs")
    for i,record in enumerate(value["update_records"],1):
        require(record.get("updates") == record.get("updates_entered") == i and record.get("epoch") == (i-1)//batches
                and isinstance(record.get("loss"),(int,float)) and np.isfinite(record["loss"]) and record["loss"] >= 0
                and all(_sha(record.get(name)) for name in ("mask_digest","adapter_receipt_sha256","provenance_sha256","batch_identity")),
                "Pilot update journal differs")
    if value["stopping"] is not None:
        stopping = EarlyStopping(**value["stopping"])
        require(stopping.max_epochs <= 50 and len(value["epoch_records"]) == stopping.completed_epochs and stopping.completed_epochs <= value["updates"]//batches,
                "Pilot stopping/update accounting differs")
        if stopping.completed_epochs:
            from .inference import selected_best_epoch
            selected_best_epoch(value)
    else:
        require(not value["epoch_records"] and value["best_model"] is None, "Pilot best state has no stopping selection")
    require(torch.is_tensor(value["cpu_rng"]) and value["cpu_rng"].dtype == torch.uint8 and value["cpu_rng"].ndim == 1
            and value["cpu_rng"].shape == torch.get_rng_state().shape,
            "Pilot CPU RNG state required")
    require(value["transform_sha256"] == digest(value["transform"]), "Pilot saved H2 transform identity differs")
    TrainOnlyTransform.from_dict(value["transform"])
    return value,torch


class PilotBestPredictor:
    def provenance_cache_info(self):
        """Internal diagnostics only; never part of canonical scientific receipts."""
        return dict(hits=getattr(self,"_provenance_cache_hits",0),misses=getattr(self,"_provenance_cache_misses",0),
            entries=int(getattr(self,"_provenance_cache",None) is not None),
            admissions=getattr(self,"_provenance_cache_admissions",0),
            explicit_views=getattr(self,"_provenance_explicit_view_uses",0))

    @classmethod
    def load(cls,path,*,expected_sha256,expected_source_identity,expected_cohort_digest,expected_mode,
             expected_fill_policy,expected_augmentation_recipe,gradient_fit_case_ids,expected_gauge_id,ridge_model=None):
        value,torch = _load_checkpoint(path,expected_sha256=expected_sha256,expected_source_identity=expected_source_identity,
            expected_cohort_digest=expected_cohort_digest,expected_mode=expected_mode,expected_fill_policy=expected_fill_policy,
            expected_augmentation_recipe=expected_augmentation_recipe,ridge_model=ridge_model)
        from .inference import selected_best_epoch
        best_epoch = selected_best_epoch(value)
        scaler = TrainOnlyTransform.from_dict(value["transform"])
        require(expected_gauge_id in GAUGE_IDS and value["gauge_id"] == expected_gauge_id
                and type(gradient_fit_case_ids) is tuple and gradient_fit_case_ids
                and all(isinstance(case,str) and re.fullmatch(expected_gauge_id+r":\d{4}-\d{2}-\d{2}T\d{2}:00:00Z",case) for case in gradient_fit_case_ids)
                and digest(list(gradient_fit_case_ids)) == scaler.training_case_ids_sha256 == value["roles"]["train_case_ids_sha256"],
                "Pilot gradient membership/gauge differs from saved transform")
        result = cls(); result.torch, result.scaler = torch, scaler
        result.spec = PilotSpec(**value["spec"])
        result.gauge_id = expected_gauge_id
        result.role_bounds = _roles(value["roles"]["role_time_bounds"])
        require("outer" in result.role_bounds, "Pilot prediction needs its explicit outer-development interval")
        with torch.random.fork_rng(devices=[]): result.model = _model(result.spec)
        _state_valid(result.model,value["best_model"],torch); result.model.load_state_dict(value["best_model"]); result.model.eval()
        result.fill_policy, result.recipe, result.ridge_model = value["fill_policy"], value["augmentation_recipe"], ridge_model
        result._provenance_cache = None
        result._provenance_cache_hits = result._provenance_cache_misses = result._provenance_cache_admissions = 0
        result._provenance_explicit_view_uses = 0
        result.receipt = dict(checkpoint_sha256=expected_sha256,training_source_identity=expected_source_identity,
            training_cohort_admission_digest=expected_cohort_digest,selected_state="best_model",selected_best_epoch=best_epoch,
            completed_epochs=value["stopping"]["completed_epochs"],feature_schema=H4_FEATURE_SCHEMA,mode=result.spec.mode,
            architecture="LSTM",device="cpu",gauge_id=result.gauge_id,seed=result.spec.seed,history_hours=168,transform_sha256=value["transform_sha256"],
            training_case_ids_sha256=scaler.training_case_ids_sha256,policy_identity=value["policy_identity"],
            prediction_units="m3/s",last_state_used=False,optimizer_constructed=False,scaler_fitted=False)
        return result

    def predict(self,arrays,*,history_provenance,expected_inference_source_identity,expected_inference_cohort_digest,
                scenario_id="clean",mask_seed=1729,epoch=0):
        require(digest(asdict(self.scaler)) == self.receipt["transform_sha256"]
                and digest(dict(fill_policy=self.fill_policy,augmentation_recipe=self.recipe)) == self.receipt["policy_identity"],
                "Pilot predictor policies/transform mutated after load")
        require(_sha(expected_inference_source_identity) and _sha(expected_inference_cohort_digest), "Explicit inference source/cohort required")
        keys = set(iter(arrays)); future = "forecast_time" if "forecast_time" in keys else "target_time"
        view = {name:np.asarray(arrays[future if name == "target_time" else name]) for name in COORDINATES}
        n = validate_coordinates(view,168)
        require(n <= 64 and np.all((np.asarray(view["origin_time"]) >= 169*3600) & (np.asarray(view["origin_time"]) < ASSESSMENT_START))
                and np.all(np.diff(np.asarray(view["origin_time"],dtype=np.int64)) > 0), "Bounded chronological pilot inference required")
        require(np.all(view["site_id"] == GAUGE_IDS.index(self.gauge_id))
                and view["case_id"].tolist() == [self.gauge_id+":"+datetime.fromtimestamp(int(t),timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") for t in view["origin_time"]],
                "Pilot inference gauge/case identity differs from training gauge")
        start,end = self.role_bounds["outer"]
        require(np.all((view["origin_time"] >= start) & (view["origin_time"] < end)) and np.all(view["target_time"] < end),
                "Pilot inference clocks cross the checkpoint's explicit outer-development interval")
        for name in ("feature_schema","cohort_admission_digest"):view[name]=np.asarray(arrays[name])
        require(view["feature_schema"].item() == FULL_FEATURE_SCHEMA and view["cohort_admission_digest"].item() == expected_inference_cohort_digest,
                "Original inference H2 source/cohort differs")
        # Target values and masks are intentionally never fetched.
        for name in MODEL_KEYS:view[name]=np.asarray(arrays[name])
        provenance = _inference_provenance_view(self,view,history_provenance,
            expected_inference_source_identity,expected_inference_cohort_digest)
        prepared = _prepare(view,provenance,self.scaler,expected_inference_source_identity,scenario_id=scenario_id,
            epoch=epoch,mask_seed=mask_seed,fill_policy=self.fill_policy,recipe=None,ridge_model=self.ridge_model)
        require(np.asarray(prepared["case_admissible"]).shape == (n,) and np.asarray(prepared["case_admissible"]).all(),
                "Pilot inference has failed/unsupported fills; no silent filtering or forwarding")
        mapped = prepared["transformed_inputs"]
        with self.torch.inference_mode():
            prediction = self.model(*(self.torch.from_numpy(np.array(mapped[name],copy=True)) for name in MODEL_KEYS))
        physical = self.scaler.inverse_target(prediction.numpy().astype(np.float64))
        require(physical.shape == (n,18) and np.isfinite(physical).all(), "Nonfinite physical pilot predictions")
        physical.setflags(write=False)
        return dict(prediction_q=physical,case_id=view["case_id"].copy(),origin_time=view["origin_time"].copy(),
            history_time=view["history_time"].copy(),forecast_time=view["target_time"].copy(),site_id=view["site_id"].copy(),
            receipt={**self.receipt,"inference_source_identity":expected_inference_source_identity,
                "inference_cohort_admission_digest":expected_inference_cohort_digest,"input_case_ids_sha256":digest(view["case_id"].tolist()),
                "input_identity":array_identity(view),"scenario_id":scenario_id,"mask_seed":mask_seed,"epoch":epoch,
                "adapter_receipt_sha256":digest(prepared["receipt"]),"provenance_sha256":_canonical_digest(prepared["provenance"]),
                "case_count":n,"target_values_read":False,"target_masks_read":False,"optimizer_updates":0,
                "posthoc_clipping":False,"scientific_scoring_performed":False})
