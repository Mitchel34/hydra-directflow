"""A proposed immutable spatial reduction contract, with no real geometry.

The acquired rectangle is not a basin average. Real weights remain unavailable
until separately reconciled geometry and cell intersections are prepared.
"""
from dataclasses import dataclass
import hashlib
import json
import math


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exact_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


@dataclass(frozen=True)
class SpatialWeights:
    site: str
    grid_sha256: str
    geometry_sha256: str
    cell_indices: tuple[int, ...]
    weights: tuple[float, ...]
    grid_cell_count: int
    method: str

    def __post_init__(self):
        require(type(self.cell_indices) is tuple and type(self.weights) is tuple, "Immutable tuple weights required")
        require(bool(self.site) and exact_sha(self.grid_sha256) and exact_sha(self.geometry_sha256),
                "Gauge and immutable grid/geometry hashes are required")
        require(type(self.grid_cell_count) is int and self.grid_cell_count > 0 and bool(self.method),
                "Explicit source grid size and weight method are required")
        require(self.cell_indices and len(self.cell_indices) == len(self.weights)
                and len(set(self.cell_indices)) == len(self.cell_indices), "Unique weighted cell identities required")
        require(all(type(i) is int and 0 <= i < self.grid_cell_count for i in self.cell_indices),
                "Spatial cells must lie inside the bound source grid")
        require(all(type(w) in (int, float) and math.isfinite(w) and w >= 0 for w in self.weights)
                and math.isclose(math.fsum(self.weights), 1.0, rel_tol=0, abs_tol=1e-10),
                "Nonnegative source-cell weights must sum to one")

    def digest(self):
        value = dict(site=self.site, grid_sha256=self.grid_sha256, geometry_sha256=self.geometry_sha256,
                     cell_indices=self.cell_indices, weights=self.weights, grid_cell_count=self.grid_cell_count,
                     method=self.method)
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def weighted_mean(values, masks, weights, *, grid_sha256, geometry_sha256,
                  min_coverage, missing_policy):
    """Partial coverage never silently becomes a full-basin mean.

    ``mask`` rejects a partial mean. ``explicit_renormalize`` is an optional
    declared policy, recorded in the result with the original coverage intact.
    Neither policy is selected for real data by this reference implementation.
    """
    require(grid_sha256 == weights.grid_sha256 and geometry_sha256 == weights.geometry_sha256,
            "Weights cannot transfer to a different grid or basin geometry")
    require(type(min_coverage) in (int, float) and math.isfinite(min_coverage) and 0 < min_coverage <= 1,
            "An explicit coverage threshold in (0,1] is required")
    require(missing_policy in ("mask", "explicit_renormalize"), "Explicit missing-cell policy required")
    observed, total = [], []
    for index, weight in zip(weights.cell_indices, weights.weights):
        present = masks.get(index, False)
        require(type(present) is bool, "Cell masks must be booleans")
        if present:
            value = values.get(index)
            require(type(value) in (int, float) and math.isfinite(value), "Admitted cell value must be finite")
            observed.append(weight)
            total.append(weight * value)
    coverage, partial_sum = math.fsum(observed), math.fsum(total)
    full = math.isclose(coverage, 1.0, rel_tol=0, abs_tol=1e-10)
    passes = coverage > 0 and coverage + 1e-12 >= min_coverage and (full or missing_policy == "explicit_renormalize")
    renormalized = bool(passes and not full)
    return dict(value=(partial_sum / coverage if renormalized else partial_sum) if passes else None,
                mask=bool(passes), coverage=coverage, renormalized=renormalized,
                observed_weighted_sum=partial_sum, weights_digest=weights.digest(),
                reason=None if passes else "MISSING_CELL_COVERAGE_NOT_ADMITTED")
