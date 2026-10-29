"""Dependency-free tensor and readiness contracts for a proposed HRRR study.

No dataset reader, model dependency, fit command or training admission lives here.
Dimensions are supplied by configuration, rather than inherited from old studies.
"""
from dataclasses import dataclass


GAUGE_IDS = ("03161000", "03164000", "03479000")
HORIZONS_HOURS = tuple(range(1, 19))
ARCHITECTURES = ("LSTM", "Transformer", "GRU_Transformer", "Mamba2_Transformer")
CORE_WEATHER_FEATURES = ("precip_mm", "temp_k", "specific_humidity", "swe_kg_m2")


@dataclass(frozen=True)
class DevelopmentProtocol:
    """Executable development candidate; only gauge/H18/model scope is fixed.

    The two-hour lag is a retrospective scenario, not a publication guarantee.
    L24 is the bounded smoke default; L168 remains a candidate to compare.
    """

    history_hours: int = 24
    history_lag_hours: int = 2
    weather_lag_hours: int = 2
    soil_lag_hours: int = 2
    max_soil_age_hours: int = 8
    allow_older_extended_cycles: bool = True
    protected_assessment_scoring_enabled: bool = False
    gauge_ids: tuple[str, ...] = GAUGE_IDS
    horizons_hours: tuple[int, ...] = HORIZONS_HOURS
    architectures: tuple[str, ...] = ARCHITECTURES

    def __post_init__(self):
        require(type(self.history_hours) is int and self.history_hours in (24, 168),
                "Development lookback must be the explicit L24 or L168 candidate")
        for value in (self.history_lag_hours, self.weather_lag_hours,
                      self.soil_lag_hours, self.max_soil_age_hours):
            require(type(value) is int and value >= 0, "Explicit nonnegative whole-hour lags/age required")
        require(self.history_lag_hours >= 1, "History must end strictly before C")
        require(type(self.allow_older_extended_cycles) is bool,
                "Older extended-cycle permission must be explicit")
        require(self.protected_assessment_scoring_enabled is False,
                "Development configuration cannot enable protected assessment scoring")
        require(self.gauge_ids == GAUGE_IDS, "The confirmed three-gauge scope must be preserved")
        require(self.horizons_hours == HORIZONS_HOURS, "The confirmed hourly H18 scope must be preserved")
        require(self.architectures == ARCHITECTURES, "All four architecture candidates, including Mamba2, are required")


def require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class TensorSpec:
    history_hours: int
    horizons_hours: tuple[int, ...]
    history_features: tuple[str, ...]
    weather_features: tuple[str, ...]
    state_features: tuple[str, ...]
    gauge_ids: tuple[str, ...]

    def __post_init__(self):
        require(all(type(value) is tuple for value in (self.horizons_hours, self.history_features,
                self.weather_features, self.state_features, self.gauge_ids)), "Immutable tuple configuration required")
        require(type(self.history_hours) is int and self.history_hours > 0,
                "A positive configured history length is required")
        require(self.horizons_hours and all(type(h) is int and h > 0 for h in self.horizons_hours),
                "Positive explicit horizon offsets are required")
        require(tuple(sorted(set(self.horizons_hours))) == self.horizons_hours,
                "Horizon offsets must be unique and increasing")
        require(self.gauge_ids and all(isinstance(site, str) and site for site in self.gauge_ids)
                and len(set(self.gauge_ids)) == len(self.gauge_ids),
                "Explicit unique gauge identifiers are required")
        for names in (self.history_features, self.weather_features, self.state_features):
            require(all(isinstance(name, str) and name for name in names)
                    and len(names) == len(set(names)), "Feature names must be explicit and unique")
            require(not any(word in name.lower() for name in names
                            for word in ("nwm", "era5", "radiation", "dswrf")),
                    "NWM, ERA5 and radiation channels are outside this HRRR input contract")
        require(self.history_features and self.weather_features, "History and forecast features are required")

    def validate_shapes(self, *, history, history_mask, weather, weather_mask,
                        state, state_mask, site_id):
        """Validate shape tuples only; target tensors are not model inputs."""
        require(len(history) == 3 and type(history[0]) is int and history[0] > 0,
                "History must have shape [batch, lookback, history features]")
        batch = history[0]
        expected_history = (batch, self.history_hours, len(self.history_features))
        expected_weather = (batch, len(self.horizons_hours), len(self.weather_features))
        expected_state = (batch, len(self.state_features))
        require(tuple(history) == tuple(history_mask) == expected_history, "History/mask shape differs")
        require(tuple(weather) == tuple(weather_mask) == expected_weather, "Forecast/mask shape differs")
        require(tuple(state) == tuple(state_mask) == expected_state, "Origin state/mask shape differs")
        require(tuple(site_id) == (batch,), "Site identifiers must have shape [batch]")
        return (batch, len(self.horizons_hours))


def readiness_blockers(proposal):
    """Report unresolved preparation gates; this does not authorize training."""
    blockers = []
    gate = proposal["acquisition_gate"]
    if gate.get("status") != "GENUINE_SEALED_INDEX_BOUND" or not gate.get("qualified_index_ref") or not gate.get("seal_ref"):
        blockers.append("genuine_sealed_qualified_index")
    if proposal["pending_user_confirmation"].get("status") != "CONFIRMED":
        blockers.append("horizon_and_gauge_scope_confirmation")
    blockers.extend(item["id"] for item in proposal["unresolved_decisions"]
                    if item.get("status") != "RESOLVED")
    return tuple(dict.fromkeys(blockers))


def require_preprocessing_ready(proposal):
    """Structural gate only; a future loader must authenticate index/seal bytes."""
    blockers = readiness_blockers(proposal)
    require(not blockers, "Preprocessing remains blocked: " + ", ".join(blockers))
    return True
