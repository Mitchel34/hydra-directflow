"""Synthetic coordinate/causality checks; no APIs, payload reads or training."""
import dataclasses
from datetime import datetime, timezone
import unittest
from unittest.mock import patch

from hydra_directflow.contract import (
    ARCHITECTURES, CORE_WEATHER_FEATURES, GAUGE_IDS, HORIZONS_HOURS,
    DevelopmentProtocol, TensorSpec,
)
from hydra_directflow.reference import (
    ASSESSMENT_END, ASSESSMENT_LAST_TARGET, ASSESSMENT_START, HOUR,
    AvailabilityPolicy, Observation, ReplayClock, ScaleCell, SoilInitialState,
    Split, WeatherRun, assessment_origin_admitted, fit_train_scaler,
    forecast_target_times, history_window, latest_weather_run,
    require_scoring_allowed, soil_state, target_window, target_window_contained,
    weather_support,
)


def utc(text):
    return int(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp())


C = utc("2021-08-23T02:00:00Z")
LAG = AvailabilityPolicy("development-two-hour-simulation", "retrospective_simulated", 2 * HOUR)
APPROVED = lambda flags: bool(flags) and all(flag in ("A", "R") for flag in flags)


def run(reference, lead=48, *, identity=None, missing=()):
    times = tuple(reference + h * HOUR for h in range(1, lead + 1))
    return WeatherRun(identity or str(reference), reference, None, times,
                      tuple((name, tuple((name, valid) not in missing for valid in times))
                            for name in CORE_WEATHER_FEATURES), ("original-provider-flag",))


def select(runs, *, fallback=True, cutoff=C):
    return latest_weather_run(runs, cutoff=cutoff, origin_anchor=cutoff, policy=LAG,
                              horizons_hours=HORIZONS_HOURS,
                              allow_older_extended_cycles=fallback)


def history(rows, length=24):
    return history_window(rows, site=GAUGE_IDS[0], origin_anchor=C, cutoff=C,
                          lookback_hours=length, history_lag_hours=2,
                          policy=LAG, admit_quality=APPROVED)


def observation(valid, value=10.0, flags=("A",), site=GAUGE_IDS[0]):
    return Observation(site, valid, None, value, flags, f"source:{site}:{valid}")


class DevelopmentContractTests(unittest.TestCase):
    def test_only_scope_is_confirmed_and_mamba_is_present(self):
        candidate = DevelopmentProtocol()
        self.assertEqual(candidate.gauge_ids, GAUGE_IDS)
        self.assertEqual(candidate.horizons_hours, tuple(range(1, 19)))
        self.assertEqual(candidate.architectures, ARCHITECTURES)
        self.assertIn("Mamba2_Transformer", candidate.architectures)
        self.assertEqual(candidate.history_hours, 24)
        self.assertEqual(DevelopmentProtocol(history_hours=168).history_hours, 168)
        self.assertFalse(candidate.protected_assessment_scoring_enabled)

    def test_scope_reduction_or_protected_scoring_rejected(self):
        for kwargs in ({"gauge_ids": GAUGE_IDS[:2]}, {"horizons_hours": (1, 2)},
                       {"architectures": ARCHITECTURES[:3]}, {"history_hours": 72},
                       {"protected_assessment_scoring_enabled": True},
                       {"history_lag_hours": 0}, {"weather_lag_hours": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                DevelopmentProtocol(**kwargs)

    def test_no_nwm_predictor_in_tensor_contract(self):
        with self.assertRaisesRegex(ValueError, "NWM"):
            TensorSpec(24, HORIZONS_HOURS, ("q_cms",), ("nwm_forecast",), (), GAUGE_IDS)

    def test_actual_artifact_creation_does_not_become_replay_time(self):
        actual = utc("2026-10-28T20:00:00Z")
        with patch("hydra_directflow.reference.time.time", return_value=actual):
            clock = ReplayClock(C, C + 30)
        self.assertEqual(clock.artifact_created_at, actual)
        self.assertEqual(clock.replay_cutoff, C)
        self.assertEqual(clock.replay_completed_at, C + 30)
        self.assertNotEqual(clock.artifact_created_at, clock.replay_cutoff)
        with self.assertRaises(ValueError):
            ReplayClock(C, C - 1)


class WeatherSelectionTests(unittest.TestCase):
    def test_latest_complete_cycle_beats_newer_incomplete_cycle(self):
        old = run(C - 8 * HOUR)
        newest = run(C - 2 * HOUR, lead=18)
        self.assertIs(select((old, newest)), old)
        self.assertEqual(select((old, newest)).source_quality_flags, ("original-provider-flag",))
        self.assertIsNone(select((old, newest), fallback=False))

    def test_newest_complete_cycle_wins(self):
        old, latest = run(C - 8 * HOUR), run(C - 2 * HOUR)
        self.assertIs(select((old, latest)), latest)

    def test_no_cross_cycle_stitching(self):
        target1, target2 = C + HOUR, C + 2 * HOUR
        left = run(C - 8 * HOUR, missing=((CORE_WEATHER_FEATURES[0], target1),))
        right = run(C - 2 * HOUR, missing=((CORE_WEATHER_FEATURES[1], target2),))
        self.assertIsNone(select((left, right)))

    def test_coordinates_without_feature_masks_do_not_qualify(self):
        candidate = run(C - 2 * HOUR)
        self.assertIsNone(select((dataclasses.replace(candidate, feature_support=()),)))
        self.assertIsNone(select((dataclasses.replace(candidate, feature_support=candidate.feature_support[:-1]),)))

    def test_unavailable_newer_and_future_cycles_do_not_change_selection(self):
        eligible = run(C - 2 * HOUR)
        expected = select((eligible,))
        self.assertIs(select((eligible, run(C - HOUR), run(C + 10 * HOUR))), expected)

    def test_changes_after_c_do_not_affect_weather_at_c(self):
        eligible = run(C - 2 * HOUR)
        future = run(C + HOUR)
        damaged_future = dataclasses.replace(future, feature_support=())
        self.assertEqual(select((eligible, future)), select((eligible, damaged_future)))

    def test_duplicate_identity_or_vintage_is_not_silently_selected(self):
        candidate = run(C - 2 * HOUR)
        for other in (candidate, dataclasses.replace(candidate, run_id="second-vintage")):
            with self.subTest(other=other.run_id), self.assertRaises(ValueError):
                select((candidate, other))

    def test_feature_masks_must_be_boolean_and_aligned(self):
        candidate = run(C - 2 * HOUR)
        for mask in ((True,), tuple(1 for _ in candidate.valid_times)):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                dataclasses.replace(candidate, feature_support=(("precip_mm", mask),))

    def test_run_age_source_lead_and_delivery_lead_are_separate(self):
        chosen = run(C - 8 * HOUR)
        support = weather_support(chosen, origin_anchor=C, completion=C + 90,
                                  horizons_hours=HORIZONS_HOURS)
        self.assertEqual(support["run_age_hours"], 8)
        self.assertEqual(support["source_lead_hours"], tuple(range(9, 27)))
        self.assertAlmostEqual(support["effective_lead_hours"][0], 0.975)
        self.assertTrue(all(support["supported"]))
        late = weather_support(chosen, origin_anchor=C, completion=C + HOUR,
                               horizons_hours=HORIZONS_HOURS)
        self.assertFalse(late["supported"][0])

    def test_documented_availability_cannot_use_acquisition_time_as_unknown(self):
        documented = AvailabilityPolicy("documented-source-publication", "documented")
        with self.assertRaises(ValueError):
            documented.available_at(C, None)
        with self.assertRaises(ValueError):
            documented.available_at(C, C - HOUR)
        self.assertEqual(LAG.available_at(C, None), C + 2 * HOUR)


class ExactObservationTests(unittest.TestCase):
    def test_history_ends_at_c_minus_lag_for_both_candidates(self):
        for length in (24, 168):
            expected = tuple(C - h * HOUR for h in range(length + 1, 1, -1))
            result = history([observation(t) for t in expected], length)
            self.assertEqual(result["times"], expected)
            self.assertEqual(result["age_hours"][-1], 2)
            self.assertTrue(all(result["mask"]))
            self.assertEqual(len(result["values"]), length)

    def test_no_searchback_interpolation_or_cross_gauge_fill(self):
        missing = C - 4 * HOUR
        rows = [observation(C - h * HOUR) for h in range(25, 1, -1) if h != 4]
        rows.extend((observation(missing - 60), observation(missing + 60),
                     observation(missing, site=GAUGE_IDS[1]), observation(C - 26 * HOUR)))
        result = history(rows)
        index = result["times"].index(missing)
        self.assertIsNone(result["values"][index])
        self.assertFalse(result["mask"][index])
        self.assertEqual(result["reasons"][index], "MISSING_EXACT_HOUR")
        self.assertEqual(len(result["times"]), 24)

    def test_future_and_lagged_out_values_cannot_change_history(self):
        past = [observation(C - h * HOUR) for h in range(25, 1, -1)]
        noisy = past + [observation(C - HOUR, 9000), observation(C, 9000),
                        observation(C + HOUR, 9000)]
        self.assertEqual(history(past), history(noisy))

    def test_quality_flags_and_source_identity_retained_when_rejected(self):
        row = observation(C - 2 * HOUR, flags=("P", "e", "original:verbatim"))
        result = history((row,))
        self.assertIsNone(result["values"][-1])
        self.assertEqual(result["quality_flags"][-1], row.quality_flags)
        self.assertEqual(result["source_ids"][-1], row.source_id)
        self.assertEqual(result["reasons"][-1], "QUALITY_NOT_ADMITTED")

    def test_documented_late_history_row_stays_missing(self):
        row = dataclasses.replace(observation(C - 2 * HOUR), available_at=C + 1)
        result = history_window((row,), site=GAUGE_IDS[0], origin_anchor=C, cutoff=C,
                                lookback_hours=24, history_lag_hours=2,
                                policy=AvailabilityPolicy("recorded", "documented"),
                                admit_quality=APPROVED)
        self.assertEqual(result["reasons"][-1], "UNAVAILABLE_AT_CUTOFF")

    def test_duplicates_and_invalid_discharge(self):
        row = observation(C - 2 * HOUR)
        with self.assertRaises(ValueError):
            history((row, row))
        for value in (None, -1.0, float("nan"), float("inf"), True):
            with self.subTest(value=value):
                result = history((dataclasses.replace(row, value=value),))
                self.assertEqual(result["reasons"][-1], "INVALID_DISCHARGE")
                self.assertIsNone(result["values"][-1])

    def test_target_flags_preserved_and_missing_targets_not_filled(self):
        row = observation(C + HOUR, flags=("A", "R"))
        result = target_window((row,), site=GAUGE_IDS[0], origin_anchor=C,
                               horizons_hours=HORIZONS_HOURS, admit_quality=APPROVED)
        self.assertEqual(result["values"][0], 10.0)
        self.assertEqual(result["quality_flags"][0], ("A", "R"))
        self.assertEqual(result["values"][1:], (None,) * 17)
        self.assertEqual(result["mask"][1:], (False,) * 17)

    def test_target_values_never_enter_history(self):
        past = [observation(C - h * HOUR) for h in range(25, 1, -1)]
        target_rows = [observation(C + h * HOUR, 999999) for h in HORIZONS_HOURS]
        self.assertEqual(history(past), history(past + target_rows))


class SoilStateTests(unittest.TestCase):
    def test_soil_is_separate_initial_state_with_age_and_original_flags(self):
        initial = SoilInitialState("soil00", C - 2 * HOUR, None, 0.3, ("soil-original",))
        result = soil_state((initial,), cutoff=C, policy=LAG, max_age_seconds=8 * HOUR)
        self.assertTrue(result["mask"])
        self.assertEqual(result["age_hours"], 2)
        self.assertEqual(result["quality_flags"], initial.quality_flags)
        self.assertNotIn("future_values", result)

    def test_stale_or_future_or_noninitial_soil_is_not_invented(self):
        initial = SoilInitialState("soil00", C - 2 * HOUR, None, 0.3)
        self.assertFalse(soil_state((initial,), cutoff=C, policy=LAG, max_age_seconds=HOUR)["mask"])
        future = dataclasses.replace(initial, reference=C + 4 * HOUR)
        self.assertFalse(soil_state((future,), cutoff=C, policy=LAG, max_age_seconds=8 * HOUR)["mask"])
        with self.assertRaises(ValueError):
            soil_state((dataclasses.replace(initial, reference=C - HOUR),),
                       cutoff=C, policy=LAG, max_age_seconds=8 * HOUR)

    def test_invalid_soil_fraction_is_masked(self):
        initial = SoilInitialState("soil00", C - 2 * HOUR, None, 1.1)
        result = soil_state((initial,), cutoff=C, policy=LAG, max_age_seconds=8 * HOUR)
        self.assertFalse(result["mask"])
        self.assertIsNone(result["value"])


class SplitAndScoringTests(unittest.TestCase):
    def test_all_8760_final_origins_preserved_with_declared_tail(self):
        admitted = [origin for origin in range(ASSESSMENT_START, ASSESSMENT_END, HOUR)
                    if assessment_origin_admitted(origin_anchor=origin,
                        target_times=forecast_target_times(origin, HORIZONS_HOURS))]
        self.assertEqual(len(admitted), 8760)
        self.assertEqual(admitted[-1] + 18 * HOUR, utc("2025-07-31T17:00:00Z"))
        self.assertEqual(admitted[-1] + 18 * HOUR, ASSESSMENT_LAST_TARGET)
        self.assertFalse(assessment_origin_admitted(origin_anchor=ASSESSMENT_END,
                         target_times=forecast_target_times(ASSESSMENT_END, HORIZONS_HOURS)))

    def test_final_cannot_expand_horizon_or_move_origin_boundary(self):
        targets = forecast_target_times(ASSESSMENT_END - HOUR, HORIZONS_HOURS)
        with self.assertRaises(ValueError):
            assessment_origin_admitted(origin_anchor=ASSESSMENT_END - HOUR,
                                       target_times=targets + (ASSESSMENT_LAST_TARGET + HOUR,))

    def test_development_purges_crossing_targets(self):
        split = Split("validation", ASSESSMENT_START - 48 * HOUR, ASSESSMENT_START)
        for hours_before, expected in ((19, True), (18, False), (1, False)):
            origin = ASSESSMENT_START - hours_before * HOUR
            self.assertEqual(target_window_contained(split, origin_anchor=origin,
                target_times=forecast_target_times(origin, HORIZONS_HOURS), embargo_seconds=0), expected)
        with self.assertRaises(ValueError):
            target_window_contained(Split("bad-development", ASSESSMENT_START, ASSESSMENT_END),
                origin_anchor=ASSESSMENT_START,
                target_times=forecast_target_times(ASSESSMENT_START, HORIZONS_HOURS), embargo_seconds=0)

    def test_final_scoring_disabled_even_when_coordinates_admitted(self):
        kwargs = dict(origin_anchor=ASSESSMENT_START,
                      target_times=forecast_target_times(ASSESSMENT_START, HORIZONS_HOURS))
        self.assertTrue(assessment_origin_admitted(**kwargs))
        with self.assertRaisesRegex(ValueError, "disabled"):
            require_scoring_allowed(**kwargs)
        with self.assertRaisesRegex(ValueError, "locked"):
            require_scoring_allowed(**kwargs, protected_assessment_enabled=True)
        self.assertTrue(require_scoring_allowed(**kwargs, protected_assessment_enabled=True,
            locked_protocol_digest="a" * 64, locked_selection_digest="b" * 64))

    def test_development_scoring_cannot_expose_boundary_labels(self):
        origin = ASSESSMENT_START - HOUR
        with self.assertRaisesRegex(ValueError, "cross"):
            require_scoring_allowed(origin_anchor=origin,
                target_times=forecast_target_times(origin, HORIZONS_HOURS))
        self.assertTrue(require_scoring_allowed(origin_anchor=C,
                        target_times=forecast_target_times(C, HORIZONS_HOURS)))

    def test_iterators_cannot_bypass_target_boundary_checks(self):
        origin = ASSESSMENT_START - HOUR
        split = Split("validation", ASSESSMENT_START - 48 * HOUR, ASSESSMENT_START)
        targets = forecast_target_times(origin, (h for h in HORIZONS_HOURS))
        self.assertEqual(len(targets), 18)
        self.assertFalse(target_window_contained(split, origin_anchor=origin,
            target_times=iter(targets), embargo_seconds=0))
        with self.assertRaisesRegex(ValueError, "cross"):
            require_scoring_allowed(origin_anchor=origin, target_times=iter(targets))
        with self.assertRaises(ValueError):
            forecast_target_times(C, iter(()))

    def test_validation_values_cannot_change_train_scaler(self):
        membership = {"train-a": "train", "train-b": "train", "validation": "validation"}
        train = (ScaleCell("train-a", "a", 10, True), ScaleCell("train-a", "b", 20, True),
                 ScaleCell("train-b", "a", 10, True))
        fitted = fit_train_scaler(train, case_splits=membership, variance_ddof=0, constant_scale=1)
        with_future = fit_train_scaler(train + (ScaleCell("validation", "future", 1e10, True),),
                                      case_splits=membership, variance_ddof=0, constant_scale=1)
        self.assertEqual(fitted, with_future)
        self.assertEqual(fitted.count, 2)
        self.assertEqual(fitted.mean, 15)


if __name__ == "__main__":
    unittest.main()
