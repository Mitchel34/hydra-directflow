"""Pure preparation of owned NWM rows and matched coordinates.

This module performs no I/O, inference, fitting, source authentication or release
of protected data. A consuming reader must authenticate its frozen release and
source bytes before passing rows here. Qualification objects bind declarations
and evidence identities; a syntactically valid digest is not proof of its bytes.

The owned BigQuery FLOAT representation is already decoded m3 s-1. The recorded
three-gauge native-file parity is evidence for that representation, not a reason
to apply NetCDF packing again. Negative finite predictions remain raw unless an
exact decoded fill value has a separate, explicitly bound qualification.
"""
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import re
from types import MappingProxyType

from .assessment_contract import paired_keys, validate_nwm_row_coordinates
from .reference import HOUR, require

SOURCE_TABLE = 'bigquery-public-data.national_water_model.short_range_channel_rt'
DECODED_REPRESENTATION = 'BIGQUERY_DECODED_FLOAT_M3_S'
UNITS = 'm3 s-1'
METADATA_FIELDS = ('feature_id', 'reference_time', 'valid_time', 'lead_hours', 'ensemble')
NUMERIC_VALID = 'NUMERIC_VALID'
SOURCE_ABSENT = 'SOURCE_VALUE_ABSENT'
SOURCE_NULL = 'SOURCE_VALUE_NULL'
SOURCE_CSV_NULL = 'SOURCE_VALUE_EMPTY_CSV_FIELD'
SOURCE_NONFINITE = 'SOURCE_VALUE_NONFINITE'
SOURCE_INVALID_TEXT = 'SOURCE_VALUE_INVALID_TEXT'
SOURCE_DECLARED_FILL = 'SOURCE_VALUE_DECLARED_FILL'

_INTEGER = re.compile(r'^(?:0|[1-9][0-9]*)$')
_DECIMAL = re.compile(r'^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$')
_NONFINITE = re.compile(r'^[+-]?(?:nan|snan|inf|infinity)$', re.IGNORECASE)
_HEX = re.compile(r'^[0-9a-f]{64}$')


def _sha(value):
    require(type(value) is str and _HEX.fullmatch(value) is not None,
            'An explicit lowercase SHA256 identity is required')
    return value


def _source_id(value):
    require(type(value) is str and bool(value.strip())
            and not any(c in value for c in ('\x00', '\r', '\n')),
            'A nonempty scalar source identity is required')
    return value


def _row_id(value):
    require((type(value) is int and value >= 0)
            or (type(value) is str and bool(value.strip())
                and not any(c in value for c in ('\x00', '\r', '\n'))),
            'An explicit nonnegative integer or nonempty string row identity is required')
    return value


def _finite_decimal(value):
    require(type(value) is str and _DECIMAL.fullmatch(value) is not None,
            'A declared decoded fill must be an exact finite decimal string')
    try:
        number = Decimal(value)
    except InvalidOperation as error:
        raise ValueError('Malformed decoded fill declaration') from error
    require(number.is_finite() and math.isfinite(float(number)),
            'A declared decoded fill must be finite in the source FLOAT representation')
    return number


@dataclass(frozen=True)
class NwmSourceQualification:
    """Immutable references requiring external byte and release authentication.

    ``decoded_fill_values`` are optional *BigQuery decoded-domain* declarations.
    They are never inferred from NetCDF sentinels, data frequency or model error.
    A nonempty fill list requires its own qualification artifact identity.
    """
    source_id: str
    snapshot_sha256: str
    parity_receipt_sha256: str
    units: str
    representation: str
    source_table: str = SOURCE_TABLE
    decoded_fill_values: tuple[str, ...] = ()
    fill_qualification_sha256: str | None = None

    def __post_init__(self):
        _source_id(self.source_id)
        _sha(self.snapshot_sha256)
        _sha(self.parity_receipt_sha256)
        require(self.units == UNITS and self.representation == DECODED_REPRESENTATION
                and self.source_table == SOURCE_TABLE,
                'The declared source must be decoded BigQuery short-range m3 s-1')
        require(type(self.decoded_fill_values) is tuple,
                'Decoded fill declarations must be an immutable tuple')
        fills = tuple(_finite_decimal(v) for v in self.decoded_fill_values)
        require(len(fills) == len(set(fills)), 'Duplicate decoded fill declarations')
        if fills:
            _sha(self.fill_qualification_sha256)
        else:
            require(self.fill_qualification_sha256 is None,
                    'An unused fill qualification cannot imply a fill policy')

    @property
    def identity(self):
        declaration = dict(source_id=self.source_id, snapshot_sha256=self.snapshot_sha256,
                           parity_receipt_sha256=self.parity_receipt_sha256,
                           units=self.units, representation=self.representation,
                           source_table=self.source_table,
                           decoded_fill_values=self.decoded_fill_values,
                           fill_qualification_sha256=self.fill_qualification_sha256)
        encoded = json.dumps(declaration, sort_keys=True, separators=(',', ':'), allow_nan=False)
        return hashlib.sha256(encoded.encode()).hexdigest()


def _integer(value):
    if type(value) is int:
        return value
    require(type(value) is str and _INTEGER.fullmatch(value) is not None,
            'NWM integer metadata must retain an integer or canonical integer string')
    return int(value)


def _utc_hour(value):
    if type(value) is int:
        require(value % HOUR == 0, 'Exact UTC hour required')
        return value
    require(type(value) is str, 'UTC timestamp string or integer epoch required')
    try:
        stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError as error:
        raise ValueError('Invalid NWM UTC timestamp') from error
    require(stamp.tzinfo is not None and stamp.utcoffset() == timezone.utc.utcoffset(stamp)
            and stamp.minute == stamp.second == stamp.microsecond == 0,
            'NWM timestamps must be exact timezone-aware UTC hours')
    return int(stamp.timestamp())


def _numeric_state(raw, present, source):
    if not present:
        return SOURCE_ABSENT, None
    if raw is None:
        return SOURCE_NULL, None
    require(type(raw) is str,
            'Preserve the original streamflow string; do not pre-convert its value')
    if raw == '':
        return SOURCE_CSV_NULL, None
    if _NONFINITE.fullmatch(raw):
        return SOURCE_NONFINITE, None
    if _DECIMAL.fullmatch(raw) is None:
        return SOURCE_INVALID_TEXT, None
    try:
        exact = Decimal(raw)
        value = float(raw)
    except (InvalidOperation, ValueError, OverflowError):
        return SOURCE_INVALID_TEXT, None
    if not exact.is_finite() or not math.isfinite(value):
        return SOURCE_NONFINITE, None
    if any(exact == _finite_decimal(fill) for fill in source.decoded_fill_values):
        return SOURCE_DECLARED_FILL, None
    # This is already physical flow. In particular, no multiplication by 0.01,
    # clipping, rounding, target-dependent exclusion, or negative-value mask.
    return NUMERIC_VALID, value


@dataclass(frozen=True)
class NormalizedNwmRow:
    key: tuple[str, int, int, int]
    source: NwmSourceQualification
    row_id: str | int
    raw_metadata: tuple[tuple[str, str | int], ...]
    raw_source_forecast_offset: str | int | None
    raw_streamflow_present: bool
    raw_streamflow: str | None
    numeric_status: str
    streamflow_m3_s: float | None

    @property
    def numeric_valid(self):
        return self.numeric_status == NUMERIC_VALID


def normalize_nwm_row(row, *, gauge_id, origin, origin_horizon, source, row_id):
    """Normalize a supplied row only after all matching metadata validates.

    The source reader, actual forecast values and authorization are external.
    Unknown extra row fields are never enumerated or accessed. Missing/null/
    nonfinite flow yields a retained source-invalid record, not an invented zero.
    """
    require(type(source) is NwmSourceQualification,
            'Explicit immutable NWM source-qualification references are required')
    _row_id(row_id)
    original = tuple((name, row[name]) for name in METADATA_FIELDS)
    metadata = {name: (_utc_hour(value) if name in ('reference_time', 'valid_time')
                       else _integer(value)) for name, value in original}
    key = validate_nwm_row_coordinates(metadata, gauge_id=gauge_id,
                                       origin_horizon=origin_horizon, origin=origin)
    offset = row['source_forecast_offset']
    require(offset is None or type(offset) in (str, int),
            'Original source_forecast_offset must remain an opaque scalar')
    try:
        raw = row['streamflow']
        present = True
    except KeyError:
        raw, present = None, False
    status, value = _numeric_state(raw, present, source)
    return NormalizedNwmRow(key, source, row_id, original, offset, present, raw, status, value)


def index_nwm_rows(rows):
    """Reject all duplicate keys; never silently deduplicate or merge sources.

    The already-audited unique view is the expected input. Even numerically equal
    rows can disagree in raw representation, original offset, row identity or
    source provenance. A future merge needs its own explicit lineage policy.
    """
    result = {}
    for row in rows:
        require(type(row) is NormalizedNwmRow, 'Only normalized NWM records may be indexed')
        _validate_normalized_row(row)
        if row.key in result:
            identical = result[row.key] == row
            raise ValueError('Duplicate NWM key requires explicit lineage' if identical
                             else 'Duplicate NWM key conflicts in value or source provenance')
        result[row.key] = row
    return MappingProxyType(dict(sorted(result.items())))


def _validate_normalized_row(row):
    """Do not trust a reconstructed record's numeric disposition or coordinates."""
    require(type(row.source) is NwmSourceQualification
            and type(row.raw_metadata) is tuple
            and all(type(pair) is tuple and len(pair) == 2 for pair in row.raw_metadata)
            and tuple(name for name, _ in row.raw_metadata) == METADATA_FIELDS,
            'Normalized row must retain the exact original metadata fields')
    _row_id(row.row_id)
    require(type(row.key) is tuple and len(row.key) == 4,
            'Normalized row must retain an explicit comparison key')
    metadata = {name: (_utc_hour(value) if name in ('reference_time', 'valid_time')
                       else _integer(value)) for name, value in row.raw_metadata}
    g, c, h, _ = row.key
    key = validate_nwm_row_coordinates(metadata, gauge_id=g, origin=c, origin_horizon=h)
    require(row.key == key and type(row.key[3]) is int,
            'Normalized key differs from retained metadata')
    require(row.raw_source_forecast_offset is None
            or type(row.raw_source_forecast_offset) in (str, int),
            'Original forecast offset must remain an opaque scalar')
    require(type(row.raw_streamflow_present) is bool
            and (row.raw_streamflow_present or row.raw_streamflow is None),
            'Absent streamflow cannot carry a hidden value')
    status, value = _numeric_state(row.raw_streamflow, row.raw_streamflow_present, row.source)
    require(status == row.numeric_status,
            'Recorded numeric disposition differs from original source value')
    require((value is None and row.streamflow_m3_s is None)
            or (value is not None and type(row.streamflow_m3_s) is float
                and row.streamflow_m3_s.hex() == value.hex()),
            'Recorded physical flow differs from unscaled source value')


def prepare_paired_matches(hydra_forecasts, nwm_rows, approved_targets):
    """Coordinate preparation with explicit NWM source-invalid accounting.

    No verification values are accessed. Target keys must come from the separately
    authenticated A/A+R target policy after the consumer's prediction barrier.
    This function does not enable scoring or change HYDRA forecast eligibility.
    """
    indexed = index_nwm_rows(nwm_rows)
    valid = frozenset(k for k, row in indexed.items() if row.numeric_valid)
    invalid = frozenset(indexed) - valid
    result = paired_keys(hydra_forecasts, valid, approved_targets)
    result.update(nwm_rows=indexed, nwm_coordinate_keys=frozenset(indexed),
                  nwm_numeric_valid_keys=valid, nwm_source_invalid_keys=invalid,
                  nwm_source_invalid_counts=MappingProxyType(dict(sorted(Counter(
                      indexed[k].numeric_status for k in invalid).items()))),
                  source_qualification_identities=tuple(sorted({r.source.identity for r in indexed.values()})))
    return result
