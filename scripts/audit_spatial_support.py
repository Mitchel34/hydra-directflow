#!/usr/bin/env python3
"""Read-only reproduction of retained basin footprints and candidate weights.

Uses resident, SHA-pinned geometry and grid metadata only. No weather/discharge
payloads, new downloads, geometry repairs, weight changes or scientific fits.
"""
import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import signal
import stat
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
GAUGES = ("03161000", "03164000", "03479000")
WEIGHTS_SHA = "9eaaf3b221360ed6faaa0376e953fd0c2d0356fb419df96daed9107404283a74"
PINS = {
    "CANDIDATE_WEIGHTS_MANIFEST.json": "7d95d99bc0566fb3da3f5d01b45b18ffe5c94c37b90282dc8661a0779d2eb6e0",
    "CANDIDATE_WEIGHTS_STATUS.json": "9ec7fca5f3cf1a57b3134e7b0515b8ae3a74b0e451c0543a524c9ed52c76d2f5",
    "PROTOCOL.json": "02c9cc79bc57dd6da7361a5710a95e0d548ef5fb412f45fecccac358715db0dc",
    "COORDINATE_EVIDENCE.json": "cefff3a079388e28c2681c2fccf506bda9e63828fc7291cf9925da862f126421",
    "RESULTS.json": "7848e3d0ac9856ed81ea58f8031b8c116ba177f7d7d316a09aabfa60fb0532aa",
    "ADMISSION_DECISIONS.json": "21a9f51854529b5ad6d734c2b92cbfc40b24981ebc602cd2c13f65d37f570c29",
}
NATIVE_CRS = "+proj=lcc +lat_0=38.5 +lon_0=-97.5 +lat_1=38.5 +lat_2=38.5 +x_0=0 +y_0=0 +R=6371229 +units=m +no_defs"


def require(condition, message):
    if not condition: raise ValueError(message)


def sha(raw): return hashlib.sha256(raw).hexdigest()


def resident_bytes(path, expected_sha, *, expected_size=None, maximum=2 * 1024**2):
    path = Path(path); before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and not getattr(before, "st_flags", 0) & 0x40000000,
            "Only resident regular files may be read; no implicit hydration")
    require(0 < before.st_size <= maximum and (expected_size is None or before.st_size == expected_size), "Source byte bound differs")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as stream:
        opened = os.fstat(stream.fileno()); raw = stream.read(maximum + 1)
    after = path.lstat(); stamp = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    require(stamp(before) == stamp(opened) == stamp(after) and len(raw) == before.st_size and sha(raw) == expected_sha,
            "Source identity changed or SHA differs")
    return raw


def availability(path):
    try: info = Path(path).lstat()
    except FileNotFoundError: return "MISSING"
    if not stat.S_ISREG(info.st_mode): return "NONREGULAR"
    return "DATALESS_NOT_READ" if getattr(info, "st_flags", 0) & 0x40000000 else "RESIDENT"


def parse_weights(raw):
    rows = list(csv.DictReader(io.StringIO(raw.decode())))
    require(len(rows) == 514, "Expected unchanged 514 candidate rows")
    by_gauge = {gauge: {} for gauge in GAUGES}
    for row in rows:
        gauge = row["gauge_id"]; index = int(row["crop_index"])
        i, j = int(row["native_i"]), int(row["native_j"])
        require(gauge in by_gauge and 0 <= index < 957 and index not in by_gauge[gauge], "Duplicate or invalid weight cell")
        require(index == (j - 478) * 29 + i - 1364 and 1364 <= i <= 1392 and 478 <= j <= 510
                and int(row["native_flat_index"]) == j * 1799 + i, "Weight grid mapping differs")
        require(row["status"] == "CANDIDATE_SCIENTIFIC_HOLD" and row["admitted"] == "false", "Original admission flags changed")
        area, basin, weight = (float(row[name]) for name in ("intersection_area_m2", "full_basin_area_m2", "weight"))
        require(all(math.isfinite(x) and x > 0 for x in (area, basin, weight)) and area <= basin
                and math.isclose(weight, area / basin, rel_tol=0, abs_tol=1e-15), "Area/weight formula differs")
        by_gauge[gauge][index] = dict(area=area, basin=basin, weight=weight)
    for gauge, values in by_gauge.items():
        require(values and len({v["basin"] for v in values.values()}) == 1, "Full basin denominator differs")
        require(abs(math.fsum(v["weight"] for v in values.values()) - 1) <= 1e-10, "Weights do not cover full candidate basin")
    return by_gauge


def polygon_geometry(value, gauge):
    from shapely.geometry import shape
    from shapely.ops import unary_union
    if gauge == "03164000":
        require(value.get("type") == "Feature" and value.get("id") == value.get("properties", {}).get("gage_id") == gauge,
                "GAGES-II feature identity differs")
        polygons = [shape(value["geometry"])]; point = None
        scopes = {"gauge_indexed_precomputed_basin": 1}
    else:
        require(value.get("type") == "FeatureCollection", "StreamStats collection required")
        features = value["features"]
        scopes = dict(Counter(f["properties"].get("scope") for f in features))
        require(scopes == dict(pourpoint=1, split_catchment=1, adjoint_catchment=1, upstream_basin=1), "StreamStats component scopes differ")
        point = shape(next(f["geometry"] for f in features if f["properties"]["scope"] == "pourpoint"))
        require(point.geom_type == "Point", "One returned pourpoint required")
        polygons = [shape(f["geometry"]) for f in features if f["properties"]["scope"] != "pourpoint"]
    # The original StreamStats response retains an empty upstream_basin when
    # split/adjoint components supply the whole basin. Preserve that zero-area
    # component; the union itself must still be valid and nonempty.
    require(all(p.geom_type in ("Polygon", "MultiPolygon") and p.is_valid for p in polygons),
            "Invalid source geometry; no repairs permitted")
    result = unary_union(polygons)
    require(result.is_valid and not result.is_empty, "Invalid component union")
    return result, point, scopes


def cell_ring(x, y, step):
    require(type(step) is int and step > 0 and 3000 % step == 0, "Exact edge spacing required")
    count = 3000 // step; left, bottom = x - 1500, y - 1500
    return ([(left + k * step, bottom) for k in range(count)]
            + [(left + 3000, bottom + k * step) for k in range(count)]
            + [(left + 3000 - k * step, bottom + 3000) for k in range(count)]
            + [(left, bottom + 3000 - k * step) for k in range(count)])


def build_cells(xs, ys, step, inverse_native, area_project):
    from shapely.geometry import Polygon
    cells = []
    for y in ys:
        for x in xs:
            ring = cell_ring(x, y, step)
            lon, lat = inverse_native.transform(*zip(*ring))
            east, north = area_project.transform(lon, lat)
            cell = Polygon(zip(east, north)); require(cell.is_valid and cell.area > 0, "Invalid projected native cell")
            cells.append(cell)
    return cells


def compare_intersections(basin, cells, supplied):
    from shapely.ops import unary_union
    areas = [basin.intersection(cell).area for cell in cells]
    positive = {i for i, value in enumerate(areas) if value > 0}
    require(positive == set(supplied), "Positive intersection footprint differs from original weights")
    area_errors = [abs(areas[i] - row["area"]) for i, row in supplied.items()]
    weight_errors = [abs(areas[i] / basin.area - row["weight"]) for i, row in supplied.items()]
    uncovered = basin.difference(unary_union(cells)).area
    tolerance = max(1.0, 1e-8 * basin.area)
    require(uncovered <= tolerance and abs(math.fsum(areas) - basin.area) <= tolerance, "Basin is not completely covered by native crop")
    require(max(area_errors) <= tolerance and math.fsum(weight_errors) <= 1e-8, "Independent intersection weights differ")
    return dict(positive_cells=len(positive), full_basin_area_m2=basin.area, uncovered_area_m2=uncovered,
                intersection_sum_m2=math.fsum(areas), max_absolute_intersection_area_error_m2=max(area_errors),
                max_absolute_weight_error=max(weight_errors), l1_weight_error=math.fsum(weight_errors),
                positive_cell_membership_exact=True, numerical_area_tolerance_m2=tolerance), areas


def run_audit(gate, weights, *, galax_body=None):
    import pyproj
    from pyproj import CRS, Transformer, Geod
    import shapely
    from shapely.geometry import Point
    from shapely.ops import transform
    pyproj.network.set_network_enabled(False)
    started = time.monotonic(); gate = Path(gate)
    records = {name: json.loads(resident_bytes(gate / name, hash_)) for name, hash_ in PINS.items()}
    protocol, coordinates = records["PROTOCOL.json"], records["COORDINATE_EVIDENCE.json"]
    manifest, decisions, historical = records["CANDIDATE_WEIGHTS_MANIFEST.json"], records["ADMISSION_DECISIONS.json"], records["RESULTS.json"]
    require(manifest["csv"]["sha256"] == WEIGHTS_SHA and coordinates["protocol_sha256"] == PINS["PROTOCOL.json"]
            and historical["coordinate_evidence_sha256"] == PINS["COORDINATE_EVIDENCE.json"], "Geometry evidence binding differs")
    source_map = {Path(item["path"]).name: item for item in protocol["input_files"]}
    prior_identity = source_map["INDEPENDENT_BASIN_VALIDATION_v2.json"]
    prior = json.loads(resident_bytes(prior_identity["path"], prior_identity["sha256"], expected_size=prior_identity["bytes"]))
    stations = {row["gauge"]: row for row in prior["station_comparisons"]}
    parsed = parse_weights(resident_bytes(weights, WEIGHTS_SHA, expected_size=62601))
    area_project = Transformer.from_crs(4326, 5070, always_xy=True, allow_ballpark=False)
    native = CRS.from_proj4(NATIVE_CRS)
    inverse_native = Transformer.from_crs(native, native.geodetic_crs, always_xy=True)
    xs, ys = coordinates["axes"]["x"]["crop_values"], coordinates["axes"]["y"]["crop_values"]
    require(len(xs) == 29 and len(ys) == 33 and all(abs(b-a-3000) < 1e-8 for v in (xs,ys) for a,b in zip(v,v[1:])), "Native crop axis differs")
    fine, coarse = (build_cells(xs, ys, step, inverse_native, area_project) for step in (125,250))
    rows, geometries, private_sources = [], {}, []
    for gauge in GAUGES:
        station = stations[gauge]; identity = station["geometry_qa"]["source"]
        require(identity["sha256"] == manifest["geometry_sha256"][gauge], "Weight/source geometry differs")
        path = Path(galax_body) if gauge == "03164000" and galax_body is not None else Path(identity["path"])
        state = availability(path)
        private_sources.append(dict(gauge_id=gauge, path=str(path.absolute()), **{k:identity[k] for k in ("sha256","bytes")}, residency=state))
        values = parsed[gauge]
        row = dict(gauge_id=gauge, source=station["selected_source"], geometry_sha256=identity["sha256"], geometry_bytes=identity["bytes"],
                   source_residency=state, original_status="CANDIDATE_SCIENTIFIC_HOLD", original_admitted=False,
                   weight_rows=len(values), weight_sum=math.fsum(v["weight"] for v in values.values()),
                   minimum_positive_weight=min(v["weight"] for v in values.values()), maximum_positive_weight=max(v["weight"] for v in values.values()),
                   zero_weight_cells=957-len(values), formula_verified=True, polygon_comparison_status="NOT_RECOMPUTED_SOURCE_NOT_RESIDENT",
                   accepted_case_ids=manifest["accepted_case_ids_by_gauge"][gauge],
                   historical_reference=dict(area_m2=historical["basins"][gauge]["full_basin_area_m2"],
                       area_minus_published_percent=station["area_minus_official_percent"], station_covered=station["gauge_coordinate_covered"],
                       station_boundary_distance_m=station["gauge_coordinate_to_boundary_m"],
                       snap_displacement_m=station.get("service_snap_displacement_from_official_coordinate_m")))
        if state == "RESIDENT":
            body = resident_bytes(path, identity["sha256"], expected_size=identity["bytes"])
            geometry, point, scopes = polygon_geometry(json.loads(body), gauge)
            metric = transform(area_project.transform, geometry); geometries[gauge] = metric
            require(metric.is_valid and metric.area > 0, "Projected basin invalid")
            fine_result, fine_areas = compare_intersections(metric, fine, values)
            _, coarse_areas = compare_intersections(metric, coarse, values)
            l1 = math.fsum(abs(a-b) / metric.area for a,b in zip(fine_areas,coarse_areas))
            require(l1 <= 1e-6, "Cell-edge densification did not converge")
            official = station["official_station_metadata"]
            gauge_point = transform(area_project.transform, Point(official["longitude"],official["latitude"]))
            official_area = official["drainage_area_sq_mi"] * 2.589988110336 * 1e6
            row.update(polygon_comparison_status="INDEPENDENT_REPRODUCTION_PASS", recomputed={**fine_result,
                "source_scope_counts":scopes, "source_valid":geometry.is_valid, "geometry_repairs":0,
                "full_basin_area_minus_stored_denominator_m2":metric.area-next(iter(values.values()))["basin"],
                "area_minus_published_percent":100*(metric.area/official_area-1), "gauge_coordinate_covered":metric.covers(gauge_point),
                "gauge_coordinate_to_boundary_m":metric.boundary.distance(gauge_point),
                "returned_pourpoint_covered":None if point is None else metric.covers(transform(area_project.transform,point)),
                "returned_pourpoint_wgs84":None if point is None else list(point.coords[0]),
                "snap_displacement_geodesic_m":None if point is None else Geod(ellps='WGS84').inv(official["longitude"],official["latitude"],point.x,point.y)[2],
                "edge_250m_to_125m_l1_weight_change":l1, "footprint_bounds_wgs84":list(geometry.bounds)})
        rows.append(row)
    nesting = dict(status="HISTORICAL_ONLY_MISSING_RESIDENT_GALAX", historical=prior["mixed_product_boundary_consistency"])
    if all(gauge in geometries for gauge in ("03161000","03164000")):
        jeff, galax = geometries["03161000"], geometries["03164000"]
        outside = jeff.difference(galax).area
        nesting.update(status="INDEPENDENT_REPRODUCTION_PASS", jefferson_outside_galax_m2=outside,
                       jefferson_outside_galax_percent=100*outside/jeff.area, galax_covers_jefferson=galax.covers(jeff),
                       forced_nesting_or_clipping=False)
    code_paths = [Path(__file__), ROOT/'src/hydra_directflow/local_data.py', ROOT/'src/hydra_directflow/spatial.py']
    report = dict(schema="hydra-spatial-support-audit/v1", status="NUMERICAL_REPRODUCTION_COMPLETE_SCIENTIFIC_ADMISSION_PENDING" if len(geometries)==3 else "PARTIAL_RESIDENT_REPRODUCTION_SCIENTIFIC_ADMISSION_PENDING",
        created_at_utc=datetime.now(timezone.utc).isoformat(), original_weight_sha256=WEIGHTS_SHA, original_weight_rows=514,
        source_evidence_sha256=PINS, independent_geometry_reproductions=len(geometries), gauges=rows, nesting=nesting,
        active_reduction=dict(implementation="local_data.basin_reduce", formula="sum_i area(B intersect C_i)/area(B) * x_i",
            coverage="sum_i weight_i * present_i", complete_mask="every positive-weight cell is present", renormalization=False,
            missing_positive_weight_cell="mask basin mean as NaN even if missing area fraction is tiny", missing_zero_weight_cell="does not affect basin mean or coverage",
            candidate_denominator="full retained delineated basin area; never published drainage-area number or observed-cell area"),
        scaffold_note="spatial.py is a generic proposal with optional explicit_renormalize; the executed development reduction uses local_data.basin_reduce and never selects that option.",
        prior_case_acceptances=[dict(id=x["id"], gauge_id=x["gauge_id"], metric=x["metric"], measured_value=x["measured_value"], disposition=x["disposition"]) for x in decisions["accepted_cases"]],
        remaining_scientific_decisions=["Confirm approximate gauge-associated basin claim with documented outlet/method/vintage limitations", "Confirm independent mixed StreamStats/GAGES-II Jefferson/Galax footprints without nesting, clipping or conservation claim"],
        limitations=["Numerical fidelity to retained official-source delineations does not establish surveyed geographic truth", "Published station coordinates/drainage areas are the October23 retained metadata snapshot, not refreshed hydrologic data", "Galax feature-specific outlet point/method/DEM vintage and StreamStats regional DEM build dates are not established", "Provider-authored Zarr CRS attribute is absent; native-grid interpretation is corroborated, not a provider CRS declaration"],
        official_method_references=["https://www.usgs.gov/streamstats/streamstats-fundamentals", "https://www.usgs.gov/publications/gages-ii-geospatial-attributes-gages-evaluating-streamflow", "https://api.water.usgs.gov/docs/nldi/basin/"],
        runtime=dict(python=sys.version.split()[0], shapely=shapely.__version__, pyproj=pyproj.__version__, proj_network_enabled=pyproj.network.is_network_enabled(),
                     area_pipeline=area_project.description, area_pipeline_stated_accuracy_m=area_project.accuracy, elapsed_seconds=time.monotonic()-started),
        implementation_sha256={str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in code_paths},
        execution=dict(source_geometry_or_weights_modified=False, original_metadata_modified=False, weather_or_discharge_values_read=False,
                       raw_archives_read=False, network_requests=0, cloud_jobs=0, optimizer_updates=0, protected_assessment_access=False,
                       scientific_admission_granted=False, missing_geometry_hydrated=False))
    private = dict(schema="hydra-spatial-audit-private-inputs/v1", source_geometry=private_sources, gate_directory=str(gate.absolute()),weights_path=str(Path(weights).absolute()))
    return report, private


def write_new(value,path):
    path=Path(path); require(not path.exists() and not path.is_symlink() and path.suffix=='.json',"Fresh JSON output required")
    require(path.parent.is_dir() and not path.parent.is_symlink(),"Resident output directory required")
    raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode(); require(len(raw)<1024**2,"Audit output bound exceeded")
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as stream:
        temp=Path(stream.name);stream.write(raw);stream.flush();os.fsync(stream.fileno())
    try:
        os.link(temp,path); fd=os.open(path.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
    finally: temp.unlink()
    return dict(sha256=sha(raw),bytes=len(raw))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('gate','weights','report','private-inputs'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--galax-body',type=Path)
    args=parser.parse_args()
    require(args.private_inputs.resolve().is_relative_to((ROOT/'.local').resolve()),"Private input paths must stay in ignored .local")
    require(not args.report.exists() and not args.private_inputs.exists(),"Outputs already exist")
    def deadline(*unused):raise TimeoutError('Spatial audit exceeded120seconds')
    previous=signal.signal(signal.SIGALRM,deadline);signal.alarm(120)
    try:
        report,private=run_audit(args.gate,args.weights,galax_body=args.galax_body)
        report['private_input_manifest']=write_new(private,args.private_inputs)
        write_new(report,args.report)
    finally:signal.alarm(0);signal.signal(signal.SIGALRM,previous)
    print(json.dumps(dict(status=report['status'],independent_geometry_reproductions=report['independent_geometry_reproductions'],scientific_admission_granted=False)))


if __name__=='__main__':main()
