"""
core.py — ProfileFinder: profile matching for variable bearing/distance chains.

Supports:
- constant or per-row bearing values
- constant or per-row distances
- distance units in meters or degrees
- approximate start point search with perpendicular offset and start-point shift
"""

import csv
import math
import threading
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
import rasterio
from pyproj import Geod
from rasterio.transform import rowcol

_GEOD = Geod(ellps="WGS84")
_METERS_PER_DEGREE = 111320.0


@dataclass
class ProfilePoint:
    bearing: float
    z_dsm: float
    distance_value: float = 0.0
    distance_unit: str = "m"

    @property
    def distance_m(self) -> float:
        if (self.distance_unit or "m").lower().startswith("deg"):
            return float(self.distance_value) * _METERS_PER_DEGREE
        return float(self.distance_value)


@dataclass
class SearchResult:
    start_lon: float
    start_lat: float
    step_m: float
    rmse: float
    mae: float
    corr: float
    matched: int
    points: List[Tuple[float, float]] = field(default_factory=list)
    profile: List[ProfilePoint] = field(default_factory=list)
    z_sampled: List[float] = field(default_factory=list)
    best_shift_points: int = 0
    bearing: float = 0.0
    perp_offset_m: float = 0.0


@dataclass
class DsmCache:
    data: np.ndarray
    transform: object
    nodata: float
    rows: int
    cols: int


def load_dsm_cache(src: rasterio.DatasetReader, band: int = 1) -> DsmCache:
    data = src.read(band).astype(np.float64)
    nodata = src.nodata
    if nodata is not None:
        data[data == nodata] = np.nan
    data[~np.isfinite(data)] = np.nan
    return DsmCache(
        data=data,
        transform=src.transform,
        nodata=nodata if nodata is not None else np.nan,
        rows=data.shape[0],
        cols=data.shape[1],
    )


def sample_cache(cache: DsmCache, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    rows, cols = rowcol(cache.transform, lons, lats)
    rows = np.asarray(rows, dtype=np.int32)
    cols = np.asarray(cols, dtype=np.int32)
    valid = (rows >= 0) & (rows < cache.rows) & (cols >= 0) & (cols < cache.cols)
    out = np.full(len(lons), np.nan, dtype=np.float64)
    out[valid] = cache.data[rows[valid], cols[valid]]
    return out


def _parse_float(value, field_name: str) -> float:
    if value is None:
        raise ValueError(f"Field '{field_name}' has empty value.")
    text = str(value).strip().replace(",", ".")
    if text == "":
        raise ValueError(f"Field '{field_name}' has empty value.")
    return float(text)


def load_profile(
    path: str,
    bearing_mode: str = "field",
    bearing_field: str = "Bearing",
    bearing_const: float = 0.0,
    z_field: str = "Z_DSM",
    distance_mode: str = "const",
    distance_field: str = "Distance",
    distance_const: float = 5.0,
    distance_unit: str = "m",
) -> List[ProfilePoint]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        sample_data = f.read(8192)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample_data, delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(f, dialect=dialect)
        raw_fields = reader.fieldnames or []
        if len(raw_fields) == 1 and ";" in raw_fields[0]:
            f.seek(0)
            reader = csv.DictReader(f, dialect=csv.excel, delimiter=";")
            raw_fields = reader.fieldnames or []
        reader.fieldnames = [h.strip() for h in raw_fields]

        points: List[ProfilePoint] = []
        for row in reader:
            row = {(k.strip() if k else k): (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
            if z_field not in row:
                raise KeyError(f"Column '{z_field}' not found. Available: {list(row.keys())}")

            if (bearing_mode or "field") == "field":
                if bearing_field not in row:
                    raise KeyError(f"Column '{bearing_field}' not found. Available: {list(row.keys())}")
                bearing = _parse_float(row[bearing_field], bearing_field)
            else:
                bearing = float(bearing_const)

            if (distance_mode or "const") == "field":
                if distance_field not in row:
                    raise KeyError(f"Column '{distance_field}' not found. Available: {list(row.keys())}")
                distance_value = _parse_float(row[distance_field], distance_field)
            else:
                distance_value = float(distance_const)

            points.append(ProfilePoint(
                bearing=float(bearing),
                z_dsm=_parse_float(row[z_field], z_field),
                distance_value=float(distance_value),
                distance_unit=distance_unit,
            ))

    if not points:
        raise ValueError("CSV is empty or fields not found.")
    return points


def smooth(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return arr.copy()
    window = max(1, int(window))
    pad = window // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(padded, kernel, mode="valid")[:len(arr)]


def compute_metrics(ref: np.ndarray, sampled: np.ndarray) -> Tuple[float, float, float, int]:
    mask = np.isfinite(ref) & np.isfinite(sampled)
    n = int(mask.sum())
    if n < 3:
        return float("inf"), float("inf"), -1.0, n
    r = ref[mask]
    s = sampled[mask]
    diff = s - r
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    sr = float(np.std(r))
    ss = float(np.std(s))
    corr = float(np.corrcoef(r, s)[0, 1]) if sr > 1e-9 and ss > 1e-9 else 0.0
    return rmse, mae, corr, n


def score(rmse: float, corr: float, mae: float) -> float:
    if not math.isfinite(rmse):
        return float("inf")
    penalty = max(0.0, 1.0 - max(-1.0, min(1.0, corr)))
    return rmse * (1.0 + penalty) + 0.1 * mae


def shift_origin(lon: float, lat: float, azimuth_deg: float, dist_m: float) -> Tuple[float, float]:
    lon2, lat2, _ = _GEOD.fwd(lon, lat, azimuth_deg, dist_m)
    return float(lon2), float(lat2)


def build_chain(start_lon: float, start_lat: float, profile: List[ProfilePoint]) -> Tuple[np.ndarray, np.ndarray]:
    n = len(profile)
    lons = np.empty(n, dtype=np.float64)
    lats = np.empty(n, dtype=np.float64)
    lons[0] = float(start_lon)
    lats[0] = float(start_lat)
    for i in range(1, n):
        prev = profile[i - 1]
        lon2, lat2, _ = _GEOD.fwd(lons[i - 1], lats[i - 1], prev.bearing, prev.distance_m)
        lons[i] = lon2
        lats[i] = lat2
    return lons, lats


def shift_profile_start(profile: List[ProfilePoint], shift_pts: int) -> List[ProfilePoint]:
    n = len(profile)
    if n == 0 or shift_pts == 0:
        return list(profile)
    if shift_pts > 0:
        if shift_pts >= n:
            shift_pts = n - 1
        return [profile[min(i + shift_pts, n - 1)] for i in range(n)]
    k = abs(shift_pts)
    if k >= n:
        k = n - 1
    prefix = [profile[0] for _ in range(k)]
    return prefix + [profile[i - k] for i in range(k, n)]


def evaluate_candidate(
    cache: DsmCache,
    ref_z: np.ndarray,
    approx_lon: float,
    approx_lat: float,
    base_bearing: float,
    profile: List[ProfilePoint],
    perp_offset_m: float,
    along_shift_pts: int,
    smooth_window: int,
) -> Tuple[float, float, float, float, int, np.ndarray, Tuple[float, float], np.ndarray, np.ndarray]:
    cand_start_lon, cand_start_lat = shift_origin(approx_lon, approx_lat, base_bearing + 90.0, perp_offset_m)

    shifted_profile = shift_profile_start(profile, along_shift_pts)
    lons, lats = build_chain(cand_start_lon, cand_start_lat, shifted_profile)
    sampled = sample_cache(cache, lons, lats)
    sampled_eval = smooth(sampled, smooth_window) if smooth_window > 1 else sampled.copy()

    rmse, mae, corr, matched = compute_metrics(ref_z, sampled_eval)
    s = score(rmse, corr, mae)
    return s, rmse, mae, corr, matched, sampled, (cand_start_lon, cand_start_lat), lons, lats


def run_search(
    dsm_path: str,
    profile: List[ProfilePoint],
    start_lon: float,
    start_lat: float,
    search_radius_m: float,
    step_m: float,
    coarse_grid_m: float = 25.0,
    smooth_window: int = 1,
    top_k: int = 1,
    refine_iterations: int = 2,
    refine_xy_radius_m: float = 0.0,
    refine_xy_step_m: float = 0.0,
    band: int = 1,
    progress_cb: Optional[Callable] = None,
    stop_event: Optional[threading.Event] = None,
    log_cb: Optional[Callable] = None,
) -> SearchResult:
    if not profile:
        raise ValueError("Profile is empty")

    ref_z_raw = np.asarray([p.z_dsm for p in profile], dtype=np.float64)
    ref_z = smooth(ref_z_raw, smooth_window) if smooth_window > 1 else ref_z_raw.copy()

    base_bearing = float(profile[0].bearing)
    mean_step_m = float(np.mean([max(0.001, p.distance_m) for p in profile[:-1]])) if len(profile) > 1 else float(step_m)
    max_along_shift_pts = max(1, int(round(search_radius_m / max(mean_step_m, 0.001))))
    coarse_step = max(1.0, coarse_grid_m)
    offsets = np.arange(-search_radius_m, search_radius_m + coarse_step * 0.5, coarse_step, dtype=np.float64)
    shifts = np.arange(-max_along_shift_pts, max_along_shift_pts + 1, 1, dtype=np.int32)

    with rasterio.open(dsm_path) as src:
        if log_cb:
            log_cb(f"Loading DSM into RAM: {src.width}x{src.height} px ...")
        cache = load_dsm_cache(src, band)

    if log_cb:
        log_cb(
            f"Variable-chain mode. Points={len(profile)}, base bearing={base_bearing:.6f}°, "
            f"mean step={mean_step_m:.3f} m"
        )
        log_cb(f"Perpendicular search: {len(offsets)} candidates; along-start shifts: ±{max_along_shift_pts} pts")

    best = None
    total = max(1, len(offsets) * len(shifts) + refine_iterations * 20)
    done = 0

    def report_progress():
        if progress_cb:
            progress_cb(min(0.999, done / total))

    for perp_offset_m in offsets:
        if stop_event and stop_event.is_set():
            break
        for shift_pts in shifts:
            if stop_event and stop_event.is_set():
                break
            s, rmse, mae, corr, matched, sampled_raw, final_start, lons, lats = evaluate_candidate(
                cache=cache,
                ref_z=ref_z,
                approx_lon=start_lon,
                approx_lat=start_lat,
                base_bearing=base_bearing,
                profile=profile,
                perp_offset_m=float(perp_offset_m),
                along_shift_pts=int(shift_pts),
                smooth_window=smooth_window,
            )
            current = {
                "score": s,
                "rmse": rmse,
                "mae": mae,
                "corr": corr,
                "matched": matched,
                "shift_pts": int(shift_pts),
                "perp_offset_m": float(perp_offset_m),
                "start_lon": final_start[0],
                "start_lat": final_start[1],
                "lons": lons,
                "lats": lats,
                "sampled_raw": sampled_raw,
            }
            if best is None or current["score"] < best["score"]:
                best = current
                if log_cb:
                    log_cb(
                        f"Best coarse: perp={best['perp_offset_m']:.1f} m, shift={best['shift_pts']} pts, "
                        f"RMSE={best['rmse']:.4f}, corr={best['corr']:.4f}"
                    )
            done += 1
            report_progress()

    if best is None:
        raise RuntimeError("No valid candidates found in search area.")

    refine_step = max(1.0, coarse_step / 5.0)
    refine_radius = max(refine_step * 2, coarse_step)
    for it in range(refine_iterations):
        if stop_event and stop_event.is_set():
            break
        local_offsets = np.arange(
            best["perp_offset_m"] - refine_radius,
            best["perp_offset_m"] + refine_radius + refine_step * 0.5,
            refine_step,
            dtype=np.float64,
        )
        local_shifts = np.arange(best["shift_pts"] - 2, best["shift_pts"] + 3, 1, dtype=np.int32)
        improved = False
        for perp_offset_m in local_offsets:
            if stop_event and stop_event.is_set():
                break
            for shift_pts in local_shifts:
                if stop_event and stop_event.is_set():
                    break
                s, rmse, mae, corr, matched, sampled_raw, final_start, lons, lats = evaluate_candidate(
                    cache=cache,
                    ref_z=ref_z,
                    approx_lon=start_lon,
                    approx_lat=start_lat,
                    base_bearing=base_bearing,
                    profile=profile,
                    perp_offset_m=float(perp_offset_m),
                    along_shift_pts=int(shift_pts),
                    smooth_window=smooth_window,
                )
                if s < best["score"]:
                    best = {
                        "score": s,
                        "rmse": rmse,
                        "mae": mae,
                        "corr": corr,
                        "matched": matched,
                        "shift_pts": int(shift_pts),
                        "perp_offset_m": float(perp_offset_m),
                        "start_lon": final_start[0],
                        "start_lat": final_start[1],
                        "lons": lons,
                        "lats": lats,
                        "sampled_raw": sampled_raw,
                    }
                    improved = True
        refine_radius = max(refine_step * 2, refine_radius / 2.0)
        refine_step = max(1.0, refine_step / 2.0)
        if log_cb:
            log_cb(
                f"Refine {it + 1}: perp={best['perp_offset_m']:.2f} m, shift={best['shift_pts']} pts, "
                f"RMSE={best['rmse']:.4f}, corr={best['corr']:.4f}"
            )
        done += 20
        report_progress()
        if not improved and refine_step <= 1.0:
            break

    final_profile = shift_profile_start(profile, int(best["shift_pts"]))
    final_lons, final_lats = build_chain(best["start_lon"], best["start_lat"], final_profile)
    final_sampled = sample_cache(cache, final_lons, final_lats)
    rmse, mae, corr, matched = compute_metrics(ref_z_raw, final_sampled)

    if log_cb:
        log_cb(
            f"Final: start=({best['start_lat']:.8f}, {best['start_lon']:.8f}), "
            f"perp={best['perp_offset_m']:.2f} m, shift={best['shift_pts']} pts, "
            f"RMSE={rmse:.4f}, corr={corr:.4f}, matched={matched}/{len(profile)}"
        )
    if progress_cb:
        progress_cb(1.0)

    return SearchResult(
        start_lon=float(best["start_lon"]),
        start_lat=float(best["start_lat"]),
        step_m=float(mean_step_m if math.isfinite(mean_step_m) else step_m),
        rmse=rmse,
        mae=mae,
        corr=corr,
        matched=matched,
        points=list(zip(final_lons.tolist(), final_lats.tolist())),
        profile=final_profile,
        z_sampled=final_sampled.tolist(),
        best_shift_points=int(best["shift_pts"]),
        bearing=float(base_bearing),
        perp_offset_m=float(best["perp_offset_m"]),
    )
