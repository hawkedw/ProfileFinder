"""
core.py — ProfileFinder: fast straight-line profile matching.

Assumptions for current version:
- all points lie on one straight line
- Bearing is constant for all points (uses first valid bearing from CSV)
- start point is approximately known

Search strategy:
1) sample DSM along many parallel candidate lines shifted perpendicular to bearing
2) for each candidate line, find best along-line shift by normalized cross-correlation
3) refine around best perpendicular shift with smaller step
4) rebuild final point coordinates and sampled profile
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


@dataclass
class ProfilePoint:
    bearing: float
    z_dsm: float


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


def load_profile(path: str, bearing_field: str = "Bearing", z_field: str = "Z_DSM") -> List[ProfilePoint]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(f, dialect=dialect)
        reader.fieldnames = [h.strip() for h in (reader.fieldnames or [])]
        points = []
        for row in reader:
            row = {(k.strip() if k else k): (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
            if bearing_field not in row:
                raise KeyError(f"Column '{bearing_field}' not found. Available: {list(row.keys())}")
            if z_field not in row:
                raise KeyError(f"Column '{z_field}' not found. Available: {list(row.keys())}")
            points.append(ProfilePoint(bearing=float(row[bearing_field]), z_dsm=float(row[z_field])))
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


def build_straight_chain(start_lon: float, start_lat: float, bearing: float, step_m: float, count: int) -> Tuple[np.ndarray, np.ndarray]:
    distances = np.arange(count, dtype=np.float64) * step_m
    lons, lats, _ = _GEOD.fwd(
        np.full(count, start_lon, dtype=np.float64),
        np.full(count, start_lat, dtype=np.float64),
        np.full(count, bearing, dtype=np.float64),
        distances,
    )
    return np.asarray(lons, dtype=np.float64), np.asarray(lats, dtype=np.float64)


def normalized_xcorr_shift(ref: np.ndarray, cand: np.ndarray, max_shift: int) -> Tuple[int, float]:
    best_shift = 0
    best_corr = -2.0
    n = len(ref)
    for shift in range(-max_shift, max_shift + 1):
        if shift >= 0:
            r = ref[: n - shift]
            c = cand[shift:]
        else:
            r = ref[-shift:]
            c = cand[: n + shift]
        mask = np.isfinite(r) & np.isfinite(c)
        if mask.sum() < 10:
            continue
        rv = r[mask]
        cv = c[mask]
        rv = rv - rv.mean()
        cv = cv - cv.mean()
        denom = float(np.linalg.norm(rv) * np.linalg.norm(cv))
        corr = float(np.dot(rv, cv) / denom) if denom > 1e-12 else -1.0
        if corr > best_corr:
            best_corr = corr
            best_shift = shift
    return best_shift, best_corr


def apply_shift_to_profile(cand: np.ndarray, shift: int, out_len: int) -> np.ndarray:
    out = np.full(out_len, np.nan, dtype=np.float64)
    if shift >= 0:
        take = min(out_len, len(cand) - shift)
        if take > 0:
            out[:take] = cand[shift:shift + take]
    else:
        start = -shift
        take = min(out_len - start, len(cand))
        if take > 0:
            out[start:start + take] = cand[:take]
    return out


def evaluate_offset(cache: DsmCache,
                    ref_z: np.ndarray,
                    approx_lon: float,
                    approx_lat: float,
                    bearing: float,
                    perp_offset_m: float,
                    step_m: float,
                    max_along_shift_pts: int,
                    smooth_window: int) -> Tuple[float, float, float, int, int, np.ndarray, Tuple[float, float], np.ndarray, np.ndarray]:
    cand_start_lon, cand_start_lat = shift_origin(approx_lon, approx_lat, bearing + 90.0, perp_offset_m)
    extra = max_along_shift_pts
    line_start_lon, line_start_lat = shift_origin(cand_start_lon, cand_start_lat, bearing, -extra * step_m)
    total_count = len(ref_z) + 2 * extra
    lons_full, lats_full = build_straight_chain(line_start_lon, line_start_lat, bearing, step_m, total_count)
    sampled_full = sample_cache(cache, lons_full, lats_full)
    if smooth_window > 1:
        sampled_full = smooth(sampled_full, smooth_window)
    shift_pts, _ = normalized_xcorr_shift(ref_z, sampled_full, max_along_shift_pts)
    aligned = apply_shift_to_profile(sampled_full, shift_pts + extra, len(ref_z))
    rmse, mae, corr, matched = compute_metrics(ref_z, aligned)
    s = score(rmse, corr, mae)
    best_start_idx = max(0, shift_pts + extra)
    final_start_lon = float(lons_full[best_start_idx])
    final_start_lat = float(lats_full[best_start_idx])
    return s, rmse, mae, matched, shift_pts, aligned, (final_start_lon, final_start_lat), lons_full, lats_full


def run_search(dsm_path: str,
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
               log_cb: Optional[Callable] = None) -> SearchResult:
    if not profile:
        raise ValueError("Profile is empty")

    bearing = float(profile[0].bearing)
    if any(abs(p.bearing - bearing) > 1e-6 for p in profile):
        if log_cb:
            log_cb("Warning: bearings differ. Current algorithm uses bearing from the first point.")

    ref_z = np.asarray([p.z_dsm for p in profile], dtype=np.float64)
    ref_z_for_match = smooth(ref_z, smooth_window) if smooth_window > 1 else ref_z.copy()

    max_along_shift_pts = max(1, int(round(search_radius_m / step_m)))
    coarse_step = max(1.0, coarse_grid_m)
    offsets = np.arange(-search_radius_m, search_radius_m + coarse_step * 0.5, coarse_step, dtype=np.float64)

    with rasterio.open(dsm_path) as src:
        if log_cb:
            log_cb(f"Loading DSM into RAM: {src.width}x{src.height} px ...")
        cache = load_dsm_cache(src, band)

    if log_cb:
        log_cb(f"Straight-line mode. Bearing={bearing:.6f}°, points={len(profile)}, step={step_m} m")
        log_cb(f"Perpendicular search: {len(offsets)} candidates, along-line max shift: ±{max_along_shift_pts} pts")

    best = None
    total = max(1, len(offsets) + refine_iterations * 10)
    done = 0

    def report_progress():
        if progress_cb:
            progress_cb(min(0.999, done / total))

    for perp_offset_m in offsets:
        if stop_event and stop_event.is_set():
            break
        s, rmse, mae, matched, shift_pts, aligned, final_start, _, _ = evaluate_offset(
            cache=cache,
            ref_z=ref_z_for_match,
            approx_lon=start_lon,
            approx_lat=start_lat,
            bearing=bearing,
            perp_offset_m=float(perp_offset_m),
            step_m=step_m,
            max_along_shift_pts=max_along_shift_pts,
            smooth_window=smooth_window,
        )
        current = {
            "score": s,
            "rmse": rmse,
            "mae": mae,
            "corr": compute_metrics(ref_z_for_match, aligned)[2],
            "matched": matched,
            "shift_pts": shift_pts,
            "perp_offset_m": float(perp_offset_m),
            "start_lon": final_start[0],
            "start_lat": final_start[1],
            "aligned": aligned,
        }
        if best is None or current["score"] < best["score"]:
            best = current
            if log_cb:
                log_cb(
                    f"Best coarse: perp={best['perp_offset_m']:.2f} m, shift={best['shift_pts']} pts, "
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
        improved = False
        for perp_offset_m in local_offsets:
            if stop_event and stop_event.is_set():
                break
            s, rmse, mae, matched, shift_pts, aligned, final_start, _, _ = evaluate_offset(
                cache=cache,
                ref_z=ref_z_for_match,
                approx_lon=start_lon,
                approx_lat=start_lat,
                bearing=bearing,
                perp_offset_m=float(perp_offset_m),
                step_m=step_m,
                max_along_shift_pts=max_along_shift_pts,
                smooth_window=smooth_window,
            )
            corr = compute_metrics(ref_z_for_match, aligned)[2]
            if s < best["score"]:
                best = {
                    "score": s,
                    "rmse": rmse,
                    "mae": mae,
                    "corr": corr,
                    "matched": matched,
                    "shift_pts": shift_pts,
                    "perp_offset_m": float(perp_offset_m),
                    "start_lon": final_start[0],
                    "start_lat": final_start[1],
                    "aligned": aligned,
                }
                improved = True
        refine_radius = max(refine_step * 2, refine_radius / 2.0)
        refine_step = max(1.0, refine_step / 2.0)
        if log_cb:
            log_cb(
                f"Refine {it + 1}: perp={best['perp_offset_m']:.2f} m, shift={best['shift_pts']} pts, "
                f"RMSE={best['rmse']:.4f}, corr={best['corr']:.4f}"
            )
        done += 10
        report_progress()
        if not improved and refine_step <= 1.0:
            break

    final_lons, final_lats = build_straight_chain(best["start_lon"], best["start_lat"], bearing, step_m, len(profile))
    final_sampled = sample_cache(cache, final_lons, final_lats)
    rmse, mae, corr, matched = compute_metrics(ref_z, final_sampled)

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
        step_m=float(step_m),
        rmse=rmse,
        mae=mae,
        corr=corr,
        matched=matched,
        points=list(zip(final_lons.tolist(), final_lats.tolist())),
        profile=profile,
        z_sampled=final_sampled.tolist(),
        best_shift_points=int(best["shift_pts"]),
        bearing=float(bearing),
        perp_offset_m=float(best["perp_offset_m"]),
    )
