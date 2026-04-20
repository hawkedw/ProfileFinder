"""
core.py — Profile matching logic for ProfileFinder.
"""

import math
import csv
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np
import rasterio
from pyproj import Geod

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


def load_profile(path: str, bearing_field: str = "Bearing", z_field: str = "Z_DSM") -> List[ProfilePoint]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(f, dialect=dialect)
        # Strip whitespace from column names
        reader.fieldnames = [h.strip() for h in (reader.fieldnames or [])]
        points = []
        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items()}
            if bearing_field not in row:
                available = list(row.keys())
                raise KeyError(f"Column '{bearing_field}' not found. Available: {available}")
            if z_field not in row:
                available = list(row.keys())
                raise KeyError(f"Column '{z_field}' not found. Available: {available}")
            b = float(row[bearing_field])
            z = float(row[z_field])
            points.append(ProfilePoint(bearing=b, z_dsm=z))
    if not points:
        raise ValueError("CSV is empty or fields not found.")
    return points


def build_chain(start_lon: float, start_lat: float,
                bearings: np.ndarray, step_m: float) -> Tuple[np.ndarray, np.ndarray]:
    n = len(bearings)
    lons = np.empty(n, dtype=np.float64)
    lats = np.empty(n, dtype=np.float64)
    lon, lat = start_lon, start_lat
    for i, az in enumerate(bearings):
        lons[i] = lon
        lats[i] = lat
        end_lon, end_lat, _ = _GEOD.fwd(lon, lat, az, step_m)
        lon, lat = end_lon, end_lat
    return lons, lats


def sample_dsm(src: rasterio.DatasetReader,
               lons: np.ndarray, lats: np.ndarray,
               band: int = 1) -> np.ndarray:
    coords = list(zip(lons.tolist(), lats.tolist()))
    sampled = np.array(
        [v[0] for v in src.sample(coords, indexes=band, masked=True)],
        dtype=np.float64
    )
    nodata = src.nodata
    if nodata is not None:
        sampled[sampled == nodata] = np.nan
    sampled[~np.isfinite(sampled)] = np.nan
    return sampled


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
    if np.std(r) < 1e-9 or np.std(s) < 1e-9:
        corr = 0.0
    else:
        corr = float(np.corrcoef(r, s)[0, 1])
    return rmse, mae, corr, n


def score(rmse: float, corr: float, mae: float) -> float:
    if not math.isfinite(rmse):
        return float("inf")
    penalty = max(0.0, 1.0 - max(-1.0, min(1.0, corr)))
    return rmse * (1.0 + penalty) + 0.1 * mae


def smooth(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return arr.copy()
    pad = window // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")[: len(arr)]


def _evaluate(src, ref_z, bearings, start_lon, start_lat, step_m, band, smooth_w):
    lons, lats = build_chain(start_lon, start_lat, bearings, step_m)
    sampled = sample_dsm(src, lons, lats, band)
    if smooth_w > 1:
        sampled = smooth(sampled, smooth_w)
    s = score(*compute_metrics(ref_z, sampled)[:3])
    return s, lons, lats


def coarse_search(src, profile, start_lon, start_lat, search_radius_m,
                  grid_step_m, step_m, band, smooth_w, top_k, progress_cb=None):
    ref_z = np.array([p.z_dsm for p in profile], dtype=np.float64)
    if smooth_w > 1:
        ref_z = smooth(ref_z, smooth_w)
    bearings = np.array([p.bearing for p in profile], dtype=np.float64)

    deg_step = grid_step_m / 111_320.0
    lat_steps = np.arange(start_lat - search_radius_m / 111_320.0,
                          start_lat + search_radius_m / 111_320.0 + deg_step / 2,
                          deg_step)
    lon_steps = np.arange(start_lon - search_radius_m / 111_320.0,
                          start_lon + search_radius_m / 111_320.0 + deg_step / 2,
                          deg_step)

    total = len(lat_steps) * len(lon_steps)
    done = 0
    candidates = []

    for lat in lat_steps:
        for lon in lon_steps:
            s, lons, lats = _evaluate(src, ref_z, bearings, lon, lat, step_m, band, smooth_w)
            rmse, mae, corr, matched = compute_metrics(ref_z, sample_dsm(src, lons, lats, band))
            candidates.append((s, SearchResult(
                start_lon=float(lon), start_lat=float(lat), step_m=step_m,
                rmse=rmse, mae=mae, corr=corr, matched=matched,
                points=list(zip(lons.tolist(), lats.tolist()))
            )))
            done += 1
            if progress_cb and done % max(1, total // 100) == 0:
                progress_cb(done / total * 0.7)

    candidates.sort(key=lambda t: t[0])
    return [c[1] for c in candidates[:top_k]]


def refine_search(src, profile, seed, step_m, band, smooth_w,
                  iterations, xy_radius_m, xy_step_m,
                  progress_cb=None, progress_base=0.7, progress_range=0.3):
    ref_z = np.array([p.z_dsm for p in profile], dtype=np.float64)
    if smooth_w > 1:
        ref_z = smooth(ref_z, smooth_w)
    bearings = np.array([p.bearing for p in profile], dtype=np.float64)

    best = seed
    best_score = score(best.rmse, best.corr, best.mae)

    for it in range(iterations):
        improved = False
        deg_step = xy_step_m / 111_320.0
        deg_radius = xy_radius_m / 111_320.0
        offsets = np.arange(-deg_radius, deg_radius + deg_step / 2, deg_step)

        for dlat in offsets:
            for dlon in offsets:
                lon = best.start_lon + dlon
                lat = best.start_lat + dlat
                s, lons, lats = _evaluate(src, ref_z, bearings, lon, lat, step_m, band, smooth_w)
                if s < best_score:
                    rmse, mae, corr, matched = compute_metrics(ref_z, sample_dsm(src, lons, lats, band))
                    best = SearchResult(
                        start_lon=lon, start_lat=lat, step_m=step_m,
                        rmse=rmse, mae=mae, corr=corr, matched=matched,
                        points=list(zip(lons.tolist(), lats.tolist()))
                    )
                    best_score = s
                    improved = True

        if progress_cb:
            progress_cb(progress_base + progress_range * (it + 1) / iterations)

        if not improved:
            xy_radius_m = max(xy_step_m, xy_radius_m / 2)
            xy_step_m = max(1.0, xy_step_m / 2)
            if xy_step_m <= 1.0:
                break

    return best


def run_search(dsm_path, profile, start_lon, start_lat, search_radius_m,
               step_m, coarse_grid_m=60.0, smooth_window=5, top_k=10,
               refine_iterations=4, refine_xy_radius_m=120.0, refine_xy_step_m=20.0,
               band=1, progress_cb=None):
    with rasterio.open(dsm_path) as src:
        seeds = coarse_search(
            src=src, profile=profile,
            start_lon=start_lon, start_lat=start_lat,
            search_radius_m=search_radius_m,
            grid_step_m=coarse_grid_m,
            step_m=step_m, band=band,
            smooth_w=smooth_window, top_k=top_k,
            progress_cb=progress_cb,
        )
        if not seeds:
            raise RuntimeError("No valid candidates found in search area.")

        best = None
        for i, seed in enumerate(seeds):
            refined = refine_search(
                src=src, profile=profile, seed=seed,
                step_m=step_m, band=band, smooth_w=smooth_window,
                iterations=refine_iterations,
                xy_radius_m=refine_xy_radius_m,
                xy_step_m=refine_xy_step_m,
                progress_cb=progress_cb,
                progress_base=0.7 + 0.3 * i / len(seeds),
                progress_range=0.3 / len(seeds),
            )
            if best is None or score(refined.rmse, refined.corr, refined.mae) < score(best.rmse, best.corr, best.mae):
                best = refined

    return best
