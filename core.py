"""
core.py — ProfileFinder: optimized profile matching.

Key optimizations vs previous version:
- build_chain: numpy-vectorized, no Python loop
- sample_dsm: pixel-index lookup from in-RAM array instead of rasterio.sample()
- coarse_search: parallel via ThreadPoolExecutor + single sample per candidate
- refine_search: same fast path
- DSM tile loaded once into memory at start
"""

import math
import csv
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable

import numpy as np
import rasterio
from rasterio.transform import rowcol
from pyproj import Geod

_GEOD = Geod(ellps="WGS84")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# DSM tile cache — load entire raster band into RAM once
# ---------------------------------------------------------------------------

@dataclass
class DsmCache:
    data: np.ndarray        # shape (rows, cols), float32/64
    transform: object       # affine.Affine
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
    """Fast in-memory sampling: convert lon/lat → pixel indices → array lookup."""
    tf = cache.transform
    # Affine: col = (lon - x0) / dx,  row = (lat - y0) / dy
    cols = ((lons - tf.c) / tf.a).astype(np.int32)
    rows = ((lats - tf.f) / tf.e).astype(np.int32)
    valid = (rows >= 0) & (rows < cache.rows) & (cols >= 0) & (cols < cache.cols)
    result = np.full(len(lons), np.nan, dtype=np.float64)
    r = rows[valid]
    c = cols[valid]
    result[valid] = cache.data[r, c]
    return result


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------

def load_profile(path: str, bearing_field: str = "Bearing",
                 z_field: str = "Z_DSM") -> List[ProfilePoint]:
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
            row = {k.strip(): v.strip() for k, v in row.items()}
            if bearing_field not in row:
                raise KeyError(
                    f"Column '{bearing_field}' not found. Available: {list(row.keys())}")
            if z_field not in row:
                raise KeyError(
                    f"Column '{z_field}' not found. Available: {list(row.keys())}")
            points.append(ProfilePoint(
                bearing=float(row[bearing_field]),
                z_dsm=float(row[z_field]),
            ))
    if not points:
        raise ValueError("CSV is empty or fields not found.")
    return points


# ---------------------------------------------------------------------------
# Vectorized chain builder
# ---------------------------------------------------------------------------

def build_chain_vectorized(start_lon: float, start_lat: float,
                            bearings: np.ndarray, step_m: float,
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build point chain without Python loop.
    Uses flat-earth approximation: accurate to <0.1% for step_m <= 100 m.
    """
    n = len(bearings)
    az_rad = np.radians(bearings)
    # Metres per degree (approximate)
    m_per_lat = 111_320.0
    m_per_lon = 111_320.0 * math.cos(math.radians(start_lat))

    dlat = step_m * np.cos(az_rad) / m_per_lat
    dlon = step_m * np.sin(az_rad) / m_per_lon

    # Cumulative offsets: point i starts at sum of deltas 0..i-1
    cum_lat = np.concatenate([[0.0], np.cumsum(dlat[:-1])])
    cum_lon = np.concatenate([[0.0], np.cumsum(dlon[:-1])])

    lats = start_lat + cum_lat
    lons = start_lon + cum_lon
    return lons, lats


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(ref: np.ndarray,
                    sampled: np.ndarray) -> Tuple[float, float, float, int]:
    mask = np.isfinite(ref) & np.isfinite(sampled)
    n = int(mask.sum())
    if n < 3:
        return float("inf"), float("inf"), -1.0, n
    r = ref[mask]
    s = sampled[mask]
    diff = s - r
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    sr, ss = np.std(r), np.std(s)
    corr = float(np.corrcoef(r, s)[0, 1]) if sr > 1e-9 and ss > 1e-9 else 0.0
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
    return np.convolve(padded, np.ones(window) / window, mode="valid")[: len(arr)]


# ---------------------------------------------------------------------------
# Single candidate evaluation (uses cache)
# ---------------------------------------------------------------------------

def _eval_candidate(cache: DsmCache, ref_z: np.ndarray, bearings: np.ndarray,
                    start_lon: float, start_lat: float,
                    step_m: float, smooth_w: int) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    lons, lats = build_chain_vectorized(start_lon, start_lat, bearings, step_m)
    sampled = sample_cache(cache, lons, lats)
    if smooth_w > 1:
        sampled = smooth(sampled, smooth_w)
    s = score(*compute_metrics(ref_z, sampled)[:3])
    return s, lons, lats, sampled


# ---------------------------------------------------------------------------
# Coarse search — parallel
# ---------------------------------------------------------------------------

def coarse_search(cache: DsmCache, profile: List[ProfilePoint],
                  start_lon: float, start_lat: float,
                  search_radius_m: float, grid_step_m: float,
                  step_m: float, smooth_w: int, top_k: int,
                  progress_cb: Optional[Callable] = None,
                  stop_event: Optional[threading.Event] = None,
                  log_cb: Optional[Callable] = None,
                  n_workers: int = 0) -> List[SearchResult]:

    ref_z = np.array([p.z_dsm for p in profile], dtype=np.float64)
    if smooth_w > 1:
        ref_z = smooth(ref_z, smooth_w)
    bearings = np.array([p.bearing for p in profile], dtype=np.float64)

    deg_step = grid_step_m / 111_320.0
    lat_r = search_radius_m / 111_320.0
    lon_r = search_radius_m / 111_320.0

    lat_steps = np.arange(start_lat - lat_r, start_lat + lat_r + deg_step / 2, deg_step)
    lon_steps = np.arange(start_lon - lon_r, start_lon + lon_r + deg_step / 2, deg_step)

    # Build full grid of candidate origins
    grid = [(float(lat), float(lon)) for lat in lat_steps for lon in lon_steps]
    total = len(grid)

    if log_cb:
        log_cb(f"Coarse grid: {total} candidates ({len(lat_steps)}×{len(lon_steps)})")

    workers = n_workers if n_workers > 0 else min(8, (len(grid) // 50) + 1)
    done_count = [0]
    lock = threading.Lock()
    candidates = []

    def task(lat, lon):
        if stop_event and stop_event.is_set():
            return None
        s, lons, lats, sampled = _eval_candidate(
            cache, ref_z, bearings, lon, lat, step_m, smooth_w)
        rmse, mae, corr, matched = compute_metrics(ref_z, sampled)
        return (s, SearchResult(
            start_lon=lon, start_lat=lat, step_m=step_m,
            rmse=rmse, mae=mae, corr=corr, matched=matched,
            points=list(zip(lons.tolist(), lats.tolist())),
            profile=profile,
            z_sampled=sampled.tolist(),
        ))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(task, lat, lon): (lat, lon) for lat, lon in grid}
        for fut in as_completed(futures):
            if stop_event and stop_event.is_set():
                break
            res = fut.result()
            if res is not None:
                candidates.append(res)
            with lock:
                done_count[0] += 1
                if progress_cb and done_count[0] % max(1, total // 200) == 0:
                    progress_cb(done_count[0] / total * 0.7)

    candidates.sort(key=lambda t: t[0])

    if log_cb and candidates:
        best = candidates[0][1]
        log_cb(f"Coarse best: RMSE={best.rmse:.3f} Pearson={best.corr:.3f} "
               f"@ ({best.start_lat:.6f}, {best.start_lon:.6f})")

    return [c[1] for c in candidates[:top_k]]


# ---------------------------------------------------------------------------
# Refine search
# ---------------------------------------------------------------------------

def refine_search(cache: DsmCache, profile: List[ProfilePoint],
                  seed: SearchResult, step_m: float, smooth_w: int,
                  iterations: int, xy_radius_m: float, xy_step_m: float,
                  progress_cb: Optional[Callable] = None,
                  progress_base: float = 0.7, progress_range: float = 0.3,
                  stop_event: Optional[threading.Event] = None,
                  log_cb: Optional[Callable] = None) -> SearchResult:

    ref_z = np.array([p.z_dsm for p in profile], dtype=np.float64)
    if smooth_w > 1:
        ref_z = smooth(ref_z, smooth_w)
    bearings = np.array([p.bearing for p in profile], dtype=np.float64)

    best = seed
    best_score = score(best.rmse, best.corr, best.mae)

    for it in range(iterations):
        if stop_event and stop_event.is_set():
            break
        improved = False
        deg_step = xy_step_m / 111_320.0
        deg_radius = xy_radius_m / 111_320.0
        offsets = np.arange(-deg_radius, deg_radius + deg_step / 2, deg_step)

        # Vectorized: evaluate all (dlat, dlon) pairs at once
        dlat_grid, dlon_grid = np.meshgrid(offsets, offsets, indexing="ij")
        dlats = dlat_grid.ravel()
        dlons = dlon_grid.ravel()

        for dlat, dlon in zip(dlats, dlons):
            if stop_event and stop_event.is_set():
                break
            lon = best.start_lon + dlon
            lat = best.start_lat + dlat
            s, lons, lats, sampled = _eval_candidate(
                cache, ref_z, bearings, lon, lat, step_m, smooth_w)
            if s < best_score:
                rmse, mae, corr, matched = compute_metrics(ref_z, sampled)
                best = SearchResult(
                    start_lon=lon, start_lat=lat, step_m=step_m,
                    rmse=rmse, mae=mae, corr=corr, matched=matched,
                    points=list(zip(lons.tolist(), lats.tolist())),
                    profile=profile,
                    z_sampled=sampled.tolist(),
                )
                best_score = s
                improved = True

        if progress_cb:
            progress_cb(progress_base + progress_range * (it + 1) / iterations)
        if log_cb:
            log_cb(f"Refine it={it+1}: RMSE={best.rmse:.4f} "
                   f"Pearson={best.corr:.4f} step={xy_step_m:.1f}m")

        if not improved:
            xy_radius_m = max(xy_step_m, xy_radius_m / 2)
            xy_step_m = max(1.0, xy_step_m / 2)
            if xy_step_m <= 1.0:
                break

    return best


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_search(dsm_path: str, profile: List[ProfilePoint],
               start_lon: float, start_lat: float,
               search_radius_m: float, step_m: float,
               coarse_grid_m: float = 60.0, smooth_window: int = 5,
               top_k: int = 10, refine_iterations: int = 4,
               refine_xy_radius_m: float = 120.0, refine_xy_step_m: float = 20.0,
               band: int = 1, progress_cb: Optional[Callable] = None,
               stop_event: Optional[threading.Event] = None,
               log_cb: Optional[Callable] = None) -> SearchResult:

    with rasterio.open(dsm_path) as src:
        if log_cb:
            log_cb(f"Loading DSM into RAM: {src.width}×{src.height} px ...")
        cache = load_dsm_cache(src, band)
        if log_cb:
            log_cb(f"DSM loaded. Starting coarse search ...")

    seeds = coarse_search(
        cache=cache, profile=profile,
        start_lon=start_lon, start_lat=start_lat,
        search_radius_m=search_radius_m,
        grid_step_m=coarse_grid_m,
        step_m=step_m, smooth_w=smooth_window, top_k=top_k,
        progress_cb=progress_cb, stop_event=stop_event, log_cb=log_cb,
    )
    if not seeds:
        raise RuntimeError("No valid candidates found in search area.")

    if log_cb:
        log_cb(f"Refining top {len(seeds)} candidates ...")

    best = None
    for i, seed in enumerate(seeds):
        if stop_event and stop_event.is_set():
            break
        refined = refine_search(
            cache=cache, profile=profile, seed=seed,
            step_m=step_m, smooth_w=smooth_window,
            iterations=refine_iterations,
            xy_radius_m=refine_xy_radius_m,
            xy_step_m=refine_xy_step_m,
            progress_cb=progress_cb,
            progress_base=0.7 + 0.3 * i / len(seeds),
            progress_range=0.3 / len(seeds),
            stop_event=stop_event, log_cb=log_cb,
        )
        if best is None or score(refined.rmse, refined.corr, refined.mae) \
                         < score(best.rmse, best.corr, best.mae):
            best = refined

    return best
