"""
core.py — ProfileFinder: anchor-last-point search algorithm.

Algorithm:
1. The approximate coordinate corresponds to the LAST point of the profile.
2. Candidate positions for the last point are searched within `search_radius_m`
   from the approximate coordinate, restricted to a bearing cone defined by
   the last point's BEAR_PREV ± bearing_cone_deg.
3. For each candidate last point, the chain is built BACKWARDS using each
   point's BEAR_PREV (reversed by +180°) and DIST_PREV.
4. DSM heights are sampled at every reconstructed position and compared to
   Z_DSM values (with a configurable Z tolerance for noisy measurements).
5. The candidate with the best RMSE/correlation score is returned.
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
    bear_prev: float      # bearing FROM previous point TO this point (deg)
    dist_prev_m: float    # distance from previous point (meters)
    z_dsm: float          # reference Z height (possibly noisy)


@dataclass
class SearchResult:
    start_lon: float      # lon of first (index 0) point
    start_lat: float      # lat of first (index 0) point
    last_lon: float       # lon of last point (the anchored one)
    last_lat: float       # lat of last point
    rmse: float
    mae: float
    corr: float
    matched: int
    points: List[Tuple[float, float]] = field(default_factory=list)  # (lon, lat) for each point
    profile: List[ProfilePoint] = field(default_factory=list)
    z_sampled: List[float] = field(default_factory=list)
    perp_offset_m: float = 0.0
    bearing_offset_deg: float = 0.0


@dataclass
class DsmCache:
    data: np.ndarray
    transform: object
    nodata: float
    rows: int
    cols: int


# ---------------------------------------------------------------------------
# DSM helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

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
    bearing_field: str = "BEAR_PREV",
    bearing_const: float = 0.0,
    z_field: str = "Z_DSM",
    distance_mode: str = "field",
    distance_field: str = "DIST_PREV",
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
                bear = _parse_float(row[bearing_field], bearing_field)
            else:
                bear = float(bearing_const)

            if (distance_mode or "field") == "field":
                if distance_field not in row:
                    raise KeyError(f"Column '{distance_field}' not found. Available: {list(row.keys())}")
                dist_val = _parse_float(row[distance_field], distance_field)
            else:
                dist_val = float(distance_const)

            # convert distance to meters if needed
            if (distance_unit or "m").lower().startswith("deg"):
                dist_m = dist_val * _METERS_PER_DEGREE
            else:
                dist_m = dist_val

            points.append(ProfilePoint(
                bear_prev=float(bear),
                dist_prev_m=float(dist_m),
                z_dsm=_parse_float(row[z_field], z_field),
            ))

    if not points:
        raise ValueError("CSV is empty or fields not found.")
    return points


# ---------------------------------------------------------------------------
# Chain building (backwards from last point)
# ---------------------------------------------------------------------------

def build_chain_backwards(
    last_lon: float,
    last_lat: float,
    profile: List[ProfilePoint],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct all point coordinates starting from the last point,
    walking backwards using reversed bearings and distances.

    Returns lons, lats arrays indexed [0..N-1] where index 0 = first profile point.
    """
    n = len(profile)
    lons = np.empty(n, dtype=np.float64)
    lats = np.empty(n, dtype=np.float64)
    lons[n - 1] = float(last_lon)
    lats[n - 1] = float(last_lat)

    for i in range(n - 1, 0, -1):
        # profile[i].bear_prev is the bearing FROM point[i-1] TO point[i]
        # Reverse it to walk from point[i] back to point[i-1]
        reverse_bearing = (profile[i].bear_prev + 180.0) % 360.0
        dist_m = max(0.001, profile[i].dist_prev_m)
        lon_prev, lat_prev, _ = _GEOD.fwd(lons[i], lats[i], reverse_bearing, dist_m)
        lons[i - 1] = lon_prev
        lats[i - 1] = lat_prev

    return lons, lats


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

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


def _bearing_diff(a: float, b: float) -> float:
    """Smallest signed difference between two bearings (degrees), result in [-180, 180]."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return float(d)


# ---------------------------------------------------------------------------
# Candidate grid generation (polar around approximate last point)
# ---------------------------------------------------------------------------

def _generate_candidates(
    approx_lon: float,
    approx_lat: float,
    last_bear_prev: float,
    search_radius_m: float,
    grid_step_m: float,
    bearing_cone_deg: float,
) -> List[Tuple[float, float]]:
    """
    Generate a grid of candidate positions for the last point.
    Candidates are within search_radius_m of the approximate coordinate AND
    within ±bearing_cone_deg of last_bear_prev (the bearing the last step arrives from).
    """
    candidates = []
    steps = max(1, int(math.ceil(search_radius_m / grid_step_m)))
    for di in range(-steps, steps + 1):
        for dj in range(-steps, steps + 1):
            dist = math.sqrt(di ** 2 + dj ** 2) * grid_step_m
            if dist > search_radius_m:
                continue
            if dist < 1e-6:
                candidates.append((approx_lon, approx_lat))
                continue
            # Cartesian offset → bearing
            bearing_to = math.degrees(math.atan2(dj, di))  # East=0 convention
            # Convert to North=0 bearing
            bearing_to = (90.0 - bearing_to) % 360.0
            # Filter by cone: we want candidates reachable from the direction
            # the last segment arrives, so the candidate should lie roughly
            # along last_bear_prev from the previous point — equivalently the
            # displacement from approx coord to candidate should be within cone.
            if abs(_bearing_diff(bearing_to, last_bear_prev)) > bearing_cone_deg:
                continue
            lon2, lat2, _ = _GEOD.fwd(approx_lon, approx_lat, bearing_to, dist)
            candidates.append((float(lon2), float(lat2)))
    return candidates


# ---------------------------------------------------------------------------
# Main search
# ---------------------------------------------------------------------------

def run_search(
    dsm_path: str,
    profile: List[ProfilePoint],
    approx_last_lon: float,
    approx_last_lat: float,
    search_radius_m: float,
    coarse_grid_m: float = 25.0,
    bearing_cone_deg: float = 45.0,
    smooth_window: int = 1,
    refine_iterations: int = 2,
    band: int = 1,
    progress_cb: Optional[Callable] = None,
    stop_event: Optional[threading.Event] = None,
    log_cb: Optional[Callable] = None,
    # kept for API compatibility but unused in new algorithm
    step_m: float = 5.0,
    top_k: int = 1,
    refine_xy_radius_m: float = 0.0,
    refine_xy_step_m: float = 0.0,
) -> SearchResult:
    if not profile:
        raise ValueError("Profile is empty")

    ref_z_raw = np.asarray([p.z_dsm for p in profile], dtype=np.float64)
    ref_z = smooth(ref_z_raw, smooth_window) if smooth_window > 1 else ref_z_raw.copy()

    last_bear_prev = profile[-1].bear_prev  # bearing of last segment (prev→last)

    with rasterio.open(dsm_path) as src:
        if log_cb:
            log_cb(f"Loading DSM into RAM: {src.width}x{src.height} px ...")
        cache = load_dsm_cache(src, band)

    if log_cb:
        log_cb(
            f"Anchor-last-point search. Points={len(profile)}, "
            f"last BEAR_PREV={last_bear_prev:.2f}°, cone=±{bearing_cone_deg}°, "
            f"radius={search_radius_m:.0f} m, grid={coarse_grid_m:.0f} m"
        )

    # --- coarse search ---
    candidates = _generate_candidates(
        approx_lon=approx_last_lon,
        approx_lat=approx_last_lat,
        last_bear_prev=last_bear_prev,
        search_radius_m=search_radius_m,
        grid_step_m=coarse_grid_m,
        bearing_cone_deg=bearing_cone_deg,
    )

    if log_cb:
        log_cb(f"Candidates (coarse): {len(candidates)}")

    total = max(1, len(candidates) + refine_iterations * 20)
    done = 0

    def report_progress():
        if progress_cb:
            progress_cb(min(0.999, done / total))

    best = None

    for cand_lon, cand_lat in candidates:
        if stop_event and stop_event.is_set():
            break

        lons, lats = build_chain_backwards(cand_lon, cand_lat, profile)
        sampled = sample_cache(cache, lons, lats)
        sampled_eval = smooth(sampled, smooth_window) if smooth_window > 1 else sampled.copy()
        rmse, mae, corr, matched = compute_metrics(ref_z, sampled_eval)
        s = score(rmse, corr, mae)

        current = {
            "score": s, "rmse": rmse, "mae": mae, "corr": corr,
            "matched": matched,
            "last_lon": cand_lon, "last_lat": cand_lat,
            "lons": lons, "lats": lats,
            "sampled": sampled,
        }
        if best is None or s < best["score"]:
            best = current
            if log_cb:
                log_cb(
                    f"  coarse best: last=({cand_lat:.7f}, {cand_lon:.7f}), "
                    f"RMSE={rmse:.4f}, corr={corr:.4f}"
                )
        done += 1
        report_progress()

    if best is None:
        raise RuntimeError("No valid candidates found in search area.")

    # --- refinement ---
    refine_step = max(1.0, coarse_grid_m / 5.0)
    refine_radius = max(refine_step * 2, coarse_grid_m)

    for it in range(refine_iterations):
        if stop_event and stop_event.is_set():
            break

        local_candidates = _generate_candidates(
            approx_lon=best["last_lon"],
            approx_lat=best["last_lat"],
            last_bear_prev=last_bear_prev,
            search_radius_m=refine_radius,
            grid_step_m=refine_step,
            bearing_cone_deg=bearing_cone_deg,
        )

        improved = False
        for cand_lon, cand_lat in local_candidates:
            if stop_event and stop_event.is_set():
                break
            lons, lats = build_chain_backwards(cand_lon, cand_lat, profile)
            sampled = sample_cache(cache, lons, lats)
            sampled_eval = smooth(sampled, smooth_window) if smooth_window > 1 else sampled.copy()
            rmse, mae, corr, matched = compute_metrics(ref_z, sampled_eval)
            s = score(rmse, corr, mae)
            if s < best["score"]:
                best = {
                    "score": s, "rmse": rmse, "mae": mae, "corr": corr,
                    "matched": matched,
                    "last_lon": cand_lon, "last_lat": cand_lat,
                    "lons": lons, "lats": lats,
                    "sampled": sampled,
                }
                improved = True

        refine_radius = max(refine_step * 2, refine_radius / 2.0)
        refine_step = max(0.5, refine_step / 2.0)

        if log_cb:
            log_cb(
                f"  refine {it + 1}: last=({best['last_lat']:.8f}, {best['last_lon']:.8f}), "
                f"RMSE={best['rmse']:.4f}, corr={best['corr']:.4f}"
            )
        done += 20
        report_progress()
        if not improved and refine_step <= 0.5:
            break

    # --- final pass with best last point ---
    final_lons, final_lats = build_chain_backwards(best["last_lon"], best["last_lat"], profile)
    final_sampled = sample_cache(cache, final_lons, final_lats)
    rmse, mae, corr, matched = compute_metrics(ref_z_raw, final_sampled)

    # Compute displacement of best last point from approx coordinate
    _, _, dist_offset = _GEOD.inv(approx_last_lon, approx_last_lat, best["last_lon"], best["last_lat"])
    bear_offset, _, _ = _GEOD.inv(approx_last_lon, approx_last_lat, best["last_lon"], best["last_lat"])

    if log_cb:
        log_cb(
            f"Final: last=({best['last_lat']:.8f}, {best['last_lon']:.8f}), "
            f"offset={dist_offset:.2f} m, "
            f"RMSE={rmse:.4f}, corr={corr:.4f}, matched={matched}/{len(profile)}"
        )
    if progress_cb:
        progress_cb(1.0)

    return SearchResult(
        start_lon=float(final_lons[0]),
        start_lat=float(final_lats[0]),
        last_lon=float(best["last_lon"]),
        last_lat=float(best["last_lat"]),
        rmse=rmse,
        mae=mae,
        corr=corr,
        matched=matched,
        points=list(zip(final_lons.tolist(), final_lats.tolist())),
        profile=list(profile),
        z_sampled=final_sampled.tolist(),
        perp_offset_m=float(dist_offset),
        bearing_offset_deg=float(bear_offset),
    )
