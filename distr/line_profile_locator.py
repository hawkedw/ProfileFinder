#!/usr/bin/env python3
import csv
import json
import math
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import click
import numpy as np
import rasterio
from rasterio.windows import Window


@dataclass
class SearchResult:
    start_x: float
    start_y: float
    azimuth_deg: float
    step_m: float
    rmse: float
    mae: float
    corr: float
    sampled_count: int


def load_profile_csv(path: str, id_field: str, z_field: str) -> Tuple[np.ndarray, np.ndarray]:
    rows = []
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((int(row[id_field]), float(row[z_field])))
    rows.sort(key=lambda x: x[0])
    ids = np.array([r[0] for r in rows], dtype=np.int64)
    zs = np.array([r[1] for r in rows], dtype=np.float64)
    return ids, zs


def get_affine_step_coords(start_x: float, start_y: float, azimuth_deg: float, step_m: float, count: int):
    az = math.radians(azimuth_deg)
    dx = math.sin(az) * step_m
    dy = math.cos(az) * step_m
    xs = start_x + np.arange(count) * dx
    ys = start_y + np.arange(count) * dy
    return xs, ys


def sample_raster(src, xs: np.ndarray, ys: np.ndarray, band: int = 1) -> np.ndarray:
    coords = list(zip(xs.tolist(), ys.tolist()))
    vals = np.array([v[0] for v in src.sample(coords, indexes=band, masked=True)], dtype=np.float64)
    vals = np.where(np.isfinite(vals), vals, np.nan)
    return vals


def metrics(obs: np.ndarray, pred: np.ndarray) -> Tuple[float, float, float, int]:
    mask = np.isfinite(obs) & np.isfinite(pred)
    if mask.sum() < 3:
        return float('inf'), float('inf'), -1.0, int(mask.sum())
    o = obs[mask]
    p = pred[mask]
    diff = p - o
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    if np.std(o) == 0 or np.std(p) == 0:
        corr = -1.0
    else:
        corr = float(np.corrcoef(o, p)[0, 1])
    return rmse, mae, corr, int(mask.sum())


def moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return arr.copy()
    pad = window // 2
    padded = np.pad(arr, (pad, pad), mode='edge')
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode='valid')[: len(arr)]


def score_result(rmse: float, corr: float, mae: float, use_corr: bool = True) -> float:
    if not math.isfinite(rmse):
        return float('inf')
    if use_corr:
        return rmse * (1.0 + max(0.0, 1.0 - max(-1.0, min(1.0, corr)))) + 0.15 * mae
    return rmse + 0.15 * mae


def search_grid(
    src,
    profile_z: np.ndarray,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    azimuth_deg: float,
    azimuth_tol: float,
    azimuth_step: float,
    step_m: float,
    coarse_grid_m: float,
    band: int,
    smooth_window: int,
    top_k: int,
):
    if smooth_window > 1:
        profile_z = moving_average(profile_z, smooth_window)

    n = len(profile_z)
    line_length = (n - 1) * step_m
    azimuths = np.arange(azimuth_deg - azimuth_tol, azimuth_deg + azimuth_tol + 0.0001, azimuth_step)

    candidates = []
    ys = np.arange(y_min, y_max + 0.0001, coarse_grid_m)
    xs = np.arange(x_min, x_max + 0.0001, coarse_grid_m)

    for ay in ys:
        for ax in xs:
            for az in azimuths:
                end_x = ax + math.sin(math.radians(az)) * line_length
                end_y = ay + math.cos(math.radians(az)) * line_length
                if not (x_min <= end_x <= x_max and y_min <= end_y <= y_max):
                    continue
                px, py = get_affine_step_coords(ax, ay, az, step_m, n)
                sampled = sample_raster(src, px, py, band)
                if smooth_window > 1:
                    sampled = moving_average(sampled, smooth_window)
                rmse, mae, corr, cnt = metrics(profile_z, sampled)
                score = score_result(rmse, corr, mae, use_corr=True)
                candidates.append((score, SearchResult(ax, ay, az, step_m, rmse, mae, corr, cnt)))

    candidates.sort(key=lambda t: t[0])
    return [c[1] for c in candidates[:top_k]]


def refine_local(
    src,
    profile_z: np.ndarray,
    seed: SearchResult,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    step_m: float,
    band: int,
    smooth_window: int,
    iterations: int,
    xy_radius_m: float,
    xy_step_m: float,
    az_radius_deg: float,
    az_step_deg: float,
):
    if smooth_window > 1:
        profile_z = moving_average(profile_z, smooth_window)
    best = seed
    n = len(profile_z)
    line_length = (n - 1) * step_m

    for _ in range(iterations):
        improved = False
        for dy in np.arange(-xy_radius_m, xy_radius_m + 0.001, xy_step_m):
            for dx in np.arange(-xy_radius_m, xy_radius_m + 0.001, xy_step_m):
                for da in np.arange(-az_radius_deg, az_radius_deg + 0.0001, az_step_deg):
                    sx = best.start_x + dx
                    sy = best.start_y + dy
                    az = best.azimuth_deg + da
                    end_x = sx + math.sin(math.radians(az)) * line_length
                    end_y = sy + math.cos(math.radians(az)) * line_length
                    if not (x_min <= sx <= x_max and y_min <= sy <= y_max and x_min <= end_x <= x_max and y_min <= end_y <= y_max):
                        continue
                    px, py = get_affine_step_coords(sx, sy, az, step_m, n)
                    sampled = sample_raster(src, px, py, band)
                    if smooth_window > 1:
                        sampled = moving_average(sampled, smooth_window)
                    rmse, mae, corr, cnt = metrics(profile_z, sampled)
                    if score_result(rmse, corr, mae) < score_result(best.rmse, best.corr, best.mae):
                        best = SearchResult(sx, sy, az, step_m, rmse, mae, corr, cnt)
                        improved = True
        if not improved:
            xy_radius_m = max(xy_step_m, xy_radius_m / 2)
            az_radius_deg = max(az_step_deg, az_radius_deg / 2)
            if xy_radius_m == xy_step_m and az_radius_deg == az_step_deg:
                break
    return best


def write_points_csv(path: str, result: SearchResult, count: int):
    xs, ys = get_affine_step_coords(result.start_x, result.start_y, result.azimuth_deg, result.step_m, count)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['Id', 'X', 'Y'])
        for i, (x, y) in enumerate(zip(xs, ys), start=1):
            w.writerow([i, round(float(x), 6), round(float(y), 6)])


def write_geojson(path: str, result: SearchResult, count: int, crs_wkt: Optional[str] = None):
    xs, ys = get_affine_step_coords(result.start_x, result.start_y, result.azimuth_deg, result.step_m, count)
    features = []
    for i, (x, y) in enumerate(zip(xs, ys), start=1):
        features.append({
            'type': 'Feature',
            'properties': {'Id': i},
            'geometry': {'type': 'Point', 'coordinates': [float(x), float(y)]}
        })
    fc = {'type': 'FeatureCollection', 'features': features}
    if crs_wkt:
        fc['crs_note'] = 'Use raster CRS from result JSON or source raster; GeoJSON CRS members are not reliably supported.'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(fc, f, ensure_ascii=False)


@click.command()
@click.option('--dsm', required=True, type=click.Path(exists=True, dir_okay=False), help='Path to DSM raster.')
@click.option('--profile-csv', required=True, type=click.Path(exists=True, dir_okay=False), help='CSV with ordered Id and Z_DSM.')
@click.option('--id-field', default='Id', show_default=True)
@click.option('--z-field', default='Z_DSM', show_default=True)
@click.option('--search-xmin', required=True, type=float)
@click.option('--search-ymin', required=True, type=float)
@click.option('--search-xmax', required=True, type=float)
@click.option('--search-ymax', required=True, type=float)
@click.option('--azimuth', required=True, type=float, help='Expected azimuth of line in degrees.')
@click.option('--azimuth-tol', default=3.0, show_default=True, type=float)
@click.option('--azimuth-step', default=0.5, show_default=True, type=float)
@click.option('--point-step', default=5.0, show_default=True, type=float, help='Distance between ordered points along line.')
@click.option('--coarse-grid', default=60.0, show_default=True, type=float, help='Coarse search grid in map units.')
@click.option('--band', default=1, show_default=True, type=int)
@click.option('--smooth-window', default=5, show_default=True, type=int, help='Odd moving average window applied to both reference and sampled profiles.')
@click.option('--top-k', default=10, show_default=True, type=int)
@click.option('--refine-iterations', default=3, show_default=True, type=int)
@click.option('--refine-xy-radius', default=120.0, show_default=True, type=float)
@click.option('--refine-xy-step', default=20.0, show_default=True, type=float)
@click.option('--refine-az-radius', default=1.0, show_default=True, type=float)
@click.option('--refine-az-step', default=0.2, show_default=True, type=float)
@click.option('--out-dir', required=True, type=click.Path(file_okay=False), help='Output directory.')
def main(dsm, profile_csv, id_field, z_field, search_xmin, search_ymin, search_xmax, search_ymax,
         azimuth, azimuth_tol, azimuth_step, point_step, coarse_grid, band, smooth_window, top_k,
         refine_iterations, refine_xy_radius, refine_xy_step, refine_az_radius, refine_az_step, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ids, profile_z = load_profile_csv(profile_csv, id_field, z_field)

    with rasterio.open(dsm) as src:
        seeds = search_grid(
            src=src,
            profile_z=profile_z,
            x_min=search_xmin,
            y_min=search_ymin,
            x_max=search_xmax,
            y_max=search_ymax,
            azimuth_deg=azimuth,
            azimuth_tol=azimuth_tol,
            azimuth_step=azimuth_step,
            step_m=point_step,
            coarse_grid_m=coarse_grid,
            band=band,
            smooth_window=smooth_window,
            top_k=top_k,
        )
        if not seeds:
            raise click.ClickException('No valid candidate lines found inside search extent.')

        best = None
        refined_rows = []
        for seed in seeds:
            ref = refine_local(
                src=src,
                profile_z=profile_z,
                seed=seed,
                x_min=search_xmin,
                y_min=search_ymin,
                x_max=search_xmax,
                y_max=search_ymax,
                step_m=point_step,
                band=band,
                smooth_window=smooth_window,
                iterations=refine_iterations,
                xy_radius_m=refine_xy_radius,
                xy_step_m=refine_xy_step,
                az_radius_deg=refine_az_radius,
                az_step_deg=refine_az_step,
            )
            refined_rows.append(ref)
            if best is None or score_result(ref.rmse, ref.corr, ref.mae) < score_result(best.rmse, best.corr, best.mae):
                best = ref

        result_json = {
            'best_result': {
                'start_x': best.start_x,
                'start_y': best.start_y,
                'azimuth_deg': best.azimuth_deg,
                'step_m': best.step_m,
                'rmse': best.rmse,
                'mae': best.mae,
                'corr': best.corr,
                'sampled_count': best.sampled_count,
                'point_count': int(len(profile_z)),
                'raster_crs': str(src.crs),
                'search_extent': [search_xmin, search_ymin, search_xmax, search_ymax]
            },
            'other_top_results': [r.__dict__ for r in refined_rows[: min(10, len(refined_rows))]]
        }

    with open(os.path.join(out_dir, 'result.json'), 'w', encoding='utf-8') as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)

    write_points_csv(os.path.join(out_dir, 'recovered_points.csv'), best, len(profile_z))
    write_geojson(os.path.join(out_dir, 'recovered_points.geojson'), best, len(profile_z))

    with open(os.path.join(out_dir, 'README.md'), 'w', encoding='utf-8') as f:
        f.write('''# Line profile locator\n\n'
                'CLI tool that searches a DSM for a straight line whose sampled elevation profile best matches an ordered Z profile.\n\n'
                '## Input\n'
                '- DSM raster supported by GDAL/rasterio\n'
                '- CSV with ordered fields `Id` and `Z_DSM` (customizable via arguments)\n\n'
                '## Core idea\n'
                'The tool generates many straight candidate lines within a search rectangle, samples raster values along each line, and compares the sampled profile with the input profile using RMSE, MAE, and correlation. It then locally refines the best candidates.\n\n'
                '## Install\n'
                '```bash\n'
                'python -m venv .venv\n'
                '.venv\\Scripts\\activate\n'
                'pip install rasterio numpy click\n'
                '```\n\n'
                '## Run example\n'
                '```bash\n'
                'python line_profile_locator.py ^\n'
                '  --dsm dsm.tif ^\n'
                '  --profile-csv profile.csv ^\n'
                '  --search-xmin 500000 --search-ymin 5600000 ^\n'
                '  --search-xmax 542000 --search-ymax 5662000 ^\n'
                '  --azimuth 50 --azimuth-tol 3 --azimuth-step 0.5 ^\n'
                '  --point-step 5 ^\n'
                '  --coarse-grid 60 ^\n'
                '  --smooth-window 5 ^\n'
                '  --out-dir result\n'
                '```\n\n'
                '## Outputs\n'
                '- `result.json` — best line parameters and metrics\n'
                '- `recovered_points.csv` — recovered point coordinates in Id order\n'
                '- `recovered_points.geojson` — same points for quick GIS loading\n\n'
                '## Notes\n'
                '- Coordinates must be in the raster CRS.\n'
                '- For projected CRS in meters, `point-step`, `coarse-grid`, and search bounds are interpreted in meters.\n'
                '- `rasterio.sample` returns nearest-pixel values, so with coarse rasters a smoothing window usually helps.\n'
                '- Runtime can be heavy on large extents; start with a coarse grid and narrow azimuth tolerance.\n'
                '- If needed, the script can later be extended with bilinear interpolation, multithreading, and a two-stage pyramid search.\n')


if __name__ == '__main__':
    main()
