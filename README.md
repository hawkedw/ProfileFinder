# ProfileFinder

Standalone GUI tool that searches a DSM raster for a chain of points whose sampled elevation profile best matches an ordered reference profile.

## Input CSV format

| Column | Description |
|--------|-------------|
| `Bearing` | Azimuth from the **previous** point to this point (degrees, 0–360) |
| `Z_DSM` | Reference DSM elevation value at this point (metres) |

First row is considered the **starting point** — its `Bearing` value is ignored (only used from row 2 onward).

## Parameters

| Parameter | Description |
|-----------|-------------|
| Start Longitude / Latitude | Approximate WGS84 coordinates of the **first** point |
| Search radius (m) | Radius around start point to search |
| Point step (m) | Distance between consecutive points along the chain |
| Coarse grid (m) | Grid spacing for the initial brute-force search |
| Smooth window | Moving average window applied to both profiles before comparison |

## Outputs

- **CSV** — `Id, Lon, Lat` for each recovered point (WGS84 decimal degrees)
- **GeoJSON** — same points as a FeatureCollection

## Install & run (Python)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

## Build standalone .exe

```bat
build.bat
```

Output: `dist\ProfileFinder.exe`

## Dependencies

- [rasterio](https://rasterio.readthedocs.io/) — DSM raster reading and sampling
- [pyproj](https://pyproj4.github.io/pyproj/) — WGS84 geodesic distance calculations
- [numpy](https://numpy.org/) — profile comparison metrics
- [PyInstaller](https://pyinstaller.org/) — standalone .exe packaging
