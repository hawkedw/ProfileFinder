"""
app.py — ProfileFinder GUI (tkinter)
"""

import csv
import json
import os
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from core import load_profile, run_search, SearchResult

SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")

DEFAULT_PARAMS = {
    "dsm": "",
    "csv": "",
    "start_lon": "0.0",
    "start_lat": "0.0",
    "search_radius": "5000",
    "step_m": "5",
    "coarse_grid": "25",
    "smooth_window": "1",
    "bearing_mode": "field",
    "bearing_const": "0",
    "bearing_field": "BEAR_PREV",
    "distance_mode": "field",
    "distance_const": "5",
    "distance_field": "DIST_PREV",
    "distance_unit": "m",
    "z_field": "Z_DSM",
}


def _browse_file(var: tk.StringVar, filetypes):
    path = filedialog.askopenfilename(filetypes=filetypes)
    if path:
        var.set(path)


def write_result_csv(path: str, result: SearchResult):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Id", "Lon", "Lat", "Z_DSM", "Bearing", "DistanceValue", "DistanceUnit", "Z_sampled", "dZ"])
        count = min(len(result.points), len(result.profile), len(result.z_sampled))
        for i in range(count):
            lon, lat = result.points[i]
            profile_pt = result.profile[i]
            z_samp = result.z_sampled[i]
            z_dsm = profile_pt.z_dsm
            dz = (z_samp - z_dsm) if (z_samp is not None and z_dsm is not None) else None
            w.writerow([
                i + 1,
                round(lon, 8),
                round(lat, 8),
                round(z_dsm, 4),
                round(profile_pt.bearing, 4),
                round(profile_pt.distance_value, 6),
                profile_pt.distance_unit,
                "" if z_samp is None or not isinstance(z_samp, (int, float)) or z_samp != z_samp else round(z_samp, 4),
                "" if dz is None or dz != dz else round(dz, 4),
            ])


def write_result_geojson(path: str, result: SearchResult):
    count = min(len(result.points), len(result.profile), len(result.z_sampled))
    features = []

    for i in range(count):
        lon, lat = result.points[i]
        profile_pt = result.profile[i]
        z_samp = result.z_sampled[i]
        z_dsm = profile_pt.z_dsm
        dz = (z_samp - z_dsm) if (z_samp is not None and z_dsm is not None) else None
        props = {
            "Id": i + 1,
            "Lon": round(lon, 8),
            "Lat": round(lat, 8),
            "Z_DSM": round(z_dsm, 4),
            "Bearing": round(profile_pt.bearing, 4),
            "DistanceValue": round(profile_pt.distance_value, 6),
            "DistanceUnit": profile_pt.distance_unit,
            "Z_sampled": None if z_samp is None or (isinstance(z_samp, float) and z_samp != z_samp) else round(z_samp, 4),
            "dZ": None if dz is None or (isinstance(dz, float) and dz != dz) else round(dz, 4),
        }
        features.append({
            "type": "Feature",
            "properties": props,
            "geometry": {
                "type": "Point",
                "coordinates": [round(lon, 8), round(lat, 8)]
            }
        })

    if count >= 2:
        features.append({
            "type": "Feature",
            "properties": {
                "type": "recovered_line",
                "RMSE": round(result.rmse, 4),
                "MAE": round(result.mae, 4),
                "Pearson": round(result.corr, 4),
                "Matched": result.matched,
                "Bearing": round(result.bearing, 4),
                "PerpOffset_m": round(result.perp_offset_m, 3),
                "AlongShiftPts": result.best_shift_points,
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [[round(result.points[i][0], 8), round(result.points[i][1], 8)] for i in range(count)]
            }
        })

    fc = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}
        },
        "features": features
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False, indent=2)


def read_csv_columns(path: str) -> list:
    with open(path, newline="", encoding="utf-8-sig") as f:
        sample = f.read(8192)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
        reader = csv.reader(f, dialect=dialect)
        try:
            row = next(reader)
            if len(row) == 1:
                for sep in (";", "\t", "|", ","):
                    parts = row[0].split(sep)
                    if len(parts) > 1:
                        return [c.strip() for c in parts]
            return [c.strip() for c in row]
        except StopIteration:
            return []


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ProfileFinder")
        self.resizable(False, False)
        self._result: SearchResult | None = None
        self._stop_event = threading.Event()
        self._csv_columns = []
        self._build_ui()
        self._load_settings()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        pad = {"padx": 8, "pady": 4}

        frame_files = ttk.LabelFrame(self, text="Inputs")
        frame_files.grid(row=0, column=0, sticky="ew", **pad)

        self._dsm_var = tk.StringVar()
        self._csv_var = tk.StringVar()

        ttk.Label(frame_files, text="DSM (.tif):").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(frame_files, textvariable=self._dsm_var, width=52).grid(row=0, column=1, **pad)
        ttk.Button(frame_files, text="Browse",
                   command=lambda: _browse_file(self._dsm_var,
                                                [("GeoTIFF", "*.tif *.tiff"), ("All", "*.*")])
                   ).grid(row=0, column=2, **pad)

        ttk.Label(frame_files, text="Profile CSV:").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(frame_files, textvariable=self._csv_var, width=52).grid(row=1, column=1, **pad)
        ttk.Button(frame_files, text="Browse", command=self._browse_csv).grid(row=1, column=2, **pad)

        frame_params = ttk.LabelFrame(self, text="Parameters")
        frame_params.grid(row=1, column=0, sticky="ew", **pad)

        self._param_vars = {
            "start_lon":      tk.StringVar(value=DEFAULT_PARAMS["start_lon"]),
            "start_lat":      tk.StringVar(value=DEFAULT_PARAMS["start_lat"]),
            "search_radius":  tk.StringVar(value=DEFAULT_PARAMS["search_radius"]),
            "step_m":         tk.StringVar(value=DEFAULT_PARAMS["step_m"]),
            "coarse_grid":    tk.StringVar(value=DEFAULT_PARAMS["coarse_grid"]),
            "smooth_window":  tk.StringVar(value=DEFAULT_PARAMS["smooth_window"]),
            "bearing_const":  tk.StringVar(value=DEFAULT_PARAMS["bearing_const"]),
            "bearing_field":  tk.StringVar(value=DEFAULT_PARAMS["bearing_field"]),
            "distance_const": tk.StringVar(value=DEFAULT_PARAMS["distance_const"]),
            "distance_field": tk.StringVar(value=DEFAULT_PARAMS["distance_field"]),
            "z_field":        tk.StringVar(value=DEFAULT_PARAMS["z_field"]),
        }

        ttk.Label(frame_params, text="Start Longitude (°):").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(frame_params, textvariable=self._param_vars["start_lon"], width=20).grid(row=0, column=1, sticky="w", **pad)

        ttk.Label(frame_params, text="Start Latitude (°):").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(frame_params, textvariable=self._param_vars["start_lat"], width=20).grid(row=1, column=1, sticky="w", **pad)

        ttk.Label(frame_params, text="Search radius (m):").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(frame_params, textvariable=self._param_vars["search_radius"], width=20).grid(row=2, column=1, sticky="w", **pad)

        ttk.Label(frame_params, text="Fallback step (m):").grid(row=3, column=0, sticky="w", **pad)
        ttk.Entry(frame_params, textvariable=self._param_vars["step_m"], width=20).grid(row=3, column=1, sticky="w", **pad)

        ttk.Label(frame_params, text="Perp step (m):").grid(row=4, column=0, sticky="w", **pad)
        ttk.Entry(frame_params, textvariable=self._param_vars["coarse_grid"], width=20).grid(row=4, column=1, sticky="w", **pad)

        ttk.Label(frame_params, text="Smooth window:").grid(row=5, column=0, sticky="w", **pad)
        ttk.Entry(frame_params, textvariable=self._param_vars["smooth_window"], width=20).grid(row=5, column=1, sticky="w", **pad)

        ttk.Label(frame_params, text="CSV Z_DSM field:").grid(row=6, column=0, sticky="w", **pad)
        self._z_combo = ttk.Combobox(frame_params, textvariable=self._param_vars["z_field"], width=22, state="normal")
        self._z_combo.grid(row=6, column=1, sticky="w", **pad)

        frame_bearing = ttk.LabelFrame(frame_params, text="Bearing source")
        frame_bearing.grid(row=0, column=2, rowspan=3, sticky="nw", padx=12, pady=4)
        self._bearing_mode = tk.StringVar(value=DEFAULT_PARAMS["bearing_mode"])
        ttk.Radiobutton(frame_bearing, text="CSV column", value="field", variable=self._bearing_mode, command=self._update_mode_state).grid(row=0, column=0, sticky="w", **pad)
        ttk.Radiobutton(frame_bearing, text="Constant", value="const", variable=self._bearing_mode, command=self._update_mode_state).grid(row=1, column=0, sticky="w", **pad)
        ttk.Label(frame_bearing, text="Column:").grid(row=2, column=0, sticky="w", **pad)
        self._bearing_combo = ttk.Combobox(frame_bearing, textvariable=self._param_vars["bearing_field"], width=22, state="readonly", postcommand=self._refresh_bearing_combo)
        self._bearing_combo.grid(row=3, column=0, sticky="w", **pad)
        ttk.Label(frame_bearing, text="Const azimuth (°):").grid(row=4, column=0, sticky="w", **pad)
        self._bearing_const_entry = ttk.Entry(frame_bearing, textvariable=self._param_vars["bearing_const"], width=24)
        self._bearing_const_entry.grid(row=5, column=0, sticky="w", **pad)

        frame_distance = ttk.LabelFrame(frame_params, text="Distance source")
        frame_distance.grid(row=3, column=2, rowspan=4, sticky="nw", padx=12, pady=4)
        self._distance_mode = tk.StringVar(value=DEFAULT_PARAMS["distance_mode"])
        ttk.Radiobutton(frame_distance, text="CSV column", value="field", variable=self._distance_mode, command=self._update_mode_state).grid(row=0, column=0, sticky="w", **pad)
        ttk.Radiobutton(frame_distance, text="Constant", value="const", variable=self._distance_mode, command=self._update_mode_state).grid(row=1, column=0, sticky="w", **pad)
        ttk.Label(frame_distance, text="Column:").grid(row=2, column=0, sticky="w", **pad)
        self._distance_combo = ttk.Combobox(frame_distance, textvariable=self._param_vars["distance_field"], width=22, state="readonly", postcommand=self._refresh_distance_combo)
        self._distance_combo.grid(row=3, column=0, sticky="w", **pad)
        ttk.Label(frame_distance, text="Const distance:").grid(row=4, column=0, sticky="w", **pad)
        self._distance_const_entry = ttk.Entry(frame_distance, textvariable=self._param_vars["distance_const"], width=24)
        self._distance_const_entry.grid(row=5, column=0, sticky="w", **pad)
        ttk.Label(frame_distance, text="Unit:").grid(row=6, column=0, sticky="w", **pad)
        self._distance_unit = tk.StringVar(value=DEFAULT_PARAMS["distance_unit"])
        self._distance_unit_combo = ttk.Combobox(frame_distance, textvariable=self._distance_unit, values=["m", "deg"], width=8, state="readonly")
        self._distance_unit_combo.grid(row=7, column=0, sticky="w", **pad)

        frame_run = ttk.Frame(self)
        frame_run.grid(row=2, column=0, **pad)

        self._btn_run = ttk.Button(frame_run, text="▶  Run", command=self._on_run)
        self._btn_run.grid(row=0, column=0, padx=4)

        self._btn_stop = ttk.Button(frame_run, text="■  Stop", state="disabled", command=self._on_stop)
        self._btn_stop.grid(row=0, column=1, padx=4)

        self._btn_csv = ttk.Button(frame_run, text="Save CSV", state="disabled", command=self._save_csv)
        self._btn_csv.grid(row=0, column=2, padx=4)

        self._btn_geojson = ttk.Button(frame_run, text="Save GeoJSON", state="disabled", command=self._save_geojson)
        self._btn_geojson.grid(row=0, column=3, padx=4)

        self._progress = ttk.Progressbar(self, length=620, mode="determinate")
        self._progress.grid(row=3, column=0, **pad)

        frame_log = ttk.LabelFrame(self, text="Log")
        frame_log.grid(row=4, column=0, sticky="nsew", **pad)

        self._log = tk.Text(frame_log, height=16, width=104, state="disabled", font=("Consolas", 9))
        self._log.grid(row=0, column=0)
        scrollbar = ttk.Scrollbar(frame_log, command=self._log.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self._log.configure(yscrollcommand=scrollbar.set)

        self._update_mode_state()

    def _refresh_bearing_combo(self):
        self._bearing_combo["values"] = self._csv_columns

    def _refresh_distance_combo(self):
        self._distance_combo["values"] = self._csv_columns

    def _load_settings(self):
        data = {}
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}
        self._dsm_var.set(data.get("dsm", DEFAULT_PARAMS["dsm"]))
        self._csv_var.set(data.get("csv", DEFAULT_PARAMS["csv"]))
        for key, var in self._param_vars.items():
            var.set(data.get(key, DEFAULT_PARAMS.get(key, var.get())))
        self._bearing_mode.set(data.get("bearing_mode", DEFAULT_PARAMS["bearing_mode"]))
        self._distance_mode.set(data.get("distance_mode", DEFAULT_PARAMS["distance_mode"]))
        self._distance_unit.set(data.get("distance_unit", DEFAULT_PARAMS["distance_unit"]))
        if self._csv_var.get() and os.path.exists(self._csv_var.get()):
            self._load_csv_columns(self._csv_var.get())
        self._update_mode_state()

    def _save_settings(self):
        try:
            data = {
                "dsm": self._dsm_var.get(),
                "csv": self._csv_var.get(),
                "bearing_mode": self._bearing_mode.get(),
                "distance_mode": self._distance_mode.get(),
                "distance_unit": self._distance_unit.get(),
            }
            for key, var in self._param_vars.items():
                data[key] = var.get()
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _on_close(self):
        self._save_settings()
        self.destroy()

    def _load_csv_columns(self, path: str):
        cols = read_csv_columns(path)
        self._csv_columns = cols
        self._bearing_combo["values"] = cols
        self._distance_combo["values"] = cols
        self._z_combo["values"] = cols
        if cols:
            self._log_write(f"CSV columns detected: {cols}")
            lower_map = {c.strip().lower(): c for c in cols}
            # авто-выбор: BEAR_PREV приоритетнее BEARING
            for candidate in ("bear_prev", "bearing"):
                if candidate in lower_map:
                    self._param_vars["bearing_field"].set(lower_map[candidate])
                    break
            # авто-выбор: DIST_PREV приоритетнее DISTANCE
            for candidate in ("dist_prev", "distance"):
                if candidate in lower_map:
                    self._param_vars["distance_field"].set(lower_map[candidate])
                    break
            if "z_dsm" in lower_map:
                self._param_vars["z_field"].set(lower_map["z_dsm"])

    def _browse_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if not path:
            return
        self._csv_var.set(path)
        self._load_csv_columns(path)
        self._update_mode_state()

    def _update_mode_state(self):
        if self._bearing_mode.get() == "field":
            self._bearing_combo.configure(state="readonly")
            self._bearing_const_entry.configure(state="disabled")
        else:
            self._bearing_combo.configure(state="disabled")
            self._bearing_const_entry.configure(state="normal")

        if self._distance_mode.get() == "field":
            self._distance_combo.configure(state="readonly")
            self._distance_const_entry.configure(state="disabled")
        else:
            self._distance_combo.configure(state="disabled")
            self._distance_const_entry.configure(state="normal")

    def _log_write(self, msg: str):
        self._log.configure(state="normal")
        self._log.insert("end", msg + "\n")
        self._log.see("end")
        self._log.configure(state="disabled")

    def _log_write_safe(self, msg: str):
        self.after(0, self._log_write, msg)

    def _progress_set(self, value: float):
        self._progress["value"] = value * 100
        self.update_idletasks()

    def _on_run(self):
        dsm = self._dsm_var.get().strip()
        csv_path = self._csv_var.get().strip()
        if not dsm or not os.path.exists(dsm):
            messagebox.showerror("Error", "DSM file not found.")
            return
        if not csv_path or not os.path.exists(csv_path):
            messagebox.showerror("Error", "Profile CSV not found.")
            return

        try:
            start_lon = float(self._param_vars["start_lon"].get().replace(",", "."))
            start_lat = float(self._param_vars["start_lat"].get().replace(",", "."))
            search_radius = float(self._param_vars["search_radius"].get().replace(",", "."))
            step_m = float(self._param_vars["step_m"].get().replace(",", "."))
            coarse_grid = float(self._param_vars["coarse_grid"].get().replace(",", "."))
            smooth_window = int(self._param_vars["smooth_window"].get())
            bearing_const = float(self._param_vars["bearing_const"].get().replace(",", "."))
            distance_const = float(self._param_vars["distance_const"].get().replace(",", "."))
            bearing_field = self._param_vars["bearing_field"].get().strip()
            distance_field = self._param_vars["distance_field"].get().strip()
            z_field = self._param_vars["z_field"].get().strip()
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid parameter: {e}")
            return

        bearing_mode = self._bearing_mode.get()
        distance_mode = self._distance_mode.get()
        distance_unit = self._distance_unit.get()

        self._stop_event.clear()
        self._btn_run.configure(state="disabled")
        self._btn_stop.configure(state="normal")
        self._btn_csv.configure(state="disabled")
        self._btn_geojson.configure(state="disabled")
        self._progress["value"] = 0
        self._log.configure(state="normal")
        self._log.delete("1.0", "end")
        self._log.configure(state="disabled")
        self._result = None
        self._save_settings()

        def worker():
            try:
                cols = read_csv_columns(csv_path)
                self._log_write_safe(f"CSV columns: {cols}")
                self._log_write_safe(
                    f"Using bearing mode='{bearing_mode}', field='{bearing_field}', const={bearing_const}; "
                    f"distance mode='{distance_mode}', field='{distance_field}', const={distance_const} {distance_unit}; "
                    f"z='{z_field}'"
                )

                profile = load_profile(
                    csv_path,
                    bearing_mode=bearing_mode,
                    bearing_field=bearing_field,
                    bearing_const=bearing_const,
                    z_field=z_field,
                    distance_mode=distance_mode,
                    distance_field=distance_field,
                    distance_const=distance_const,
                    distance_unit=distance_unit,
                )
                self._log_write_safe(f"Loaded {len(profile)} points.")
                self._log_write_safe(
                    f"Search from ({start_lat:.6f}, {start_lon:.6f}), radius={search_radius} m, "
                    f"fallback_step={step_m} m, perp_step={coarse_grid} m..."
                )

                result = run_search(
                    dsm_path=dsm,
                    profile=profile,
                    start_lon=start_lon,
                    start_lat=start_lat,
                    search_radius_m=search_radius,
                    step_m=step_m,
                    coarse_grid_m=coarse_grid,
                    smooth_window=smooth_window,
                    progress_cb=lambda v: self.after(0, self._progress_set, v),
                    stop_event=self._stop_event,
                    log_cb=self._log_write_safe,
                )

                if self._stop_event.is_set():
                    self.after(0, self._on_stopped)
                else:
                    self._result = result
                    self.after(0, self._on_done, result)

            except Exception:
                tb = traceback.format_exc()
                self.after(0, self._on_error, tb)

        threading.Thread(target=worker, daemon=True).start()

    def _on_stop(self):
        self._stop_event.set()
        self._btn_stop.configure(state="disabled")
        self._log_write("Stopping after current iteration...")

    def _on_stopped(self):
        self._progress["value"] = 0
        self._log_write("Search stopped by user.")
        self._btn_run.configure(state="normal")
        self._btn_stop.configure(state="disabled")

    def _on_done(self, result: SearchResult):
        self._progress["value"] = 100
        self._log_write("-" * 60)
        self._log_write("RESULT:")
        self._log_write(f"  Start:         Lon={result.start_lon:.8f}  Lat={result.start_lat:.8f}")
        self._log_write(f"  Base bearing:  {result.bearing:.4f}")
        self._log_write(f"  Perp offset:   {result.perp_offset_m:.3f} m")
        self._log_write(f"  Along shift:   {result.best_shift_points} pts")
        self._log_write(f"  RMSE:          {result.rmse:.4f} m")
        self._log_write(f"  MAE:           {result.mae:.4f} m")
        self._log_write(f"  Pearson:       {result.corr:.4f}")
        self._log_write(f"  Matched:       {result.matched} / {len(result.points)} points")
        self._log_write("-" * 60)
        self._btn_run.configure(state="normal")
        self._btn_stop.configure(state="disabled")
        self._btn_csv.configure(state="normal")
        self._btn_geojson.configure(state="normal")

    def _on_error(self, tb: str):
        self._log_write("ERROR:\n" + tb)
        self._btn_run.configure(state="normal")
        self._btn_stop.configure(state="disabled")
        last_line = [l for l in tb.strip().splitlines() if l.strip()][-1]
        messagebox.showerror("Error", last_line)

    def _save_csv(self):
        if not self._result:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")],
            initialfile="recovered_points.csv"
        )
        if path:
            try:
                write_result_csv(path, self._result)
                self._log_write(f"Saved CSV: {path}")
            except Exception:
                tb = traceback.format_exc()
                self._log_write("SAVE CSV ERROR:\n" + tb)
                messagebox.showerror("Save CSV error", tb.splitlines()[-1])

    def _save_geojson(self):
        if not self._result:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".geojson",
            filetypes=[("GeoJSON", "*.geojson"), ("All", "*.*")],
            initialfile="recovered_points.geojson"
        )
        if path:
            try:
                write_result_geojson(path, self._result)
                self._log_write(f"Saved GeoJSON: {path}")
            except Exception:
                tb = traceback.format_exc()
                self._log_write("SAVE GEOJSON ERROR:\n" + tb)
                messagebox.showerror("Save GeoJSON error", tb.splitlines()[-1])


if __name__ == "__main__":
    app = App()
    app.mainloop()
