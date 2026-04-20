"""
app.py — ProfileFinder GUI (tkinter)

Layout:
  [DSM path]         [Browse]
  [CSV path]         [Browse]
  Start Lon/Lat      [fields]
  Search radius (m)  [field]
  Point step (m)     [field]
  Coarse grid (m)    [field]
  Smooth window      [field]
  [Run]  [Save CSV]  [Save GeoJSON]
  Progress bar
  Log / results panel
"""

import csv
import json
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from core import load_profile, run_search, SearchResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _browse_file(var: tk.StringVar, filetypes):
    path = filedialog.askopenfilename(filetypes=filetypes)
    if path:
        var.set(path)


def _save_file(var: tk.StringVar, filetypes, default_ext):
    path = filedialog.asksaveasfilename(filetypes=filetypes, defaultextension=default_ext)
    if path:
        var.set(path)


def write_result_csv(path: str, result: SearchResult):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Id", "Lon", "Lat"])
        for i, (lon, lat) in enumerate(result.points, start=1):
            w.writerow([i, round(lon, 8), round(lat, 8)])


def write_result_geojson(path: str, result: SearchResult):
    features = []
    for i, (lon, lat) in enumerate(result.points, start=1):
        features.append({
            "type": "Feature",
            "properties": {"Id": i},
            "geometry": {"type": "Point", "coordinates": [round(lon, 8), round(lat, 8)]}
        })
    fc = {"type": "FeatureCollection", "features": features}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ProfileFinder")
        self.resizable(False, False)
        self._result: SearchResult | None = None
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        pad = {"padx": 8, "pady": 4}

        # ── File inputs ────────────────────────────────────────────────
        frame_files = ttk.LabelFrame(self, text="Inputs")
        frame_files.grid(row=0, column=0, sticky="ew", **pad)

        self._dsm_var = tk.StringVar()
        self._csv_var = tk.StringVar()

        ttk.Label(frame_files, text="DSM (.tif):").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(frame_files, textvariable=self._dsm_var, width=48).grid(row=0, column=1, **pad)
        ttk.Button(frame_files, text="Browse",
                   command=lambda: _browse_file(self._dsm_var,
                                                [("GeoTIFF", "*.tif *.tiff"), ("All", "*.*")])
                   ).grid(row=0, column=2, **pad)

        ttk.Label(frame_files, text="Profile CSV:").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(frame_files, textvariable=self._csv_var, width=48).grid(row=1, column=1, **pad)
        ttk.Button(frame_files, text="Browse",
                   command=lambda: _browse_file(self._csv_var,
                                                [("CSV", "*.csv"), ("All", "*.*")])
                   ).grid(row=1, column=2, **pad)

        # ── Parameters ────────────────────────────────────────────────
        frame_params = ttk.LabelFrame(self, text="Parameters")
        frame_params.grid(row=1, column=0, sticky="ew", **pad)

        params = [
            ("Start Longitude (°):",  "start_lon",      "0.0"),
            ("Start Latitude (°):",   "start_lat",      "0.0"),
            ("Search radius (m):",    "search_radius",  "5000"),
            ("Point step (m):",       "step_m",         "5"),
            ("Coarse grid (m):",      "coarse_grid",    "60"),
            ("Smooth window:",        "smooth_window",  "5"),
            ("CSV Bearing field:",    "bearing_field",  "Bearing"),
            ("CSV Z_DSM field:",      "z_field",        "Z_DSM"),
        ]
        self._param_vars = {}
        for i, (label, key, default) in enumerate(params):
            ttk.Label(frame_params, text=label).grid(row=i, column=0, sticky="w", **pad)
            var = tk.StringVar(value=default)
            ttk.Entry(frame_params, textvariable=var, width=20).grid(row=i, column=1, sticky="w", **pad)
            self._param_vars[key] = var

        # ── Run button ────────────────────────────────────────────────
        frame_run = ttk.Frame(self)
        frame_run.grid(row=2, column=0, **pad)

        self._btn_run = ttk.Button(frame_run, text="▶  Run Search", command=self._on_run)
        self._btn_run.grid(row=0, column=0, padx=4)

        self._btn_csv = ttk.Button(frame_run, text="Save CSV", state="disabled",
                                   command=self._save_csv)
        self._btn_csv.grid(row=0, column=1, padx=4)

        self._btn_geojson = ttk.Button(frame_run, text="Save GeoJSON", state="disabled",
                                       command=self._save_geojson)
        self._btn_geojson.grid(row=0, column=2, padx=4)

        # ── Progress ──────────────────────────────────────────────────
        self._progress = ttk.Progressbar(self, length=540, mode="determinate")
        self._progress.grid(row=3, column=0, **pad)

        # ── Log ───────────────────────────────────────────────────────
        frame_log = ttk.LabelFrame(self, text="Log")
        frame_log.grid(row=4, column=0, sticky="nsew", **pad)

        self._log = tk.Text(frame_log, height=12, width=70, state="disabled",
                            font=("Consolas", 9))
        self._log.grid(row=0, column=0)
        scrollbar = ttk.Scrollbar(frame_log, command=self._log.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self._log.configure(yscrollcommand=scrollbar.set)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_write(self, msg: str):
        self._log.configure(state="normal")
        self._log.insert("end", msg + "\n")
        self._log.see("end")
        self._log.configure(state="disabled")

    def _progress_set(self, value: float):
        self._progress["value"] = value * 100
        self.update_idletasks()

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

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
            start_lon = float(self._param_vars["start_lon"].get())
            start_lat = float(self._param_vars["start_lat"].get())
            search_radius = float(self._param_vars["search_radius"].get())
            step_m = float(self._param_vars["step_m"].get())
            coarse_grid = float(self._param_vars["coarse_grid"].get())
            smooth_window = int(self._param_vars["smooth_window"].get())
            bearing_field = self._param_vars["bearing_field"].get().strip()
            z_field = self._param_vars["z_field"].get().strip()
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid parameter: {e}")
            return

        self._btn_run.configure(state="disabled")
        self._btn_csv.configure(state="disabled")
        self._btn_geojson.configure(state="disabled")
        self._progress["value"] = 0
        self._log.configure(state="normal")
        self._log.delete("1.0", "end")
        self._log.configure(state="disabled")
        self._result = None

        def worker():
            try:
                self._log_write("Loading profile CSV...")
                profile = load_profile(csv_path, bearing_field, z_field)
                self._log_write(f"  {len(profile)} points loaded.")
                self._log_write(f"Starting search from ({start_lat:.6f}, {start_lon:.6f}), "
                                f"radius={search_radius} m, step={step_m} m...")

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
                )

                self._result = result
                self.after(0, self._on_done, result)

            except Exception as e:
                self.after(0, self._on_error, str(e))

        threading.Thread(target=worker, daemon=True).start()

    def _on_done(self, result: SearchResult):
        self._progress["value"] = 100
        self._log_write("─" * 50)
        self._log_write(f"RESULT:")
        self._log_write(f"  Start:   Lon={result.start_lon:.8f}  Lat={result.start_lat:.8f}")
        self._log_write(f"  RMSE:    {result.rmse:.4f} m")
        self._log_write(f"  MAE:     {result.mae:.4f} m")
        self._log_write(f"  Pearson: {result.corr:.4f}")
        self._log_write(f"  Matched: {result.matched} / {len(result.points)} points")
        self._log_write("─" * 50)
        self._btn_run.configure(state="normal")
        self._btn_csv.configure(state="normal")
        self._btn_geojson.configure(state="normal")

    def _on_error(self, msg: str):
        self._log_write(f"ERROR: {msg}")
        self._btn_run.configure(state="normal")
        messagebox.showerror("Error", msg)

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------

    def _save_csv(self):
        if not self._result:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")],
            initialfile="recovered_points.csv"
        )
        if path:
            write_result_csv(path, self._result)
            self._log_write(f"Saved CSV: {path}")

    def _save_geojson(self):
        if not self._result:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".geojson",
            filetypes=[("GeoJSON", "*.geojson"), ("All", "*.*")],
            initialfile="recovered_points.geojson"
        )
        if path:
            write_result_geojson(path, self._result)
            self._log_write(f"Saved GeoJSON: {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = App()
    app.mainloop()
