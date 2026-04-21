import arcpy
from arcpy import da
import random
import os
import csv
import math

# ============================================================
# ПАРАМЕТРЫ ИНСТРУМЕНТА
# 0  Lines     Feature Class  Input   Required
# 1  Interval  Double         Input   Required   default: 5
# 2  Error     Double         Input   Required   default: 0
# 3  DSM       Raster Layer   Input   Required
# 4  Error_Z   Double         Input   Optional   default: 0
# 5  Points    Feature Class  Output  Derived
# 6  CSV       File           Output  Optional
# ============================================================
LINE_FC  = arcpy.GetParameterAsText(0)
INTERVAL = float(arcpy.GetParameter(1))
JITTER   = float(arcpy.GetParameter(2))
DSM_PATH = arcpy.GetParameterAsText(3)
_error_z_raw = arcpy.GetParameterAsText(4)
ERROR_Z  = float(_error_z_raw) if _error_z_raw not in ("", None) else 0.0
OUT_FC   = arcpy.GetParameterAsText(5)
OUT_CSV  = arcpy.GetParameterAsText(6)

SR = arcpy.SpatialReference(4326)


def geodesic_bearing(pt_from, pt_to):
    p1 = pt_from.firstPoint
    p2 = pt_to.firstPoint
    dlon = math.radians(p2.X - p1.X)
    lat1 = math.radians(p1.Y)
    lat2 = math.radians(p2.Y)
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def geodesic_dist(pt_from, pt_to):
    seg = arcpy.Polyline(arcpy.Array([pt_from.firstPoint, pt_to.firstPoint]), SR)
    return seg.getLength("GEODESIC", "METERS")


def sample_dsm(pt_geom):
    pt = pt_geom.firstPoint
    location = f"{pt.X} {pt.Y}"
    result = arcpy.management.GetCellValue(DSM_PATH, location, "1")
    val = result.getOutput(0)
    if val in ("NoData", "", None):
        return None
    return float(val.replace(",", "."))


# --- Определяем поле name ---
line_fields = [f.name for f in arcpy.ListFields(LINE_FC)]
name_field_actual = next((f for f in line_fields if f.lower() == "name"), None)
has_name = name_field_actual is not None

# --- Создаём выходной FC ---
ws, fc_name = OUT_FC.rsplit("\\", 1)
if arcpy.Exists(OUT_FC):
    arcpy.management.Delete(OUT_FC)

arcpy.management.CreateFeatureclass(ws, fc_name, "POINT", spatial_reference=SR)
arcpy.management.AddField(OUT_FC, "name",      "TEXT",   field_length=255)
arcpy.management.AddField(OUT_FC, "PT_ORDER",  "LONG")
arcpy.management.AddField(OUT_FC, "DIST_PREV", "DOUBLE")
arcpy.management.AddField(OUT_FC, "BEAR_PREV", "DOUBLE")
arcpy.management.AddField(OUT_FC, "Z_DSM",     "DOUBLE")

search_fields = ["SHAPE@", name_field_actual] if has_name else ["SHAPE@"]
out_fields    = ["SHAPE@", "name", "PT_ORDER", "DIST_PREV", "BEAR_PREV", "Z_DSM"]

csv_rows  = []
total_pts = 0

with da.SearchCursor(LINE_FC, search_fields) as s_cur:
    with da.InsertCursor(OUT_FC, out_fields) as i_cur:
        for row in s_cur:
            geom = row[0]
            name = row[1] if has_name else ""
            line_sr = geom.spatialReference

            total_length = geom.getLength("GEODESIC", "METERS")
            if total_length <= 0:
                continue

            # Масштаб: метры → единицы SR линии
            total_length_map = geom.length
            scale = total_length_map / total_length

            # Расстояния в метрах
            distances = [0.0]
            d = 0.0
            while True:
                step = INTERVAL + random.uniform(-JITTER, JITTER) if JITTER > 0 else INTERVAL
                d += max(0.1, step)
                if d >= total_length:
                    break
                distances.append(d)
            distances.append(total_length)

            # Точки вдоль линии (в единицах SR линии)
            pts_geom = [geom.positionAlongLine(d * scale) for d in distances]

            prev_pt = None
            for idx, pt_geom in enumerate(pts_geom):
                # Переводим в WGS84 для bearing/dist/DSM
                pt_wgs = pt_geom.projectAs(SR) if line_sr.factoryCode != SR.factoryCode else pt_geom

                if prev_pt is None:
                    dist_prev, bear_prev = 0.0, 0.0
                else:
                    dist_prev = round(geodesic_dist(prev_pt, pt_wgs), 4)
                    bear_prev = round(geodesic_bearing(prev_pt, pt_wgs), 4)

                z_raw = sample_dsm(pt_wgs)
                if z_raw is not None and ERROR_Z > 0:
                    z_dsm = round(z_raw + random.uniform(-ERROR_Z, ERROR_Z), 4)
                else:
                    z_dsm = z_raw

                i_cur.insertRow([pt_wgs, name, idx, dist_prev, bear_prev, z_dsm])
                csv_rows.append({
                    "name":      name,
                    "PT_ORDER":  idx,
                    "DIST_PREV": dist_prev,
                    "BEAR_PREV": bear_prev,
                    "Z_DSM":     z_dsm if z_dsm is not None else "",
                })
                prev_pt = pt_wgs
                total_pts += 1

# --- CSV ---
if OUT_CSV:
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "PT_ORDER", "DIST_PREV", "BEAR_PREV", "Z_DSM"], delimiter=";")
        writer.writeheader()
        writer.writerows(csv_rows)

arcpy.SetParameter(5, OUT_FC)
arcpy.AddMessage(f"Done. Points: {total_pts}")
