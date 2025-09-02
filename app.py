
# ICEYE Coverage Viewer — Streamlit + Folium
# ------------------------------------------
# - Two fixed logos in the header (ats_logo.png & fapco.png)
# - Single date picker (default = 01-09-2025)
# - Folium map renders immediately (AOI + full swaths + S1 layers)
# - Swaths are SHOWN fully (not clipped to AOI); coverage is computed vs AOI
# - One toggleable layer per swath (Layer Control)
# - Sidebar checkboxes control which sensors contribute to coverage stats
# - Sentinel-1 reference coverage (S1A/S1C) using 12-day phase logic:
#     shows phase-matched subset for chosen date if present; otherwise shows ALL footprints as fallback
# - Tables: per-frame details and per-satellite/sensor summary

from pathlib import Path
import json
import re
from datetime import date, datetime, timedelta, timezone

import streamlit as st
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union
from shapely.geometry.base import BaseGeometry

# Optional geodesic area (accurate); will fall back to planar approx if missing
try:
    from pyproj import Geod
    GEOD = Geod(ellps="WGS84")
except Exception:
    GEOD = None

# Optional Shapely make_valid (Shapely>=2)
try:
    from shapely.make_valid import make_valid
except Exception:
    make_valid = None

st.set_page_config(page_title="ICEYE Coverage Planner", layout="wide")

# --------------------------- Constants ---------------------------

DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")

SENSOR_KEYS = [
    ("TOPSAR WIDE", "LEFT"),
    ("TOPSAR WIDE", "RIGHT"),
    ("TOPSAR", "LEFT"),
    ("TOPSAR", "RIGHT"),
]

SENSOR_LABELS = {
    ("TOPSAR", "LEFT"): "TopSAR (Left)",
    ("TOPSAR", "RIGHT"): "TopSAR (Right)",
    ("TOPSAR WIDE", "LEFT"): "TopSAR Wide (Left)",
    ("TOPSAR WIDE", "RIGHT"): "TopSAR Wide (Right)",
}

SENSOR_COLORS = {
    ("TOPSAR", "LEFT"): "#1f77b4",
    ("TOPSAR", "RIGHT"): "#ff7f0e",
    ("TOPSAR WIDE", "LEFT"): "#2ca02c",
    ("TOPSAR WIDE", "RIGHT"): "#d62728",
}

S1_COLORS = {
    "S1A": "#8c564b",
    "S1C": "#9467bd",
}

# --------------------------- Helpers ---------------------------

def _base64_of(path: Path):
    import base64
    if path and path.exists():
        return base64.b64encode(path.read_bytes()).decode("utf-8")
    return None

def _mk_valid(g):
    if g is None:
        return None
    if make_valid is not None:
        try:
            return make_valid(g)
        except Exception:
            pass
    try:
        g2 = g.buffer(0)
        if g2.is_valid:
            return g2
    except Exception:
        pass
    return g

def _empty_gdf():
    # Create an empty GeoDataFrame that already has a 'geometry' column.
    return gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326").set_geometry("geometry")

def _geodesic_area_m2(geom: BaseGeometry) -> float:
    """Geodesic area on WGS84 in m^2; falls back to crude approx if pyproj missing."""
    if geom is None or geom.is_empty:
        return 0.0
    try:
        if GEOD is not None:
            if geom.geom_type == "Polygon":
                lon, lat = geom.exterior.coords.xy
                area, _ = GEOD.polygon_area_perimeter(lon, lat)
                total = abs(area)
                for ring in geom.interiors:
                    lon_i, lat_i = ring.coords.xy
                    a_i, _ = GEOD.polygon_area_perimeter(lon_i, lat_i)
                    total -= abs(a_i)
                return abs(total)
            elif geom.geom_type in ("MultiPolygon", "GeometryCollection"):
                return sum(_geodesic_area_m2(g) for g in geom.geoms if isinstance(g, BaseGeometry))
    except Exception:
        pass
    # Fallback approximate (degrees to km)
    approx_km2 = geom.area * (111.32**2)
    return approx_km2 * 1_000_000.0 / 1_000_000.0

def _load_gdf_from_path(label: str, try_paths):
    """Load a GeoJSON strictly from disk (no uploader UI)."""
    if isinstance(try_paths, (str, Path)):
        try_paths = [try_paths]
    paths = [Path(p) for p in try_paths]
    hit = next((p for p in paths if p.exists()), None)
    if hit is None:
        st.error(f"{label} not found. Tried: " + ", ".join(str(p) for p in paths))
        return _empty_gdf()
    try:
        gj = json.loads(hit.read_text(encoding="utf-8"))
        feats = gj.get("features", [])
        gdf = gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326")
        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True)
        else:
            gdf = gdf.to_crs(epsg=4326)
        gdf["geometry"] = gdf["geometry"].apply(_mk_valid)
        gdf = gdf[~gdf["geometry"].is_empty & gdf["geometry"].notna()].reset_index(drop=True)
        return gdf
    except Exception as e:
        st.error(f"Failed to parse {label} at {hit}: {e}")
        return _empty_gdf()

def _extract_dates_from_row(row, columns):
    dates = set()
    for col in columns:
        if col == "geometry":
            continue
        v = row.get(col)
        if v is None:
            continue
        s = v if isinstance(v, str) else str(v)
        for iso in DATE_RE.findall(s):
            dates.add(iso)
    return sorted(dates)

def _same_phase(chosen_dt: datetime, ref_dt: datetime, period_days: int) -> bool:
    return ((chosen_dt.date() - ref_dt.date()).days % period_days) == 0

def _get_ref_date(chosen_dt: datetime, reference_dates, period_days: int):
    """Pick the phase-aligned reference date <= chosen date wherever possible."""
    if not reference_dates:
        return None
    matches = [r for r in reference_dates if _same_phase(chosen_dt, r, period_days)]
    if matches:
        matches.sort()
        for r in reversed(matches):
            if r <= chosen_dt:
                return r
        return matches[0]
    anchor = max([r for r in reference_dates if r <= chosen_dt], default=min(reference_dates))
    shift = (chosen_dt.date() - anchor.date()).days % period_days
    out = (chosen_dt - timedelta(days=shift)).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
    return out

@st.cache_data(show_spinner=False)
def _load_reference(try_paths):
    """Return (gdf, sorted_unique_datesUTC) for S1 reference plan (no uploader).
    Collect date tokens from ANY non-geometry column (robust to different exports)."""
    if isinstance(try_paths, (str, Path)):
        try_paths = [try_paths]
    paths = [Path(p) for p in try_paths]
    hit = next((p for p in paths if p.exists()), None)
    if hit is None:
        return _empty_gdf(), []
    gj = json.loads(hit.read_text(encoding="utf-8"))
    feats = gj.get("features", [])
    gdf = gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326")
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)
    else:
        gdf = gdf.to_crs(epsg=4326)
    gdf["geometry"] = gdf["geometry"].apply(_mk_valid)

    seen = set()
    for _, row in gdf.iterrows():
        for col in gdf.columns:
            if col == "geometry":
                continue
            v = row.get(col)
            if v is None:
                continue
            s = v if isinstance(v, str) else str(v)
            for iso in DATE_RE.findall(s):
                try:
                    seen.add(datetime.strptime(iso, "%Y-%m-%d").replace(tzinfo=timezone.utc))
                except Exception:
                    pass
    return gdf, sorted(seen)

def _s1_phase_subset(gdf: gpd.GeoDataFrame, ref_dates, period_days: int, chosen: datetime):
    """Return (subset_gdf, ref_ymd) for the chosen date phasing, or (empty, None) if no exact match."""
    if gdf.empty or not ref_dates:
        return _empty_gdf(), None
    ref_dt = _get_ref_date(chosen, ref_dates, period_days)
    if ref_dt is None:
        return _empty_gdf(), None
    ref_ymd = ref_dt.strftime("%Y-%m-%d")
    mask = []
    cols = list(gdf.columns)
    for _, row in gdf.iterrows():
        has = ref_ymd in _extract_dates_from_row(row, cols)
        mask.append(has)
    sub = gdf[pd.Series(mask, index=gdf.index)]
    return sub.reset_index(drop=True), ref_ymd

# --------------------------- Header (two fixed logos) ---------------------------

def header_two_logos():
    left, mid, right = st.columns([1.2, 5, 1.2])

    def _find_logo(*names):
        for n in names:
            p = Path(n)
            if p.exists():
                return p
        for n in names:  # fallback to /mnt/data
            p = Path("/mnt/data") / n
            if p.exists():
                return p
        return None

    with left:
        p_left = _find_logo("ats_logo.png")
        if p_left:
            b64 = _base64_of(p_left)
            st.markdown(f"<img src='data:image/png;base64,{b64}' style='height:100px;'>",
                        unsafe_allow_html=True)
    with mid:
        st.markdown("<h2 style='text-align:center;margin:6px 0;'>ICEYE Daily Swath Planner & S1 Reference Coverage</h2>",
                    unsafe_allow_html=True)
    with right:
        p_right = _find_logo("fapco.png")
        if p_right:
            b64r = _base64_of(p_right)
            st.markdown(f"<img src='data:image/png;base64,{b64r}' style='height:100px;float:right;'>",
                        unsafe_allow_html=True)

header_two_logos()

# --------------------------- Inputs ---------------------------

st.sidebar.header("Inputs")

# Single-date picker (default 01-09-2025)
chosen_date = st.sidebar.date_input("Pick a date", value=date(2025, 9, 1), key="single_date")
if isinstance(chosen_date, (list, tuple)):  # just in case
    chosen_date = chosen_date[0]
chosen_ymd = chosen_date.strftime("%Y-%m-%d")
chosen_dt = datetime.strptime(chosen_ymd, "%Y-%m-%d").replace(tzinfo=timezone.utc)

# Load data (no uploaders; read from local or /mnt/data fallback)
AOI_PATHS  = ["AOI.geojson",  "AOI.geojson"]
SWATHS_PATHS = ["fapco_3months.geojson", "fapco_3months.geojson"]
S1A_PATHS = ["S1A_12day_reference_coverage_plan.geojson", "S1A_12day_reference_coverage_plan.geojson"]
S1C_PATHS = ["S1C_12day_reference_coverage_plan.geojson", "S1C_12day_reference_coverage_plan.geojson"]

aoi_gdf  = _load_gdf_from_path("AOI", AOI_PATHS)
swaths_gdf = _load_gdf_from_path("ICEYE Swaths", SWATHS_PATHS)
s1a_gdf, s1a_dates = _load_reference(S1A_PATHS)
s1c_gdf, s1c_dates = _load_reference(S1C_PATHS)

# Sensor toggles (these control coverage stats; map layers are independent toggles)
st.sidebar.subheader("Sensors to include in coverage (ICEYE)")
sensor_flags = {}
for key in [("TOPSAR", "LEFT"), ("TOPSAR", "RIGHT"), ("TOPSAR WIDE", "LEFT"), ("TOPSAR WIDE", "RIGHT")]:
    sensor_flags[key] = st.sidebar.checkbox(SENSOR_LABELS[key], value=True, key=f"chk_{key}")

st.sidebar.subheader("Sentinel-1 reference")
show_s1a = st.sidebar.checkbox("Show S1A coverage", value=True)
show_s1c = st.sidebar.checkbox("Show S1C coverage", value=True)

st.sidebar.divider()
st.sidebar.info("Layer Control lets you show/hide individual swaths. "
                "Coverage % is driven by the ICEYE checkboxes above.")

# --------------------------- Filter ICEYE swaths by date (no clipping) ---------------------------

def _filter_swaths_by_date_and_sensor(gdf: gpd.GeoDataFrame, ymd: str, aoi_union: BaseGeometry):
    if gdf.empty:
        return {k: _empty_gdf() for k in SENSOR_KEYS}

    if "properties" not in gdf.columns:
        gdf = gdf.copy()
        gdf["properties"] = [{} for _ in range(len(gdf))]

    # Build date mask
    mask_date = []
    for _, row in gdf.iterrows():
        props = row.get("properties", {})
        blob = " ".join([str(v) for v in getattr(props, "values", lambda: [])() if v is not None])
        for col in gdf.columns:
            if col in ("geometry", "properties"):
                continue
            val = row.get(col)
            if val is not None:
                blob += " " + str(val)
        found = any(hit == ymd for hit in DATE_RE.findall(blob))
        if not found:
            s = row.get("start") or row.get("Start") or row.get("acq_date")
            if s and str(s)[:10] == ymd:
                found = True
        mask_date.append(found)

    gdf_d = gdf[pd.Series(mask_date, index=gdf.index)]
    if gdf_d.empty:
        return {k: _empty_gdf() for k in SENSOR_KEYS}

    # Keep only swaths that intersect AOI (do not clip)
    if aoi_union and not aoi_union.is_empty:
        gdf_d = gdf_d[gdf_d.intersects(aoi_union)]
        if gdf_d.empty:
            return {k: _empty_gdf() for k in SENSOR_KEYS}

    # Classify by sensor (bucket by index)
    def _classify(props: dict):
        blob = " ".join([f"{k}={v}" for k, v in (props or {}).items() if v is not None]).lower()
        mode = None
        if any(w in blob for w in ("topsar wide", "topsar_wide", "topsar-wide", "topsarw")):
            mode = "TOPSAR WIDE"
        elif "topsar" in blob:
            mode = "TOPSAR"
        look = None
        if "left" in blob:
            look = "LEFT"
        elif "right" in blob:
            look = "RIGHT"
        for k in ("look", "lookdirection", "side", "direction"):
            v = (props or {}).get(k) or (props or {}).get(k.title()) or (props or {}).get(k.upper())
            if isinstance(v, str) and not look:
                if v.strip().lower().startswith("l"):
                    look = "LEFT"
                elif v.strip().lower().startswith("r"):
                    look = "RIGHT"
        return (mode, look) if (mode, look) in SENSOR_LABELS else (None, None)

    buckets = {k: [] for k in SENSOR_KEYS}
    for idx, row in gdf_d.iterrows():
        mode, look = _classify(row.get("properties", {}))
        if (mode, look) not in buckets:
            props2 = dict(row.get("properties", {}))
            for c in gdf_d.columns:
                if c not in ("geometry", "properties"):
                    props2[c] = row.get(c)
            mode, look = _classify(props2)
        if (mode, look) in buckets:
            buckets[(mode, look)].append(idx)

    out = {}
    for key in SENSOR_KEYS:
        idxs = buckets.get(key, [])
        out[key] = gdf_d.loc[idxs].copy() if idxs else _empty_gdf()
    return out

aoi_union = aoi_gdf.unary_union if not aoi_gdf.empty else None
sensor_gdfs = _filter_swaths_by_date_and_sensor(swaths_gdf, chosen_ymd, aoi_union)
no_swaths_today = all(gdf.empty for gdf in sensor_gdfs.values())

# --------------------------- S1 phase subset or fallback ---------------------------

s1a_sub, s1a_ref = _s1_phase_subset(s1a_gdf, s1a_dates, 12, chosen_dt)
s1c_sub, s1c_ref = _s1_phase_subset(s1c_gdf, s1c_dates, 12, chosen_dt)

# --------------------------- Map ---------------------------

m = folium.Map(location=[25, 45], zoom_start=4, tiles=None, control_scale=True)
folium.TileLayer("OpenStreetMap", name="OpenStreetMap", show=True).add_to(m)

# AOI Layer
if not aoi_gdf.empty:
    folium.GeoJson(
        aoi_gdf.__geo_interface__,
        name="AOI",
        style_function=lambda x: {"color": "#f1c40f", "weight": 2, "fill": True, "fillOpacity": 0.08},
        highlight_function=lambda x: {"weight": 3, "color": "#f39c12"},
        tooltip=folium.GeoJsonTooltip(fields=[], aliases=[], labels=False)
    ).add_to(m)

# ICEYE swaths — draw FULL swaths, one layer per swath; compute coverage separately
coverage_union = None
detail_rows = []

for key in SENSOR_KEYS:
    label = SENSOR_LABELS[key]
    color = SENSOR_COLORS[key]
    sub = sensor_gdfs.get(key, _empty_gdf())
    if sub.empty:
        continue

    # Draw each swath as its own toggleable layer
    for i, row in sub.iterrows():
        geom_json = row.geometry.__geo_interface__
        props = row.get("properties", {}) if "properties" in sub.columns else {}
        sat = props.get("satellite") or props.get("sat") or props.get("platform") or "ICEYE"
        fid = props.get("frame_id") or props.get("id") or f"frame_{i}"
        dates = _extract_dates_from_row(row, list(sub.columns))
        date_str = dates[0] if dates else ""
        layer_name = f"ICEYE • {label} • {fid}" + (f" • {date_str}" if date_str else "")
        folium.GeoJson(
            geom_json,
            name=layer_name,
            style_function=lambda x, c=color: {"color": c, "fillColor": c, "weight": 1, "fillOpacity": 0.25},
            highlight_function=lambda x: {"weight": 2},
        ).add_to(m)

        # If this sensor is enabled in the sidebar, include in coverage stats
        if sensor_flags.get(key, False) and aoi_union and not aoi_union.is_empty:
            try:
                inter = row.geometry.intersection(aoi_union)
            except Exception:
                inter = None
            if inter and not inter.is_empty:
                area_km2 = _geodesic_area_m2(inter) / 1_000_000.0
                detail_rows.append({
                    "Satellite": sat,
                    "Sensor": label,
                    "Frame_ID": fid,
                    "Date": date_str or chosen_ymd,
                    "Area_km²_in_AOI": round(area_km2, 3)
                })
                coverage_union = inter if coverage_union is None else coverage_union.union(inter)

# Sentinel-1 layers — only swaths intersecting AOI (one layer per swath) + tables
if aoi_union and (not aoi_union.is_empty):
    # S1A
    if show_s1a and (not s1a_gdf.empty):
        src = s1a_sub if (not s1a_sub.empty and s1a_ref) else s1a_gdf
        cols = list(src.columns)
        for i, row in src.iterrows():
            try:
                inter = row.geometry.intersection(aoi_union)
            except Exception:
                inter = None
            if inter is None or inter.is_empty:
                continue  # skip non-intersecting swaths
            # Map layer (draw FULL swath geometry, not clipped)
            geom_json = row.geometry.__geo_interface__
            dates = _extract_dates_from_row(row, cols)
            label_date = s1a_ref if (not s1a_sub.empty and s1a_ref) else (dates[0] if dates else "")
            fid = row.get("id") or row.get("frame_id") or row.get("name") or f"S1A"
            folium.GeoJson(
                geom_json,
                name=f"S1A",
                style_function=lambda x: {"color": S1_COLORS["S1A"], "fillColor": S1_COLORS["S1A"], "weight": 1, "fillOpacity": 0.15},
                highlight_function=lambda x: {"weight": 2},
            ).add_to(m)
            # Table entry
            area_km2 = _geodesic_area_m2(inter) / 1_000_000.0
            detail_rows.append({
                "Satellite": "Sentinel-1A",
                "Sensor": "Reference (12d)",
                "Frame_ID": fid,
                "Date": label_date or chosen_ymd,
                "Area_km²_in_AOI": round(area_km2, 3)
            })

    # S1C
    if show_s1c and (not s1c_gdf.empty):
        src = s1c_sub if (not s1c_sub.empty and s1c_ref) else s1c_gdf
        cols = list(src.columns)
        for i, row in src.iterrows():
            try:
                inter = row.geometry.intersection(aoi_union)
            except Exception:
                inter = None
            if inter is None or inter.is_empty:
                continue  # skip non-intersecting swaths
            # Map layer (draw FULL swath geometry, not clipped)
            geom_json = row.geometry.__geo_interface__
            dates = _extract_dates_from_row(row, cols)
            label_date = s1c_ref if (not s1c_sub.empty and s1c_ref) else (dates[0] if dates else "")
            fid = row.get("id") or row.get("frame_id") or row.get("name") or f"S1C"
            folium.GeoJson(
                geom_json,
                name=f"S1C",
                style_function=lambda x: {"color": S1_COLORS["S1C"], "fillColor": S1_COLORS["S1C"], "weight": 1, "fillOpacity": 0.15},
                highlight_function=lambda x: {"weight": 2},
            ).add_to(m)
            # Table entry
            area_km2 = _geodesic_area_m2(inter) / 1_000_000.0
            detail_rows.append({
                "Satellite": "Sentinel-1C",
                "Sensor": "Reference (12d)",
                "Frame_ID": fid,
                "Date": label_date or chosen_ymd,
                "Area_km²_in_AOI": round(area_km2, 3)
            })
else:
    # No AOI => do not draw S1 layers (require AOI intersection)
    pass

# Fit map to AOI
if not aoi_gdf.empty:
    minx, miny, maxx, maxy = aoi_gdf.total_bounds
    m.fit_bounds([[miny, minx], [maxy, maxx]])

folium.LayerControl(collapsed=False).add_to(m)

if no_swaths_today:
    st.info("No ICEYE swaths found for this exact date. AOI and Sentinel-1 layers still load; try another date.")

# Render Folium map
_ = st_folium(m, height=740, use_container_width=True)

# --------------------------- Coverage stats + tables ---------------------------

st.markdown("### Coverage & Tables")

if aoi_union and coverage_union and (not coverage_union.is_empty):
    aoi_area_km2 = _geodesic_area_m2(aoi_union) / 1_000_000.0
    covered_area_km2 = _geodesic_area_m2(aoi_union.intersection(coverage_union)) / 1_000_000.0
    pct = (covered_area_km2 / aoi_area_km2) * 100 if aoi_area_km2 > 0 else 0.0
    st.success(f"**Covered area:** {covered_area_km2:,.2f} km² of {aoi_area_km2:,.2f} km² (**{pct:.2f}%**) — from enabled sensors.")
else:
    st.info("No enabled sensor layers intersect the AOI on this date.")

# Detail table (per frame) + summary
if detail_rows:
    df_detail = pd.DataFrame(detail_rows)

    # Normalize non-ASCII column name for robust ops
    if "Area_km²_in_AOI" in df_detail.columns:
        df_detail = df_detail.rename(columns={"Area_km²_in_AOI": "Area_km2_in_AOI"})
    df_detail["Area_km2_in_AOI"] = pd.to_numeric(df_detail.get("Area_km2_in_AOI"), errors="coerce").fillna(0.0)

    # Summary by satellite & sensor (version-proof)
    try:
        df_summary = (df_detail
                      .groupby(["Satellite", "Sensor"], dropna=False, as_index=False)
                      .agg(Frames=("Frame_ID", "count"),
                           Area_km2=("Area_km2_in_AOI", "sum"))
                      .sort_values(["Satellite", "Sensor"]))
    except Exception:
        gb = df_detail.groupby(["Satellite", "Sensor"], dropna=False)
        df_summary = (gb["Frame_ID"].count().to_frame("Frames")
                      .join(gb["Area_km2_in_AOI"].sum().to_frame("Area_km2"))
                      .reset_index()
                      .sort_values(["Satellite", "Sensor"]))

    with st.expander("Per-frame details"):
        st.dataframe(df_detail.rename(columns={"Area_km2_in_AOI": "Area_km²_in_AOI"}), use_container_width=True)
    with st.expander("Per-satellite & sensor summary"):
        st.dataframe(df_summary.rename(columns={"Area_km2": "Area_km²"}), use_container_width=True)
else:
    st.info("No per-frame intersections to tabulate for the chosen date & sensor selection.")

st.caption("Tip: Toggle individual swaths in the map’s Layer Control for visualization. "
           "Use the sidebar sensor checkboxes to include/exclude sensors in the coverage stats.")
