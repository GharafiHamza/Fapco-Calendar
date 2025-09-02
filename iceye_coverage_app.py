
from pathlib import Path
import json
import re
from datetime import datetime, timedelta, timezone

import streamlit as st
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union
from shapely.geometry import shape, mapping
from shapely.geometry.base import BaseGeometry

try:
    from shapely.make_valid import make_valid  # Shapely ≥2
except Exception:
    make_valid = None

try:
    from pyproj import Geod
    GEOD = Geod(ellps="WGS84")
except Exception:
    GEOD = None  # we'll approximate if pyproj isn't available

st.set_page_config(page_title="ICEYE Coverage Planner", layout="wide")

# --------------------------- Helpers ---------------------------

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

def _geodesic_area_m2(geom: BaseGeometry) -> float:
    """Geodesic area on WGS84 in m^2; falls back to crude approx if pyproj missing."""
    if geom is None:
        return 0.0
    try:
        if GEOD is not None:
            # Handle multipolygons by iterating
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
    # Fallback: approximate degrees→km (rough; acceptable as last resort)
    approx_km2 = geom.area * (111.32**2)
    return approx_km2 * 1_000_000.0 / 1_000_000.0  # keep units explicit

def _load_gdf_from_path_or_upload(label: str, default_path: Path, file_types=("geojson",)):
    """Load a GeoJSON as GeoDataFrame from default path or via uploader."""
    col1, col2 = st.columns([3, 2])
    with col1:
        st.caption(f"**{label}**")
        st.code(str(default_path))
    with col2:
        up = st.file_uploader(f"Upload {label}", type=list(file_types), key=f"up_{label}")
    raw = None
    if default_path.exists():
        raw = default_path.read_bytes()
    elif up is not None:
        raw = up.read()
    if raw is None:
        st.warning(f"Missing: {label}")
        return gpd.GeoDataFrame([], columns=["geometry"], geometry="geometry", crs="EPSG:4326")
    try:
        gj = json.loads(raw.decode("utf-8"))
        feats = gj.get("features", [])
        gdf = gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326")
        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True)
        else:
            gdf = gdf.to_crs(epsg=4326)
        gdf["geometry"] = gdf["geometry"].apply(_mk_valid)
        gdf = gdf[~gdf["geometry"].is_empty & gdf["geometry"].notna()]
        gdf.reset_index(drop=True, inplace=True)
        return gdf
    except Exception as e:
        st.error(f"Failed to parse {label}: {e}")
        return gpd.GeoDataFrame([], columns=["geometry"], geometry="geometry", crs="EPSG:4326")

def _extract_dates_from_props(props: dict):
    dates = set()
    if not isinstance(props, dict):
        return dates
    for v in props.values():
        if v is None:
            continue
        if not isinstance(v, str):
            v = str(v)
        for hit in DATE_RE.findall(v):
            dates.add(hit)
    return sorted(dates)

def _classify_sensor(props: dict):
    """Return a tuple (mode, look) among SENSOR_KEYS or (None, None)."""
    blob = " ".join([f"{k}={v}" for k, v in (props or {}).items() if v is not None]).lower()
    mode = None
    if "topsar wide" in blob or "topsar_wide" in blob or "topsar-wide" in blob or "topsarw" in blob:
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
    if (mode, look) in SENSOR_LABELS:
        return mode, look
    return None, None

def _get_ref_date(chosen_dt: datetime, reference_dates, period_days: int):
    """Pick the phase-aligned reference date <= chosen date wherever possible."""
    if not reference_dates:
        return None
    # Choose the latest reference date <= chosen date with same phase
    best = None
    for r in reference_dates:
        if ((chosen_dt.date() - r.date()).days % period_days) == 0:
            if r <= chosen_dt and (best is None or r > best):
                best = r
    if best is not None:
        return best
    # fallback to anchor
    anchor = max([r for r in reference_dates if r <= chosen_dt], default=min(reference_dates))
    shift = (chosen_dt.date() - anchor.date()).days % period_days
    return (chosen_dt - timedelta(days=shift)).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)

@st.cache_data(show_spinner=False)
def _load_reference(gj_path: Path):
    """Return (gdf, sorted_unique_datesUTC) for S1 reference plan."""
    if not gj_path.exists():
        return gpd.GeoDataFrame([], geometry="geometry", crs="EPSG:4326"), []
    gj = json.loads(gj_path.read_text(encoding="utf-8"))
    feats = gj.get("features", [])
    gdf = gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326")
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)
    else:
        gdf = gdf.to_crs(epsg=4326)
    gdf["geometry"] = gdf["geometry"].apply(_mk_valid)
    seen = set()
    for _, row in gdf.iterrows():
        props = row.get("properties", {})
        for iso in _extract_dates_from_props(props):
            try:
                seen.add(datetime.strptime(iso, "%Y-%m-%d").replace(tzinfo=timezone.utc))
            except Exception:
                pass
    return gdf, sorted(seen)

# --------------------------- Header (two logos) ---------------------------
def header_two_logos():
    left, mid, right = st.columns([1.2, 5, 1.2])
    with left:
        file_left = st.file_uploader("Upload left logo", type=["png", "jpg", "jpeg"], key="logo_left")
        p_left = Path("logo_left.png")
        b64 = None
        if file_left is not None:
            import base64
            b64 = base64.b64encode(file_left.read()).decode("utf-8")
        elif p_left.exists():
            b64 = _base64_of(p_left)
        if b64:
            st.markdown(f"<img src='data:image/png;base64,{b64}' style='height:58px;'>", unsafe_allow_html=True)
    with mid:
        st.markdown("<h2 style='text-align:center;margin:6px 0;'>ICEYE Daily Swath Planner & S1 Reference Coverage</h2>", unsafe_allow_html=True)
    with right:
        file_right = st.file_uploader("Upload right logo", type=["png", "jpg", "jpeg"], key="logo_right")
        p_right = Path("logo_right.png")
        b64r = None
        if file_right is not None:
            import base64
            b64r = base64.b64encode(file_right.read()).decode("utf-8")
        elif p_right.exists():
            b64r = _base64_of(p_right)
        if b64r:
            st.markdown(f"<img src='data:image/png;base64,{b64r}' style='height:58px;float:right;'>", unsafe_allow_html=True)

header_two_logos()

# --------------------------- Inputs ---------------------------
st.sidebar.header("Inputs")

# Date picker
from datetime import date
chosen_date = st.sidebar.date_input('Pick a date', value=date(2025, 9, 1), key='single_date')
if isinstance(chosen_date, (list, tuple)):
    chosen_date = chosen_date[0].strftime("%Y-%m-%d")))
chosen_ymd = chosen_date.strftime("%Y-%m-%d")
chosen_dt = datetime.strptime(chosen_ymd, "%Y-%m-%d").replace(tzinfo=timezone.utc)

# AOI
AOI_PATH = Path("AOI.geojson")
def _load_gdf_from_path(label, path):
    if not path.exists():
        st.warning(f"Missing: {label} at {path}")
        return gpd.GeoDataFrame([], geometry="geometry", crs="EPSG:4326")
    gj = json.loads(path.read_text(encoding="utf-8"))
    gdf = gpd.GeoDataFrame.from_features(gj.get("features", []), crs="EPSG:4326")
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)
    else:
        gdf = gdf.to_crs(epsg=4326)
    gdf["geometry"] = gdf["geometry"].apply(_mk_valid)
    gdf = gdf[~gdf["geometry"].is_empty & gdf["geometry"].notna()].reset_index(drop=True)
    return gdf

aoi_gdf = _load_gdf_from_path("AOI", AOI_PATH)

# ICEYE swaths
SWATHS_PATH = Path("fapco_3months.geojson")
swaths_gdf = _load_gdf_from_path("ICEYE Swaths", SWATHS_PATH)

# Sentinel-1 reference plans
S1A_PATH = Path("S1A_12day_reference_coverage_plan.geojson")
S1C_PATH = Path("S1C_12day_reference_coverage_plan.geojson")

s1a_gdf, s1a_dates = _load_reference(S1A_PATH)
s1c_gdf, s1c_dates = _load_reference(S1C_PATH)

# Sensor toggles
st.sidebar.subheader("Sensors to display (ICEYE)")
SENSOR_LABELS = {
    ("TOPSAR", "LEFT"): "TopSAR (Left)",
    ("TOPSAR", "RIGHT"): "TopSAR (Right)",
    ("TOPSAR WIDE", "LEFT"): "TopSAR Wide (Left)",
    ("TOPSAR WIDE", "RIGHT"): "TopSAR Wide (Right)",
}
sensor_flags = {}
for key in [("TOPSAR", "LEFT"), ("TOPSAR", "RIGHT"), ("TOPSAR WIDE", "LEFT"), ("TOPSAR WIDE", "RIGHT")]:
    sensor_flags[key] = st.sidebar.checkbox(SENSOR_LABELS[key], value=True, key=f"chk_{key}")

st.sidebar.subheader("Sentinel-1 reference")
show_s1a = st.sidebar.checkbox("Show S1A coverage (12-day phase)", value=True)
show_s1c = st.sidebar.checkbox("Show S1C coverage (12-day phase)", value=True)

st.sidebar.divider()
st.sidebar.info("Layer control is available on the map. Coverage stats update with the checkboxes above.")

# --------------------------- Filter ICEYE swaths by date & sensor ---------------------------
def _filter_swaths_by_date_and_sensor(gdf: gpd.GeoDataFrame, ymd: str, aoi_union: BaseGeometry):
    if gdf.empty:
        return {k: gpd.GeoDataFrame([], geometry="geometry", crs="EPSG:4326") for k in SENSOR_KEYS}

    # Ensure a 'properties' column exists
    if "properties" not in gdf.columns:
        gdf = gdf.copy()
        gdf["properties"] = [{} for _ in range(len(gdf))]

    # Build a mask for date
    mask = []
    for _, row in gdf.iterrows():
        props = row.get("properties", {})
        blob = " ".join([str(v) for v in props.values() if v is not None])
        for col in gdf.columns:
            if col in ("geometry", "properties"):
                continue
            val = row.get(col)
            if val is not None:
                blob += " " + str(val)
        found = False
        for hit in DATE_RE.findall(blob):
            if hit == ymd:
                found = True
                break
        if not found:
            s = row.get("start") or row.get("Start") or row.get("acq_date")
            if s and str(s)[:10] == ymd:
                found = True
        mask.append(found)

    gdf_d = gdf[pd.Series(mask, index=gdf.index)]
    if gdf_d.empty:
        return {k: gpd.GeoDataFrame([], geometry="geometry", crs="EPSG:4326") for k in SENSOR_KEYS}

    # Intersect with AOI
    if aoi_union and not aoi_union.is_empty:
        gdf_d = gdf_d[gdf_d.intersects(aoi_union)]
        if gdf_d.empty:
            return {k: gpd.GeoDataFrame([], geometry="geometry", crs="EPSG:4326") for k in SENSOR_KEYS}

    # Classify sensors
    def _classify(props: dict):
        blob = " ".join([f"{k}={v}" for k, v in (props or {}).items() if v is not None]).lower()
        mode = None
        if "topsar wide" in blob or "topsar_wide" in blob or "topsar-wide" in blob or "topsarw" in blob:
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
        if (mode, look) in SENSOR_LABELS:
            return mode, look
        return None, None

    buckets = {k: [] for k in SENSOR_KEYS}
    for _, row in gdf_d.iterrows():
        props = row.get("properties", {})
        mode, look = _classify(props)
        if (mode, look) not in buckets:
            props2 = props.copy() if isinstance(props, dict) else {}
            for c in gdf.columns:
                if c not in ("geometry", "properties"):
                    props2[c] = row.get(c)
            mode, look = _classify(props2)
        if (mode, look) in buckets:
            buckets[(mode, look)].append(row)

    out = {}
    for key in SENSOR_KEYS:
        rows = buckets.get(key, [])
        if not rows:
            out[key] = gpd.GeoDataFrame([], geometry="geometry", crs="EPSG:4326")
        else:
            out[key] = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326").reset_index(drop=True)
    return out

aoi_union = aoi_gdf.unary_union if not aoi_gdf.empty else None
sensor_gdfs = _filter_swaths_by_date_and_sensor(swaths_gdf, chosen_ymd, aoi_union)
no_swaths_today = all(gdf.empty for gdf in sensor_gdfs.values())

# --------------------------- Sentinel-1 phase selection ---------------------------
def _s1_phase_subset(gdf: gpd.GeoDataFrame, ref_dates, period_days: int, chosen: datetime):
    if gdf.empty or not ref_dates:
        return gpd.GeoDataFrame([], geometry="geometry", crs="EPSG:4326"), None
    ref_dt = _get_ref_date(chosen, ref_dates, period_days)
    if ref_dt is None:
        return gpd.GeoDataFrame([], geometry="geometry", crs="EPSG:4326"), None
    ref_ymd = ref_dt.strftime("%Y-%m-%d")

    mask = []
    for _, row in gdf.iterrows():
        props = row.get("properties", {})
        has = False
        for v in props.values():
            if isinstance(v, str):
                for hit in DATE_RE.findall(v):
                    if hit == ref_ymd:
                        has = True
                        break
            if has:
                break
        mask.append(has)
    sub = gdf[pd.Series(mask, index=gdf.index)]
    if not sub.empty and aoi_union and not aoi_union.is_empty:
        sub = sub[sub.intersects(aoi_union)].reset_index(drop=True)
    return sub, ref_ymd

s1a_sub, s1a_ref = _s1_phase_subset(s1a_gdf, s1a_dates, 12, chosen_dt)
s1c_sub, s1c_ref = _s1_phase_subset(s1c_gdf, s1c_dates, 12, chosen_dt)

# --------------------------- Map ---------------------------
m = folium.Map(location=[25, 45], zoom_start=4, tiles=None, control_scale=True)
folium.TileLayer("OpenStreetMap", name="OpenStreetMap", show=True).add_to(m)

# AOI Layer
if not aoi_gdf.empty:
    folium.GeoJson(
        aoi_gdf.__geo_interface__,
        name='AOI',
        name="AOI",
        style_function=lambda x: {"color": "#f1c40f", "weight": 2, "fill": True, "fillOpacity": 0.08},
        highlight_function=lambda x: {"weight": 3, "color": "#f39c12"},
        tooltip=folium.GeoJsonTooltip(fields=[], aliases=[], labels=False)
    ).add_to(m)

# ICEYE Sensor layers + union for coverage
coverage_union = None
detail_rows = []  # for per-frame table

SENSOR_COLORS = {
    ("TOPSAR", "LEFT"): "#1f77b4",
    ("TOPSAR", "RIGHT"): "#ff7f0e",
    ("TOPSAR WIDE", "LEFT"): "#2ca02c",
    ("TOPSAR WIDE", "RIGHT"): "#d62728",
}

for key in SENSOR_KEYS:
    label = SENSOR_LABELS[key]
    color = SENSOR_COLORS[key]
    sub = sensor_gdfs.get(key, gpd.GeoDataFrame([], geometry="geometry", crs="EPSG:4326"))
    if sensor_flags.get(key, False) and not sub.empty:
        if aoi_union and not aoi_union.is_empty:
            inter_geoms = []
            for i, row in sub.iterrows():
                geom = row.geometry
                try:
                    inter = geom.intersection(aoi_union)
                except Exception:
                    inter = None
                if inter and not inter.is_empty:
                    inter_geoms.append(inter)
                    props = row.get("properties", {})
                    sat = props.get("satellite") or props.get("sat") or props.get("platform") or "ICEYE"
                    fid = props.get("frame_id") or props.get("id") or f"frame_{i}"
                    dates = [d for d in re.findall(r"\\b(20\\d{2}-\\d{2}-\\d{2})\\b", " ".join([str(v) for v in props.values() if v is not None]))]
                    date_str = dates[0] if dates else ""
                    area_km2 = _geodesic_area_m2(inter) / 1_000_000.0
                    detail_rows.append({
                        "Satellite": sat,
                        "Sensor": label,
                        "Frame_ID": fid,
                        "Date": date_str or "{ymd}".format(ymd="{0}".format("{chosen}".format(chosen="{0}".format("")))),
                        "Area_km²_in_AOI": round(area_km2, 3)
                    })
            if inter_geoms:
                layer_geojson = gpd.GeoSeries(inter_geoms, crs="EPSG:4326").__geo_interface__
                folium.GeoJson(
                    layer_geojson,
                    name=f"ICEYE • {label}",
                    style_function=lambda x, color=color: {"color": color, "fillColor": color, "weight": 1, "fillOpacity": 0.25},
                ).add_to(m)
                union_piece = unary_union(inter_geoms)
                coverage_union = union_piece if coverage_union is None else coverage_union.union(union_piece)
        else:
            folium.GeoJson(
                sub.__geo_interface__,
                name=f"ICEYE • {label}",
                style_function=lambda x, color=color: {"color": color, "fillColor": color, "weight": 1, "fillOpacity": 0.25},
            ).add_to(m)

# Sentinel-1 layers
if show_s1a and not s1a_sub.empty:
    folium.GeoJson(
        s1a_sub.__geo_interface__,
        name=f"S1A (ref {s1a_ref})",
        style_function=lambda x: {"color": S1_COLORS["S1A"], "fillColor": S1_COLORS["S1A"], "weight": 1, "fillOpacity": 0.15}
    ).add_to(m)
if show_s1c and not s1c_sub.empty:
    folium.GeoJson(
        s1c_sub.__geo_interface__,
        name=f"S1C (ref {s1c_ref})",
        style_function=lambda x: {"color": S1_COLORS["S1C"], "fillColor": S1_COLORS["S1C"], "weight": 1, "fillOpacity": 0.15}
    ).add_to(m)

# Fit map
if not aoi_gdf.empty:
    minx, miny, maxx, maxy = aoi_gdf.total_bounds
    m.fit_bounds([[miny, minx], [maxy, maxx]])

folium.LayerControl(collapsed=False).add_to(m)
if no_swaths_today:
    st.info('No ICEYE swaths found for this exact date in the input file. The map still loads with AOI and any available layers; pick another date or adjust inputs.')
map_state = st_folium(m, height=740, use_container_width=True)

# --------------------------- Coverage stats + tables ---------------------------
st.markdown("### Coverage & Tables")

if aoi_union and coverage_union and (not coverage_union.is_empty):
    aoi_area_km2 = _geodesic_area_m2(aoi_union) / 1_000_000.0
    covered_area_km2 = _geodesic_area_m2(aoi_union.intersection(coverage_union)) / 1_000_000.0
    pct = (covered_area_km2 / aoi_area_km2) * 100 if aoi_area_km2 > 0 else 0.0
    st.success(f"**Covered area:** {covered_area_km2:,.2f} km² of {aoi_area_km2:,.2f} km² (**{pct:.2f}%**) — updated live from the sensor toggles above.")
else:
    st.info("No ICEYE sensor layers enabled or no intersections with AOI on this date.")

# Detail table (per frame)
if detail_rows:
    df_detail = pd.DataFrame(detail_rows)
    # --- Normalize column names & dtypes for robust ops ---
    display_cols = {}
    if "Area_km²_in_AOI" in df_detail.columns:
        df_detail = df_detail.rename(columns={"Area_km²_in_AOI": "Area_km2_in_AOI"})
        display_cols["Area_km2_in_AOI"] = "Area_km²_in_AOI"
    # Coerce numeric
    if "Area_km2_in_AOI" in df_detail.columns:
        df_detail["Area_km2_in_AOI"] = pd.to_numeric(df_detail["Area_km2_in_AOI"], errors="coerce").fillna(0.0)

    # --- Summary by satellite & sensor ---
    try:
        df_summary = (df_detail
                      .groupby(["Satellite", "Sensor"], dropna=False, as_index=False)
                      .agg(Frames=("Frame_ID", "count"),
                           Area_km2=("Area_km2_in_AOI", "sum"))
                      .sort_values(["Satellite", "Sensor"]))
    except Exception:
        # Fallback compatible with older pandas
        gb = df_detail.groupby(["Satellite", "Sensor"], dropna=False)
        df_summary = gb["Frame_ID"].count().to_frame("Frames").join(
            gb["Area_km2_in_AOI"].sum().to_frame("Area_km2")
        ).reset_index().sort_values(["Satellite", "Sensor"])

    with st.expander("Per-frame details"):
        # Swap back to km² in UI if we normalized the column
        df_show = df_detail.copy()
        if "Area_km2_in_AOI" in df_show.columns:
            df_show = df_show.rename(columns={"Area_km2_in_AOI": "Area_km²_in_AOI"})
        st.dataframe(df_show, use_container_width=True)

    with st.expander("Per-satellite & sensor summary"):
        df_sum_show = df_summary.copy()
        if "Area_km2" in df_sum_show.columns:
            df_sum_show = df_sum_show.rename(columns={"Area_km2": "Area_km²"})
        st.dataframe(df_sum_show, use_container_width=True)
else:
    st.info("No per-frame intersections to tabulate for the chosen date & sensor selection.")

st.caption("Tip: Use the sidebar checkboxes to include/exclude ICEYE sensors; the coverage metric updates accordingly. Map layer control is for visual inspection.")
