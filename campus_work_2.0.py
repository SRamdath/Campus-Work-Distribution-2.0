import streamlit as st
import pandas as pd
import geopandas as gpd
import pydeck as pdk
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from io import BytesIO
import base64
from difflib import get_close_matches

# ─── Page setup ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Map of Campus and building work-orders", layout="wide")
st.title("Map of Campus and building work-orders")

# ─── Constants ──────────────────────────────────────────────────────────────
CRAFT_COLORS = {
    "HVAC":               "#1f77b4",
    "ELECTRIC":           "#ff7f0e",
    "CARPENTRY":          "#2ca02c",
    "PLUMBING":           "#d62728",
    "MULTI-CRAFT":        "#9467bd",
    "PAINT":              "#8c564b",
    "ADMINISTRATIVE":     "#e377c2",
    "PROJECT MANAGEMENT": "#7f7f7f",
}
SEASON_MONTHS = {
    "Winter": [12, 1, 2],
    "Spring": [3, 4, 5],
    "Summer": [6, 7, 8],
    "Fall":   [9, 10, 11],
}
PCT_THRESHOLD = 5.0

# ─── Sidebar filters ────────────────────────────────────────────────────────
st.sidebar.header("🔍 Filters")

@st.cache_data
def load_year_bounds():
    df = pd.read_csv("DF_WO_GaTech.csv", parse_dates=["WORKDATE"], usecols=["WORKDATE"])
    yrs = df["WORKDATE"].dt.year
    return int(yrs.min()), int(yrs.max())

min_y, max_y = load_year_bounds()
years_sel = st.sidebar.slider("Year range", min_y, max_y, (min_y, max_y))

filter_months = st.sidebar.checkbox("Month-range filter", False)
if filter_months:
    tmp = pd.read_csv("DF_WO_GaTech.csv", parse_dates=["WORKDATE"], usecols=["WORKDATE"])
    mn, mx = int(tmp["WORKDATE"].dt.month.min()), int(tmp["WORKDATE"].dt.month.max())
    months_sel = st.sidebar.slider("Month range", mn, mx, (mn, mx))
else:
    months_sel = (None, None)

filter_season = st.sidebar.checkbox("Season filter", False)
if filter_season:
    season_sel = st.sidebar.selectbox("Season", list(SEASON_MONTHS))
    season_months = SEASON_MONTHS[season_sel]
else:
    season_months = None

# ─── Load & filter work-orders ─────────────────────────────────────────────
@st.cache_data
def load_and_filter_orders(years, months, season):
    df = pd.read_csv("DF_WO_GaTech.csv", parse_dates=["WORKDATE"])
    df["year"]  = df["WORKDATE"].dt.year
    df["month"] = df["WORKDATE"].dt.month
    df["FAC_ID"] = (
        df["FAC_ID"]
          .str.upper()
          .str.replace(r"\s+", " ", regex=True)
          .str.strip()
    )
    mask = df["year"].between(*years)
    if months[0] is not None:
        mask &= df["month"].between(*months)
    if season is not None:
        mask &= df["month"].isin(season)
    return df.loc[mask].copy()

df = load_and_filter_orders(years_sel, months_sel, season_months)

# ─── Compute craft counts & totals ──────────────────────────────────────────
craft_counts = df.groupby("FAC_ID")["CRAFT"].value_counts().unstack(fill_value=0)
order_sums   = craft_counts.sum(axis=1)
max_orders   = order_sums.max() or 1

# ─── Load building footprints ───────────────────────────────────────────────
@st.cache_data
def load_buildings():
    gdf = gpd.read_file("campus_buildings.geojson")
    gdf["FAC_NAME"] = (
        gdf["Sheet3__Common_Name"]
           .str.upper()
           .str.replace(r"\s+", " ", regex=True)
           .str.strip()
    )
    return gdf

gdf = load_buildings()

# ─── Name→FAC_ID matching (manual + fuzzy) ─────────────────────────────────
manual_map = {
    "COLLEGE OF COMPUTING": "COLL OF COMPUTI",
    "COC":                   "COLL OF COMPUTI",
    "575 14TH STREET":       "575 14TH STREET",
    # add more overrides here...
}

def find_df_key(name: str):
    if not isinstance(name, str):
        return None, "none"
    n = name.strip()
    if n in manual_map:
        return manual_map[n], "manual"
    if n in craft_counts.index:
        return n, "exact"
    m = get_close_matches(n, craft_counts.index, n=1, cutoff=0.7)
    return (m[0], "fuzzy") if m else (None, "none")

# ─── Compute per‐feature order_sum via matcher ─────────────────────────────
def lookup_total(name: str) -> int:
    key, _ = find_df_key(name)
    return int(order_sums.get(key, 0))

gdf["order_sum"] = gdf["FAC_NAME"].apply(lookup_total)

# ─── Heat‐map coloring ───────────────────────────────────────────────────────
cmap = cm.get_cmap("OrRd")

def compute_color(total):
    if total <= 0:
        return [200, 200, 200, 80]  # light gray
    raw   = float(total) / max_orders
    ratio = min(0.2 + 0.8 * raw, 1.0)
    r, g, b, _ = cmap(ratio)
    return [int(r*255), int(g*255), int(b*255), 180]

gdf["fill_color"] = gdf["order_sum"].apply(compute_color)

# ─── Pie‐chart rendering ────────────────────────────────────────────────────
@st.cache_data
def make_pie_datauri_cached(idx, vals):
    s = pd.Series(vals, index=idx)
    fig, ax = plt.subplots(figsize=(2,2))
    cols = [CRAFT_COLORS.get(c, "#CCCCCC") for c in s.index]
    auto = lambda p: f"{p:.0f}%" if p >= PCT_THRESHOLD else ""
    ax.pie(s, autopct=auto, startangle=90, colors=cols, wedgeprops={"edgecolor":"white"})
    ax.axis("equal")
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", transparent=True)
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()

def make_pie_datauri(counts):
    key = (tuple(counts.index), tuple(counts.values))
    return make_pie_datauri_cached(*key)

# ─── Tooltip HTML ───────────────────────────────────────────────────────────
def build_tooltip_html(row):
    nm, _ = find_df_key(row["FAC_NAME"])
    if not nm or nm not in craft_counts.index:
        return f"<div><strong>{row['FAC_NAME']}</strong><br>No work-order data</div>"
    ct  = craft_counts.loc[nm]
    ct  = ct[ct > 0]
    pct = (ct / ct.sum() * 100).round(1)
    uri = make_pie_datauri(ct)
    lines = []
    for craft, p in zip(ct.index, pct):
        col = CRAFT_COLORS.get(craft, "#CCCCCC")
        sw  = f"<span style='display:inline-block;width:12px;height:12px;background:{col};margin-right:4px;'></span>"
        lines.append(f"{sw}{craft}: {p}%")
    legend = "<br>".join(lines)
    return (
        "<div style='text-align:center;'>"
          f"<strong>{row['FAC_NAME']}</strong><br>"
          f"<img src='{uri}' width='120px'><br>"
          "<div style='column-count:2;column-gap:8px;font-size:0.9em;overscroll-behavior:contain;'>"
            f"{legend}"
          "</div>"
        "</div>"
    )

gdf["tooltip_html"] = gdf.apply(build_tooltip_html, axis=1)

# ─── Render Pydeck map ──────────────────────────────────────────────────────
view = pdk.ViewState(latitude=33.7756, longitude=-84.3963, zoom=16, pitch=0)
layer = pdk.Layer(
    "GeoJsonLayer",
    data=gdf,
    pickable=True,
    stroked=True,
    filled=True,
    extruded=False,
    get_fill_color="fill_color",
    get_line_color=[255,255,255,200],
)
deck = pdk.Deck(
    layers=[layer],
    initial_view_state=view,
    tooltip={"html":"{tooltip_html}", "style":{"backgroundColor":"rgba(0,0,0,0.8)","color":"white"}}
)

st.write("Hover over a building to see its pie-chart and color‐coded fill by total orders.")
st.pydeck_chart(deck, use_container_width=True)

