import streamlit as st
import pandas as pd
import geopandas as gpd
import pydeck as pdk
import matplotlib.pyplot as plt
import matplotlib as mpl
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
    mn, mx = tmp["WORKDATE"].dt.month.min(), tmp["WORKDATE"].dt.month.max()
    months_sel = st.sidebar.slider("Month range", int(mn), int(mx), (int(mn), int(mx)))
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

# ─── Precompute craft counts ────────────────────────────────────────────────
craft_counts = df.groupby("FAC_ID")["CRAFT"].value_counts().unstack(fill_value=0)

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

# Manual overrides + fuzzy matcher 
manual_map = {
    "COLLEGE OF COMPUTING": "COLL OF COMPUTI",
    "COC":                   "COLL OF COMPUTI",
    "575 14TH STREET":       "575 14TH STREET",  # override invalid fuzzy pick
    # add more overrides here.
}

def find_df_key(name: str):
    """
    Returns (matched_key, method)
    method in {'manual','exact','fuzzy','none'}
    """
    if not isinstance(name, str):
        return None, "none"
    n = name.strip()
    if n in manual_map:
        return manual_map[n], "manual"
    if n in craft_counts.index:
        return n, "exact"
    m = get_close_matches(n, craft_counts.index, n=1, cutoff=0.7)
    return (m[0], "fuzzy") if m else (None, "none")

#  Display manual & fuzzy overrides ─
log = []
for b in sorted([x for x in gdf["FAC_NAME"] if isinstance(x, str)]):
    key, method = find_df_key(b)
    if method in ("manual", "fuzzy"):
        log.append({"Building": b, "Matched FAC_ID": key, "Method": method})
log_df = pd.DataFrame(log)
st.sidebar.markdown("### Overrides review")
st.sidebar.dataframe(log_df, use_container_width=True)

# Pie-chart renderer (cached) 
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

#  Tooltip HTML 
def build_tooltip_html(row):
    bname = row["FAC_NAME"]
    key, method = find_df_key(bname)
    # guard: must exist in craft_counts
    if not key or key not in craft_counts.index:
        return f"<div><strong>{bname}</strong><br>No work-order data</div>"
    cnts = craft_counts.loc[key]
    cnts = cnts[cnts > 0]
    pct  = (cnts / cnts.sum() * 100).round(1)
    uri  = make_pie_datauri(cnts)
    lines = []
    for craft, p in zip(cnts.index, pct):
        col = CRAFT_COLORS.get(craft, "#CCCCCC")
        sw  = f"<span style='display:inline-block;width:12px;height:12px;background:{col};margin-right:4px;'></span>"
        lines.append(f"{sw}{craft}: {p}%")
    legend = "<br>".join(lines)
    return (
        "<div style='text-align:center;'>"
          f"<strong>{bname}</strong><br>"
          f"<img src='{uri}' width='120px'><br>"
          "<div style='column-count:2;column-gap:8px;font-size:0.9em;overscroll-behavior:contain;'>"
            f"{legend}"
          "</div>"
        "</div>"
    )

gdf["tooltip_html"] = gdf.apply(build_tooltip_html, axis=1)

# Render map 
view = pdk.ViewState(latitude=33.7756, longitude=-84.3963, zoom=16, pitch=0)
layer = pdk.Layer(
    "GeoJsonLayer", data=gdf, pickable=True,
    stroked=True, filled=True, extruded=False,
    get_fill_color=[50,100,200,80], get_line_color=[255,255,255,200]
)
deck = pdk.Deck(
    layers=[layer],
    initial_view_state=view,
    tooltip={"html":"{tooltip_html}", "style":{"backgroundColor":"rgba(0,0,0,0.8)","color":"white"}}
)

st.write("Hover over a building to see its pie-chart and legend.")
st.pydeck_chart(deck)