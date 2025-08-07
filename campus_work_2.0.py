import streamlit as st
import pandas as pd
import geopandas as gpd
import pydeck as pdk
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from io import BytesIO
import base64
from difflib import get_close_matches

# Page setup
st.set_page_config(page_title="Campus Building Work Orders", layout="wide")
st.title("Map of Campus and Building Work-Orders")

# Color palettes & thresholds
CRAFT_COLORS = {
    "HVAC": "#1f77b4", "ELECTRIC": "#ff7f0e", "CARPENTRY": "#2ca02c",
    "PLUMBING": "#d62728", "MULTI-CRAFT": "#9467bd", "PAINT": "#8c564b",
    "ADMINISTRATIVE": "#e377c2", "PROJECT MANAGEMENT": "#7f7f7f",
}
SEASON_MONTHS = {
    "Winter": [12, 1, 2],
    "Spring": [3, 4, 5],
    "Summer": [6, 7, 8],
    "Fall":   [9, 10, 11],
}
PCT_THRESHOLD = 5.0

# Sidebar filters
st.sidebar.header("Filters")

@st.cache_data
def get_year_bounds():
    df = pd.read_csv("DF_WO_GaTech.csv", parse_dates=["WORKDATE"], usecols=["WORKDATE"])
    yrs = df["WORKDATE"].dt.year
    return int(yrs.min()), int(yrs.max())

yr_min, yr_max = get_year_bounds()
years_sel = st.sidebar.slider("Year range", yr_min, yr_max, (yr_min, yr_max))

# Use a radio so only one of month-range OR season can be active
mode = st.sidebar.radio("Also filter by…", ["None", "Month-range", "Season"])

if mode == "Month-range":
    tmp = pd.read_csv("DF_WO_GaTech.csv", parse_dates=["WORKDATE"], usecols=["WORKDATE"])
    mo_min, mo_max = int(tmp["WORKDATE"].dt.month.min()), int(tmp["WORKDATE"].dt.month.max())
    months_sel = st.sidebar.slider("Months", mo_min, mo_max, (mo_min, mo_max))
    season_months = None
elif mode == "Season":
    months_sel = (None, None)
    season_sel = st.sidebar.selectbox("Season", list(SEASON_MONTHS))
    season_months = SEASON_MONTHS[season_sel]
else:
    months_sel = (None, None)
    season_months = None

# Load & filter work-orders
@st.cache_data
def load_orders(years, months, season_months):
    df = pd.read_csv("DF_WO_GaTech.csv", parse_dates=["WORKDATE"])
    df["year"]  = df["WORKDATE"].dt.year
    df["month"] = df["WORKDATE"].dt.month
    df["FAC_ID"] = (
        df["FAC_ID"].str.upper()
                  .str.replace(r"\s+", " ", regex=True)
                  .str.strip()
    )
    mask = df["year"].between(*years)
    if months[0] is not None:
        mask &= df["month"].between(*months)
    if season_months:
        mask &= df["month"].isin(season_months)
    return df[mask]

orders = load_orders(years_sel, months_sel, season_months)

# Count crafts per building
craft_counts = orders.groupby("FAC_ID")["CRAFT"].value_counts().unstack(fill_value=0)
order_totals  = craft_counts.sum(axis=1)
max_total     = order_totals.max() or 1

# Load campus footprints
@st.cache_data
def load_buildings():
    g = gpd.read_file("campus_buildings.geojson")
    g["FAC_NAME"] = (
        g["Sheet3__Common_Name"]
         .str.upper()
         .str.replace(r"\s+", " ", regex=True)
         .str.strip()
    )
    return g

gdf = load_buildings()

# Manual name overrides
manual_map = {
    "COLLEGE OF COMPUTING": "COLL OF COMPUTI",
    "COC":                   "COLL OF COMPUTI",
    "575 14TH STREET":       "575 14TH STREET",
    "KLAUS BLDG": "ADV COMP BLDG.",

    # Add more here when confirmed
}

def find_fac_id(name):
    if not isinstance(name, str):
        return None
    nm = name.strip()
    if nm in manual_map:
        return manual_map[nm]
    if nm in craft_counts.index:
        return nm
    matches = get_close_matches(nm, craft_counts.index, n=1, cutoff=0.7)
    return matches[0] if matches else None

# Compute total orders PER feature
def lookup_total(name):
    fid = find_fac_id(name)
    return int(order_totals.get(fid, 0))

gdf["order_sum"] = gdf["FAC_NAME"].apply(lookup_total)

# Heat-map color: gray if zero, else at least 20% into “OrRd”
cmap = cm.get_cmap("OrRd")
def heat_color(n):
    if n <= 0:
        return [200,200,200,80]
    frac  = n / max_total
    level = min(0.2 + 0.8 * frac, 1.0)
    r, g, b, _ = cmap(level)
    return [int(r*255),int(g*255),int(b*255),180]

gdf["fill_color"] = gdf["order_sum"].apply(heat_color)

# Pie chart generator
@st.cache_data
def pie_data_uri(idx, vals):
    s = pd.Series(vals, index=idx)
    fig, ax = plt.subplots(figsize=(2,2))
    cols = [CRAFT_COLORS.get(c, "#CCCCCC") for c in s.index]
    ax.pie(s, autopct=lambda p: f"{p:.0f}%" if p>=PCT_THRESHOLD else "",
           startangle=90, colors=cols, wedgeprops={"edgecolor":"white"})
    ax.axis("equal")
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", transparent=True)
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()

def make_pie(counts):
    key = (tuple(counts.index), tuple(counts.values))
    return pie_data_uri(*key)

# Build per-feature tooltip HTML
def build_tooltip(row):
    nm = find_fac_id(row["FAC_NAME"])
    if not nm or nm not in craft_counts.index:
        return f"<div><strong>{row['FAC_NAME']}</strong><br>No data</div>"
    ct  = craft_counts.loc[nm]
    ct  = ct[ct>0]
    pct = (ct/ct.sum()*100).round(1)
    uri = make_pie(ct)
    lines = []
    for craft, p in zip(ct.index, pct):
        color = CRAFT_COLORS.get(craft, "#CCCCCC")
        sw = f"<span style='display:inline-block;width:12px;height:12px;background:{color};margin-right:4px;'></span>"
        lines.append(f"{sw}{craft}: {p}%")
    legend = "<br>".join(lines)
    return (
        "<div style='text-align:center'>"
          f"<strong>{row['FAC_NAME']}</strong><br>"
          f"<img src='{uri}' width='120px'><br>"
          "<div style='column-count:2; font-size:0.9em;'>"
            f"{legend}"
          "</div>"
        "</div>"
    )

gdf["tooltip_html"] = gdf.apply(build_tooltip, axis=1)

# Render PyDeck map
view = pdk.ViewState(latitude=33.7756, longitude=-84.3963, zoom=16)
layer = pdk.Layer(
    "GeoJsonLayer",
    data=gdf,
    pickable=True,
    stroked=True,
    filled=True,
    extruded=False,
    get_fill_color="fill_color",
    get_line_color=[255,255,255,200]
)
deck = pdk.Deck(
    layers=[layer],
    initial_view_state=view,
    tooltip={"html":"{tooltip_html}",
             "style":{"backgroundColor":"rgba(0,0,0,0.8)","color":"white"}}
)

st.write("Hover a building for its work-order pie chart and heat color.")
st.pydeck_chart(deck, use_container_width=True)




