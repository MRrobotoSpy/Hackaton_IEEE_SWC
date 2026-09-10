import math
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "datasets"
    / "mobeenfatimah"
    / "cityflow-smart-urban-mobility-and-traffic-iot"
    / "versions"
    / "1"
    / "smart_city_traffic_mobility.csv"
)

st.set_page_config(page_title="CityFlow 3D", layout="wide")


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    return df


df = load_data()

st.title("CityFlow — 3D Traffic Explorer")
st.caption("Trascina col tasto destro per inclinare/ruotare la mappa (stile Google Maps 3D).")

with st.sidebar:
    st.header("Filtri")

    zones = sorted(df["city_zone"].unique())
    selected_zones = st.multiselect("City zone", zones, default=zones)

    hour_range = st.slider("Fascia oraria", 0, 23, (0, 23))

    days_map = {0: "Lun", 1: "Mar", 2: "Mer", 3: "Gio", 4: "Ven", 5: "Sab", 6: "Dom"}
    selected_days = st.multiselect(
        "Giorno della settimana",
        options=list(days_map.keys()),
        default=list(days_map.keys()),
        format_func=lambda d: days_map[d],
    )

    metric = st.selectbox(
        "Metrica (altezza colonne)",
        [
            "vehicle_count",
            "congestion_score",
            "average_wait_time",
            "queue_length",
            "emission_estimate",
            "air_quality_index",
        ],
    )

    layer_type = st.radio(
        "Tipo di layer 3D",
        [
            "Colonne per intersezione",
            "Scatter preciso (punti individuali)",
            "Hexagon aggregato",
        ],
        index=0,
    )

    agg_fn_label = st.selectbox(
        "Aggregazione temporale per intersezione",
        ["Media", "Massimo", "Somma", "Ultimo valore"],
        index=0,
    )

    basemap_options = {
        "Road (Carto)": "road",
        "OpenStreetMap": "osm",
        "Satellite": "satellite",
        "Light": "light",
        "Dark": "dark",
    }
    basemap_label = st.selectbox("Basemap", list(basemap_options.keys()), index=0)
    basemap = basemap_options[basemap_label]

    st.divider()
    st.subheader("Ranking intersezioni")
    show_ranking = st.checkbox("Evidenzia peggiori/migliori sulla mappa", value=True)
    top_n = st.slider("Quante evidenziare (per lato)", 3, 20, 10)

mask = (
    df["city_zone"].isin(selected_zones)
    & df["hour"].between(hour_range[0], hour_range[1])
    & df["day_of_week"].isin(selected_days)
)
filtered = df.loc[mask]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Record", f"{len(filtered):,}")
col2.metric("Intersezioni", filtered["intersection_id"].nunique())
col3.metric("Congestione media", f"{filtered['congestion_score'].mean():.1f}")
col4.metric("Velocità media", f"{filtered['average_speed'].mean():.1f} km/h")

if filtered.empty:
    st.warning("Nessun dato con i filtri correnti.")
    st.stop()

lat_center = float(filtered["latitude"].mean())
lng_center = float(filtered["longitude"].mean())
lat_span = float(filtered["latitude"].max() - filtered["latitude"].min())
lng_span = float(filtered["longitude"].max() - filtered["longitude"].min())
span = max(lat_span, lng_span, 0.01)
fit_zoom = max(10.0, min(15.0, math.log2(360.0 / span) - 1))

view_state = pdk.ViewState(
    latitude=lat_center,
    longitude=lng_center,
    zoom=fit_zoom,
    pitch=55,
    bearing=20,
)

agg_fn = {"Media": "mean", "Massimo": "max", "Somma": "sum", "Ultimo valore": "last"}[agg_fn_label]

if layer_type == "Hexagon aggregato":
    layer = pdk.Layer(
        "HexagonLayer",
        data=filtered[["latitude", "longitude", metric]],
        get_position=["longitude", "latitude"],
        get_elevation_weight=metric,
        elevation_scale=8,
        radius=120,
        extruded=True,
        coverage=0.9,
        pickable=True,
        auto_highlight=True,
        color_range=[
            [65, 182, 196],
            [127, 205, 187],
            [199, 233, 180],
            [237, 177, 32],
            [253, 141, 60],
            [227, 26, 28],
        ],
    )
    tooltip = {"text": "Elevazione ≈ somma {}".format(metric)}
else:
    agg = (
        filtered.sort_values("timestamp")
        .groupby(["intersection_id", "latitude", "longitude"], as_index=False)[metric]
        .agg(agg_fn)
    )
    max_val = agg[metric].max() or 1
    agg["norm"] = agg[metric] / max_val
    agg["r"] = (agg["norm"] * 255).astype(int)
    agg["g"] = ((1 - agg["norm"]) * 200).astype(int)
    agg["b"] = 80
    agg["value"] = agg[metric].round(2)

    if layer_type == "Colonne per intersezione":
        elev_max = float(agg[metric].max()) or 1.0
        elev_scale = 800.0 / elev_max
        layer = pdk.Layer(
            "ColumnLayer",
            data=agg,
            get_position=["longitude", "latitude"],
            get_elevation=metric,
            elevation_scale=elev_scale,
            radius=25,
            get_fill_color=["r", "g", "b", 220],
            pickable=True,
            auto_highlight=True,
        )
    else:
        agg["radius_m"] = (40 + agg["norm"] * 80).astype(int)
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=agg,
            get_position=["longitude", "latitude"],
            get_fill_color=["r", "g", "b", 220],
            get_radius="radius_m",
            radius_min_pixels=4,
            radius_max_pixels=20,
            stroked=True,
            get_line_color=[255, 255, 255, 200],
            line_width_min_pixels=1,
            pickable=True,
            auto_highlight=True,
        )
    tooltip = {"text": "{intersection_id}\n" + metric + ": {value}"}

BAD_METRICS = [
    "congestion_score",
    "emission_estimate",
    "fuel_waste_estimate",
    "air_quality_index",
    "average_wait_time",
]
per_intersection = (
    filtered.groupby(["intersection_id", "city_zone", "latitude", "longitude"], as_index=False)[
        BAD_METRICS + ["average_speed"]
    ].mean()
)
for m in BAD_METRICS:
    lo, hi = per_intersection[m].min(), per_intersection[m].max()
    per_intersection[f"{m}_n"] = (
        (per_intersection[m] - lo) / (hi - lo) if hi > lo else 0.0
    )
per_intersection["badness"] = per_intersection[[f"{m}_n" for m in BAD_METRICS]].mean(axis=1)
ranked = per_intersection.sort_values("badness", ascending=False).reset_index(drop=True)

worst = ranked.head(top_n)
best = ranked.tail(top_n).sort_values("badness")

highlight_layers = []
if show_ranking:
    highlight_layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=worst,
            get_position=["longitude", "latitude"],
            get_fill_color=[227, 26, 28, 90],
            get_radius=180,
            radius_min_pixels=12,
            radius_max_pixels=40,
            stroked=True,
            get_line_color=[227, 26, 28, 255],
            line_width_min_pixels=2,
            pickable=True,
        )
    )
    highlight_layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=best,
            get_position=["longitude", "latitude"],
            get_fill_color=[46, 160, 67, 90],
            get_radius=180,
            radius_min_pixels=12,
            radius_max_pixels=40,
            stroked=True,
            get_line_color=[46, 160, 67, 255],
            line_width_min_pixels=2,
            pickable=True,
        )
    )

layers = []
if basemap == "osm":
    layers.append(
        pdk.Layer(
            "TileLayer",
            data="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
            min_zoom=0,
            max_zoom=19,
            tile_size=256,
        )
    )
layers.append(layer)
layers.extend(highlight_layers)

deck = pdk.Deck(
    layers=layers,
    initial_view_state=view_state,
    map_style=None if basemap == "osm" else basemap,
    tooltip=tooltip,
)

st.pydeck_chart(deck, use_container_width=True, height=650)

st.subheader("Confronto intersezioni")
st.caption(
    "Score di 'badness' = media normalizzata (0–1) di congestione, emissioni, spreco carburante, AQI, tempo d'attesa. Più alto = peggio."
)
rank_cols = [
    "intersection_id",
    "city_zone",
    "badness",
    "congestion_score",
    "emission_estimate",
    "fuel_waste_estimate",
    "air_quality_index",
    "average_wait_time",
    "average_speed",
]
c_worst, c_best = st.columns(2)
with c_worst:
    st.markdown(f"### 🔴 Peggiori {top_n}")
    st.dataframe(
        worst[rank_cols].style.format({"badness": "{:.2f}"}).background_gradient(
            subset=["badness"], cmap="Reds"
        ),
        use_container_width=True,
        hide_index=True,
    )
with c_best:
    st.markdown(f"### 🟢 Migliori {top_n}")
    st.dataframe(
        best[rank_cols].style.format({"badness": "{:.2f}"}).background_gradient(
            subset=["badness"], cmap="Greens_r"
        ),
        use_container_width=True,
        hide_index=True,
    )

with st.expander("Anteprima dati filtrati (grezzi)"):
    st.dataframe(filtered.head(200), use_container_width=True)
