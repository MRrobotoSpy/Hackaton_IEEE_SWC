import math
import time
from pathlib import Path

import folium
import pandas as pd
import pydeck as pdk
import streamlit as st
from streamlit_folium import st_folium

import router as rtr
from onda import dashboard as onda_dash
from onda import llm as onda_llm


def _pos_at(coords: list, times: list, t: float) -> tuple[float, float] | None:
    if not coords or not times or len(coords) != len(times):
        return None
    if t <= times[0]:
        return coords[0]
    if t >= times[-1]:
        return coords[-1]
    for i in range(len(times) - 1):
        if times[i] <= t <= times[i + 1]:
            span = times[i + 1] - times[i]
            frac = 0.0 if span <= 0 else (t - times[i]) / span
            y0, x0 = coords[i]
            y1, x1 = coords[i + 1]
            return (y0 + (y1 - y0) * frac, x0 + (x1 - x0) * frac)
    return None


def _speed_at(edge_speeds_kph: list, times: list, t: float) -> float | None:
    """Return the speed on the edge the vehicle is currently on, or None if
    the vehicle hasn't started / has arrived."""
    if not edge_speeds_kph or not times or len(times) != len(edge_speeds_kph) + 1:
        return None
    if t < times[0] or t >= times[-1]:
        return None
    for i in range(len(times) - 1):
        if times[i] <= t < times[i + 1]:
            return float(edge_speeds_kph[i])
    return None


def _haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    r = 6_371_000.0
    lat1, lng1, lat2, lng2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    dlat, dlng = lat2 - lat1, lng2 - lng1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))

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
    return pd.read_csv(DATA_PATH, parse_dates=["timestamp"])


@st.cache_resource(show_spinner="Loading NYC road graph from OpenStreetMap...")
def load_base_graph(bbox: tuple[float, float, float, float]):
    return rtr.load_road_graph(bbox)


@st.cache_data
def global_metric_ranges() -> dict:
    d = load_data()
    inters = d.groupby(["intersection_id", "latitude", "longitude"], as_index=False)[
        ["congestion_score", "emission_estimate", "fuel_waste_estimate"]
    ].mean()
    return {
        c: (float(inters[c].min()), float(inters[c].max()))
        for c in ["congestion_score", "emission_estimate", "fuel_waste_estimate"]
    }


@st.cache_resource(show_spinner="Enriching graph with time-of-day metrics...")
def enrich_graph_for_time(
    bbox: tuple[float, float, float, float],
    hour: int,
    day_kind: str,
):
    base = load_base_graph(bbox)
    G = base.copy()
    d = load_data()
    d = d[d["hour"] == hour]
    if day_kind == "Weekday":
        d = d[d["is_weekend"] == 0]
    elif day_kind == "Weekend":
        d = d[d["is_weekend"] == 1]
    inters = d.groupby(["intersection_id", "latitude", "longitude"], as_index=False)[
        ["congestion_score", "emission_estimate", "fuel_waste_estimate", "average_speed"]
    ].mean()
    G = rtr.enrich_graph(G, inters, global_ranges=global_metric_ranges())
    G = rtr.add_weights(G)
    return G


df = load_data()

st.title("CityFlow — 3D Traffic Explorer")

with st.sidebar:
    st.header("Filters (Explorer tab)")

    zones = sorted(df["city_zone"].unique())
    selected_zones = st.multiselect("City zone", zones, default=zones)

    hour_range = st.slider("Hour range", 0, 23, (0, 23))

    days_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    selected_days = st.multiselect(
        "Day of week",
        options=list(days_map.keys()),
        default=list(days_map.keys()),
        format_func=lambda d: days_map[d],
    )

    metric = st.selectbox(
        "Metric (column height)",
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
        "3D layer type",
        [
            "Columns per intersection",
            "Precise scatter (individual points)",
            "Aggregated hexagon",
        ],
        index=0,
    )

    agg_fn_label = st.selectbox(
        "Time aggregation per intersection",
        ["Mean", "Max", "Sum", "Last value"],
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
    st.subheader("Intersection ranking")
    show_ranking = st.checkbox("Highlight worst/best on the map", value=True)
    top_n = st.slider("How many to highlight (per side)", 3, 20, 10)

tab_explorer, tab_router, tab_onda = st.tabs(
    ["🗺️ Explorer 3D", "🚗 Route Planner", "🌊 ONDA Scenarios"]
)

# ─────────────────────────── EXPLORER TAB ───────────────────────────
with tab_explorer:
    st.caption("Right-click drag to tilt/rotate the map (Google-Maps-3D style).")

    mask = (
        df["city_zone"].isin(selected_zones)
        & df["hour"].between(hour_range[0], hour_range[1])
        & df["day_of_week"].isin(selected_days)
    )
    filtered = df.loc[mask]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Records", f"{len(filtered):,}")
    col2.metric("Intersections", filtered["intersection_id"].nunique())
    col3.metric("Avg congestion", f"{filtered['congestion_score'].mean():.1f}")
    col4.metric("Avg speed", f"{filtered['average_speed'].mean():.1f} km/h")

    if filtered.empty:
        st.warning("No data with the current filters.")
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

    agg_fn = {"Mean": "mean", "Max": "max", "Sum": "sum", "Last value": "last"}[
        agg_fn_label
    ]

    if layer_type == "Aggregated hexagon":
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
        tooltip = {"text": "Elevation ≈ sum of {}".format(metric)}
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

        if layer_type == "Columns per intersection":
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
        filtered.groupby(
            ["intersection_id", "city_zone", "latitude", "longitude"], as_index=False
        )[BAD_METRICS + ["average_speed"]].mean()
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

    st.subheader("Intersection comparison")
    st.caption(
        "'Badness' score = normalized mean (0–1) of congestion, emissions, fuel waste, "
        "AQI, wait time. Higher = worse."
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
        st.markdown(f"### 🔴 Worst {top_n}")
        st.dataframe(
            worst[rank_cols].style.format({"badness": "{:.2f}"}).background_gradient(
                subset=["badness"], cmap="Reds"
            ),
            use_container_width=True,
            hide_index=True,
        )
    with c_best:
        st.markdown(f"### 🟢 Best {top_n}")
        st.dataframe(
            best[rank_cols].style.format({"badness": "{:.2f}"}).background_gradient(
                subset=["badness"], cmap="Greens_r"
            ),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Filtered data preview (raw)"):
        st.dataframe(filtered.head(200), use_container_width=True)

# ─────────────────────────── ROUTE PLANNER TAB ───────────────────────────
with tab_router:
    st.subheader("Plan a route")

    for k in ("car_start", "car_end", "amb_start", "amb_end"):
        st.session_state.setdefault(k, None)

    mode_col, hour_col, day_col = st.columns([2, 2, 2])
    with mode_col:
        mode = st.radio(
            "Mode",
            ["🚗 Car only", "🚗 + 🚑 Car with ambulance"],
            index=0,
            help=(
                "Car only: two clicks (start / end) and see 3 alternative routes.\n\n"
                "Ambulance: four clicks. First the car, then the ambulance. "
                "The car is rerouted to clear the ambulance's corridor."
            ),
        )
    with hour_col:
        dep_hour = st.slider("🕐 Departure time (hour)", 0, 23, 8)
    with day_col:
        day_kind = st.radio(
            "📅 Day type",
            ["All", "Weekday", "Weekend"],
            index=1,
            horizontal=True,
        )

    is_amb_mode = mode.startswith("🚗 + 🚑")

    if is_amb_mode:
        buf_col, disp_col, win_col = st.columns([2, 2, 2])
        with buf_col:
            corridor_buffer_m = st.slider(
                "🚑 Protected corridor width (m)",
                100, 2000, 1200, step=100,
                help=(
                    "Radius around the ambulance's route inside which cars are diverted. "
                    "On Manhattan you need ~1200 m to force a visible detour, because "
                    "there are two parallel highways (FDR / West Side)."
                ),
            )
        with disp_col:
            dispatch_offset_s = st.slider(
                "⏱️ Ambulance dispatched after (s)",
                0, 1800, 700, step=30,
                help=(
                    "How many seconds *after* the car departs the ambulance is called. "
                    "Since both travel at the same observed speed, the timing gap between "
                    "them is roughly constant along the shared corridor — so either most "
                    "shared nodes are conflicts, or none of them are."
                ),
            )
        with win_col:
            meeting_window_s = st.slider(
                "🎯 Conflict window (s)",
                10, 180, 60, step=5,
                help=(
                    "Tolerance on 'same moment'. If car and ambulance are at the same "
                    "intersection within this many seconds of each other, it's a conflict."
                ),
            )

        sim_col1, sim_col2 = st.columns([1, 2])
        with sim_col1:
            simulate_ambient = st.checkbox(
                "🚦 Simulate ambient traffic",
                value=True,
                help=(
                    "Sample random cars whose route crosses the ambulance corridor "
                    "and show all of them getting rerouted."
                ),
            )
        with sim_col2:
            n_ambient_cars = st.slider(
                "Number of simulated cars",
                3, 40, 15, step=1,
                disabled=not simulate_ambient,
            )
    else:
        corridor_buffer_m = 800
        dispatch_offset_s = 0
        meeting_window_s = 30
        simulate_ambient = False
        n_ambient_cars = 0
    slot_labels = {
        "car_start": "🚗 Car start",
        "car_end": "🏁 Car end",
        "amb_start": "🚑 Amb start",
        "amb_end": "🏥 Amb end",
    }
    slots = ["car_start", "car_end"] + (["amb_start", "amb_end"] if is_amb_mode else [])

    def _label(key: str) -> str:
        p = st.session_state[key]
        return f"{p[0]:.5f}, {p[1]:.5f}" if p else "—"

    # Auto-select next empty slot; user can override to update a specific point.
    # This block runs BEFORE the st.radio(key="target_slot") widget so it can
    # legally seed / advance the session-state value.
    if "target_slot" not in st.session_state or st.session_state.target_slot not in slots:
        st.session_state["target_slot"] = next(
            (s for s in slots if st.session_state[s] is None), slots[0]
        )
    else:
        current = st.session_state["target_slot"]
        if st.session_state[current] is not None:
            remaining = [s for s in slots if st.session_state[s] is None]
            if remaining:
                st.session_state["target_slot"] = remaining[0]

    st.caption(
        "**Click on the map** to place / move the selected point. "
        "The selector auto-advances to the next empty slot, or pick any slot to "
        "move only that one — no need to reset everything."
    )

    st.radio(
        "The next map click will update:",
        options=slots,
        format_func=lambda s: f"{slot_labels[s]}  ({_label(s)})",
        horizontal=is_amb_mode,
        key="target_slot",
    )

    if is_amb_mode:
        info_cols = st.columns(4)
        for col, key in zip(info_cols, slots):
            marker = "🟢 " if key == st.session_state.target_slot else ""
            col.info(f"**{marker}{slot_labels[key]}**\n\n{_label(key)}")
    else:
        info_cols = st.columns(2)
        for col, key in zip(info_cols, slots):
            marker = "🟢 " if key == st.session_state.target_slot else ""
            col.info(f"**{marker}{slot_labels[key]}**\n\n{_label(key)}")

    ctrl1, ctrl2 = st.columns([1, 1])
    with ctrl1:
        if st.button("🔄 Reset points", use_container_width=True):
            for k in ("car_start", "car_end", "amb_start", "amb_end"):
                st.session_state[k] = None
            st.rerun()
    with ctrl2:
        example_label = (
            "📍 Example (car + ambulance)" if is_amb_mode
            else "📍 Example (Fin. Dist. → Central Park)"
        )
        if st.button(example_label, use_container_width=True):
            st.session_state.car_start = (40.7075, -74.0113)
            st.session_state.car_end = (40.7829, -73.9654)
            if is_amb_mode:
                st.session_state.amb_start = (40.7299, -73.9914)
                st.session_state.amb_end = (40.7623, -73.9702)
            else:
                st.session_state.amb_start = None
                st.session_state.amb_end = None
            st.rerun()

    bbox = (
        float(df["latitude"].min()),
        float(df["longitude"].min()),
        float(df["latitude"].max()),
        float(df["longitude"].max()),
    )

    G = None
    routes = None
    reroute = None
    corridor_polygon = None
    ambient_sim = None
    err_msg = None

    car_ready = st.session_state.car_start and st.session_state.car_end
    amb_ready = st.session_state.amb_start and st.session_state.amb_end
    if car_ready:
        try:
            G = enrich_graph_for_time(bbox, dep_hour, day_kind)
            if is_amb_mode and amb_ready:
                reroute = rtr.compute_ambulance_reroute(
                    G,
                    st.session_state.car_start,
                    st.session_state.car_end,
                    st.session_state.amb_start,
                    st.session_state.amb_end,
                    corridor_buffer_m=corridor_buffer_m,
                    ambulance_dispatch_offset_s=dispatch_offset_s,
                    meeting_window_s=meeting_window_s,
                )
                corridor_polygon = rtr.compute_corridor_polygon(
                    G, reroute["ambulance"]["path"], corridor_buffer_m
                )
                if simulate_ambient and n_ambient_cars > 0:
                    ambient_sim = rtr.simulate_corridor_traffic(
                        G,
                        reroute["ambulance"]["path"],
                        corridor_buffer_m,
                        n_cars=n_ambient_cars,
                    )
            elif not is_amb_mode:
                routes = rtr.compute_routes(
                    G,
                    st.session_state.car_start[0],
                    st.session_state.car_start[1],
                    st.session_state.car_end[0],
                    st.session_state.car_end[1],
                )
        except Exception as exc:
            err_msg = str(exc)

    if err_msg:
        st.error(f"Error computing the route: {err_msg}")

    center = st.session_state.car_start or (
        float(df["latitude"].mean()),
        float(df["longitude"].mean()),
    )
    fmap = folium.Map(location=list(center), zoom_start=13, tiles="OpenStreetMap")

    marker_specs = [
        ("car_start", "🚗 Car — Start", "green", "play"),
        ("car_end", "🏁 Car — End", "darkgreen", "flag"),
        ("amb_start", "🚑 Ambulance — Start", "red", "plus"),
        ("amb_end", "🏥 Ambulance — Hospital", "darkred", "heart"),
    ]
    for key, popup, color, icon in marker_specs:
        p = st.session_state[key]
        if p is None or (key.startswith("amb_") and not is_amb_mode):
            continue
        is_target = key == st.session_state.target_slot
        folium.Marker(
            location=list(p),
            popup=f"{popup}<br/>{'👉 currently selected' if is_target else 'select this slot above, then click on the map to move it'}",
            tooltip=slot_labels[key] + (" (selected)" if is_target else ""),
            icon=folium.Icon(
                color=color,
                icon=icon,
                prefix="fa",
                icon_color="white" if not is_target else "yellow",
            ),
        ).add_to(fmap)

    if reroute:
        if corridor_polygon:
            folium.Polygon(
                locations=corridor_polygon,
                color="#e74c3c",
                weight=1,
                fill=True,
                fill_color="#e74c3c",
                fill_opacity=0.12,
                tooltip=f"Protected corridor ({corridor_buffer_m} m buffer)",
            ).add_to(fmap)

        if ambient_sim and ambient_sim["cars"]:
            for c in ambient_sim["cars"]:
                folium.PolyLine(
                    locations=c["orig_coords"],
                    color="#95a5a6",
                    weight=2,
                    opacity=0.5,
                    dash_array="4, 6",
                    tooltip=f"Ambient car — original ({c['orig_time_min']:.1f} min)",
                ).add_to(fmap)
                folium.PolyLine(
                    locations=c["new_coords"],
                    color="#2ecc71",
                    weight=3,
                    opacity=0.75,
                    tooltip=(
                        f"Ambient car — rerouted (+{c['extra_time_min']:.1f} min, "
                        f"{c['tokens_earned']} tokens)"
                    ),
                ).add_to(fmap)

        folium.PolyLine(
            locations=reroute["car_original"]["coords"],
            color="#7f8c8d",
            weight=5,
            opacity=0.55,
            dash_array="8, 12",
            tooltip="Car — original route",
        ).add_to(fmap)
        folium.PolyLine(
            locations=reroute["car_rerouted"]["coords"],
            color="#1f78b4",
            weight=6,
            opacity=0.9,
            tooltip="Car — rerouted",
        ).add_to(fmap)
        folium.PolyLine(
            locations=reroute["ambulance"]["coords"],
            color="#e74c3c",
            weight=8,
            opacity=0.9,
            tooltip="Ambulance — priority corridor",
        ).add_to(fmap)
        for c in reroute["conflicts"][:20]:
            folium.CircleMarker(
                location=list(c["coords"]),
                radius=8,
                color="#f1c40f",
                fill=True,
                fill_color="#f1c40f",
                fill_opacity=0.9,
                tooltip=(
                    f"⚠️ Conflict: car at t={c['t_car_s']:.0f}s, "
                    f"ambulance at t={c['t_amb_s']:.0f}s (Δ={c['dt_s']:+.0f}s)"
                ),
            ).add_to(fmap)
    elif routes:
        route_colors = {"Fastest": "#1f78b4", "Balanced": "#ff7f00", "Eco": "#33a02c"}
        for name, r in routes.items():
            folium.PolyLine(
                locations=r["coords"],
                color=route_colors[name],
                weight=6,
                opacity=0.75,
                tooltip=name,
            ).add_to(fmap)

    click_result = st_folium(
        fmap,
        key="route_map",
        height=550,
        use_container_width=True,
        returned_objects=["last_clicked"],
    )

    last = click_result.get("last_clicked") if click_result else None
    if last:
        clicked = (round(last["lat"], 6), round(last["lng"], 6))
        target = st.session_state.target_slot
        if st.session_state.get(target) != clicked:
            st.session_state[target] = clicked
            # Do NOT modify target_slot here — the pre-radio block will advance
            # it on the next run (Streamlit forbids writing to a widget's key
            # after the widget is instantiated).
            st.rerun()

    if reroute:
        st.subheader("Comparison: what happens when the ambulance arrives")

        legend_cols = st.columns(3)
        legend_cols[0].markdown(
            "<div style='background:#e74c3c;padding:8px;border-radius:6px;"
            "color:white;text-align:center;font-weight:600'>🚑 Ambulance (priority corridor)</div>",
            unsafe_allow_html=True,
        )
        legend_cols[1].markdown(
            "<div style='background:#7f8c8d;padding:8px;border-radius:6px;"
            "color:white;text-align:center;font-weight:600'>🚗 Car — original (blocked)</div>",
            unsafe_allow_html=True,
        )
        legend_cols[2].markdown(
            "<div style='background:#1f78b4;padding:8px;border-radius:6px;"
            "color:white;text-align:center;font-weight:600'>🚗 Car — rerouted</div>",
            unsafe_allow_html=True,
        )

        amb = reroute["ambulance"]["summary"]
        car_o = reroute["car_original"]["summary"]
        car_n = reroute["car_rerouted"]["summary"]
        info = reroute["reroute_info"]

        rows = [
            {
                "Route": "🚑 Ambulance (with corridor)",
                "Distance (km)": round(amb["distance_km"], 2),
                "Time (min)": round(amb["time_min"], 1),
                "Avg speed (kph)": round(amb["distance_km"] / max(amb["time_min"] / 60, 1e-6), 1),
                "Fuel (L)": round(amb["fuel_liters"], 2),
                "CO₂ (kg)": round(amb["co2_kg"], 2),
            },
            {
                "Route": "🚗 Car — original route",
                "Distance (km)": round(car_o["distance_km"], 2),
                "Time (min)": round(car_o["time_min"], 1),
                "Avg speed (kph)": round(car_o["distance_km"] / max(car_o["time_min"] / 60, 1e-6), 1),
                "Fuel (L)": round(car_o["fuel_liters"], 2),
                "CO₂ (kg)": round(car_o["co2_kg"], 2),
            },
            {
                "Route": "🚗 Car — rerouted",
                "Distance (km)": round(car_n["distance_km"], 2),
                "Time (min)": round(car_n["time_min"], 1),
                "Avg speed (kph)": round(car_n["distance_km"] / max(car_n["time_min"] / 60, 1e-6), 1),
                "Fuel (L)": round(car_n["fuel_liters"], 2),
                "CO₂ (kg)": round(car_n["co2_kg"], 2),
            },
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        col_a, col_b, col_c, col_d, col_e = st.columns(5)
        col_a.metric(
            "⚠️ Space-time conflicts",
            len(reroute["conflicts"]),
            help=(
                f"Nodes where car and ambulance would be at the same intersection "
                f"within ±{reroute['meeting_window_s']:.0f}s. If 0, no reroute needed."
            ),
        )
        col_b.metric(
            "⏱️ Ambulance time",
            f"{amb['time_min']:.1f} min",
            help=(
                "Ambulance travels at the same observed speed as regular traffic. "
                "The priority is spatial (cars reroute) rather than a speed boost."
            ),
        )
        col_c.metric(
            "🚗 Car extra time",
            f"+{info['extra_time_min']:.1f} min",
            delta=f"+{info['extra_distance_km']:.2f} km",
            delta_color="inverse",
        )
        col_d.metric(
            "🎟️ Tokens earned",
            int(info["tokens_earned"]),
            help=(
                "Reward for the driver who accepts to detour to let the ambulance pass.\n"
                "10 tokens per minute of extra travel × equity multiplier."
            ),
        )
        col_e.metric(
            "🛣️ Reroute happened?",
            "Yes" if reroute["car_rerouted"]["reroute_happened"] else "No",
            help=(
                "Yes when the car's original route crosses the protected corridor "
                "(spatial rule). It reroutes as a precaution even if no exact "
                "space-time conflict occurs."
            ),
        )

        crosses_buffer = reroute["car_original"].get("crosses_corridor_buffer", False)
        if not crosses_buffer:
            st.info(
                "The car's original route does not enter the protected corridor — "
                "no reroute needed."
            )
        elif not reroute["conflicts"]:
            st.warning(
                "🛣️ The car's original route enters the protected corridor, so it is "
                "rerouted as a precaution. **No space-time conflict** was detected — "
                "the car and ambulance would not have been at the same intersection at "
                "the same moment — but the corridor stays clear."
            )
        else:
            st.error(
                f"⚠️ **{len(reroute['conflicts'])} space-time conflicts** detected "
                f"(yellow dots on the map). Rerouting prevents an actual collision."
            )

        st.caption(
            f"Car departs at **{dep_hour:02d}:00** ({day_kind.lower()}). "
            f"Ambulance dispatched **{dispatch_offset_s}s** after the car left. "
            f"Conflict window: ±{meeting_window_s}s. "
            "Both vehicles travel at the observed average speed at each intersection — "
            "the ambulance's priority is only spatial (cars step aside)."
        )

        # ─── Ambient traffic simulation ───
        if ambient_sim is not None:
            st.divider()
            st.markdown("### 🚦 Ambient traffic inside the corridor")
            st.caption(
                "Random cars whose fastest route naturally crosses the ambulance corridor. "
                "Grey dashed = original path (blocked by the corridor). "
                "Green = the reroute they must take."
            )
            n_shown = len(ambient_sim["cars"])
            n_nodes_in = ambient_sim.get("n_nodes_in_corridor", 0)
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("🚗 Cars rerouted", n_shown)
            s2.metric(
                "🛰️ Intersections in corridor",
                n_nodes_in,
                help=(
                    "OSM nodes falling inside the buffer polygon around the ambulance path. "
                    "This is the population the sampler draws from."
                ),
            )
            s3.metric(
                "⏱️ Total extra time",
                f"{ambient_sim['total_extra_time_min']:.1f} min",
            )
            s4.metric(
                "🎟️ Total tokens paid",
                int(ambient_sim["total_tokens"]),
                help="Sum across all simulated cars — this is how much the city compensates the drivers who made room.",
            )
            if n_shown:
                per_car_df = pd.DataFrame(
                    [
                        {
                            "Car": i + 1,
                            "Original (min)": round(c["orig_time_min"], 1),
                            "Rerouted (min)": round(c["new_time_min"], 1),
                            "Δ time (min)": round(c["extra_time_min"], 2),
                            "Δ distance (km)": round(c["extra_distance_km"], 2),
                            "Tokens": int(c["tokens_earned"]),
                        }
                        for i, c in enumerate(ambient_sim["cars"])
                    ]
                )
                st.dataframe(per_car_df, use_container_width=True, hide_index=True)
            else:
                st.info(
                    "No sampled car ended up needing a real detour — try a bigger "
                    "corridor width or more sampled cars."
                )

        # ─── Time animation ───
        st.divider()
        st.markdown("### 🎥 Animation — do they really meet?")
        st.caption(
            "Drag the slider or press **Play** to advance time. "
            "**Left = ACTUAL** (car reroutes, ambulance free). "
            "**Right = HYPOTHETICAL** (car ignores the reroute — see them meet). "
            "The right map turns yellow when the car is within 150 m of the ambulance."
        )

        amb_end_t = float(reroute["ambulance"]["times"][-1]) if reroute["ambulance"]["times"] else 0.0
        car_orig_end_t = float(reroute["car_original"]["times"][-1]) if reroute["car_original"]["times"] else 0.0
        car_new_end_t = float(reroute["car_rerouted"]["times"][-1]) if reroute["car_rerouted"]["times"] else 0.0
        max_t = int(math.ceil(max(amb_end_t, car_orig_end_t, car_new_end_t, 1.0)))

        anim_ctrl = st.columns([1, 3, 2])
        with anim_ctrl[0]:
            play = st.button("▶ Play", use_container_width=True, key="anim_play")
        with anim_ctrl[1]:
            t_cursor = st.slider(
                "Time (s from car departure)",
                0, max_t, min(max_t // 2, max_t), step=5, key="anim_slider",
            )
        with anim_ctrl[2]:
            step_s = st.slider("Play step (s)", 5, 60, 15, step=5, key="anim_step")

        anim_placeholder = st.empty()

        all_lats_view = (
            [c[0] for c in reroute["ambulance"]["coords"]]
            + [c[0] for c in reroute["car_rerouted"]["coords"]]
            + [c[0] for c in reroute["car_original"]["coords"]]
        )
        all_lngs_view = (
            [c[1] for c in reroute["ambulance"]["coords"]]
            + [c[1] for c in reroute["car_rerouted"]["coords"]]
            + [c[1] for c in reroute["car_original"]["coords"]]
        )
        shared_view = pdk.ViewState(
            latitude=sum(all_lats_view) / len(all_lats_view),
            longitude=sum(all_lngs_view) / len(all_lngs_view),
            zoom=12,
            pitch=0,
            bearing=0,
        )

        def _build_side_deck(
            car_coords: list,
            car_pos: tuple | None,
            car_color: list[int],
            amb_pos: tuple | None,
            highlight_conflict: bool = False,
        ) -> pdk.Deck:
            layers = [
                pdk.Layer(
                    "PathLayer",
                    data=[{"path": [[c[1], c[0]] for c in reroute["ambulance"]["coords"]]}],
                    get_path="path",
                    get_color=[231, 76, 60, 200],
                    width_min_pixels=5,
                    get_width=6,
                ),
                pdk.Layer(
                    "PathLayer",
                    data=[{"path": [[c[1], c[0]] for c in car_coords]}],
                    get_path="path",
                    get_color=car_color + [200],
                    width_min_pixels=4,
                    get_width=5,
                ),
            ]
            markers = []
            if amb_pos is not None:
                markers.append({
                    "pos": [amb_pos[1], amb_pos[0]],
                    "color": [231, 76, 60, 255],
                    "radius": 45 if highlight_conflict else 40,
                    "label": "AMB",
                })
            if car_pos is not None:
                marker_color = (
                    [255, 193, 7, 255] if highlight_conflict else car_color + [255]
                )
                markers.append({
                    "pos": [car_pos[1], car_pos[0]],
                    "color": marker_color,
                    "radius": 45 if highlight_conflict else 35,
                    "label": "CAR",
                })
            if markers:
                layers.append(
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=markers,
                        get_position="pos",
                        get_fill_color="color",
                        get_radius="radius",
                        radius_min_pixels=8,
                        radius_max_pixels=24,
                        stroked=True,
                        get_line_color=[255, 255, 255, 255],
                        line_width_min_pixels=2,
                        pickable=True,
                    )
                )
            return pdk.Deck(
                layers=layers,
                initial_view_state=shared_view,
                map_style="road",
                tooltip={"text": "{label}"},
            )

        def _render_frame(t: float):
            amb_pos = _pos_at(reroute["ambulance"]["coords"], reroute["ambulance"]["times"], t)
            car_new_pos = _pos_at(
                reroute["car_rerouted"]["coords"], reroute["car_rerouted"]["times"], t
            )
            car_orig_pos = _pos_at(
                reroute["car_original"]["coords"], reroute["car_original"]["times"], t
            )

            dist_actual = (
                _haversine_m(amb_pos, car_new_pos) if (amb_pos and car_new_pos) else None
            )
            dist_hypo = (
                _haversine_m(amb_pos, car_orig_pos) if (amb_pos and car_orig_pos) else None
            )
            hypo_conflict = dist_hypo is not None and dist_hypo < 150

            amb_speed = _speed_at(
                reroute["ambulance"]["edge_speeds_kph"],
                reroute["ambulance"]["times"], t,
            )
            car_speed = _speed_at(
                reroute["car_rerouted"]["edge_speeds_kph"],
                reroute["car_rerouted"]["times"], t,
            )
            car_orig_speed = _speed_at(
                reroute["car_original"]["edge_speeds_kph"],
                reroute["car_original"]["times"], t,
            )

            def _kph(v):
                return f"{v:.0f} kph" if v is not None else "—"

            deck_actual = _build_side_deck(
                car_coords=reroute["car_rerouted"]["coords"],
                car_pos=car_new_pos,
                car_color=[31, 120, 180],
                amb_pos=amb_pos,
                highlight_conflict=False,
            )
            deck_hypo = _build_side_deck(
                car_coords=reroute["car_original"]["coords"],
                car_pos=car_orig_pos,
                car_color=[127, 140, 141],
                amb_pos=amb_pos,
                highlight_conflict=hypo_conflict,
            )

            with anim_placeholder.container():
                st.metric("⏱️ Time from car departure", f"{t:.0f} s")

                left, right = st.columns(2)
                with left:
                    st.markdown("#### ✅ ACTUAL — car reroutes")
                    st.pydeck_chart(deck_actual, use_container_width=True, height=440)
                    a1, a2, a3 = st.columns(3)
                    a1.metric("🚑 Amb speed", _kph(amb_speed))
                    a2.metric("🚗 Car speed", _kph(car_speed))
                    a3.metric(
                        "🚑 ↔ 🚗 Distance",
                        f"{dist_actual:.0f} m" if dist_actual is not None else "—",
                        delta=(
                            "Safe ✓"
                            if (dist_actual is not None and dist_actual >= 200)
                            else None
                        ),
                    )
                with right:
                    st.markdown(
                        "#### ⚠️ HYPOTHETICAL — car ignores reroute"
                        if hypo_conflict
                        else "#### 🟠 HYPOTHETICAL — car ignores reroute"
                    )
                    st.pydeck_chart(deck_hypo, use_container_width=True, height=440)
                    h1, h2, h3 = st.columns(3)
                    h1.metric("🚑 Amb speed", _kph(amb_speed))
                    h2.metric("⚪ Car speed", _kph(car_orig_speed))
                    h3.metric(
                        "🚑 ↔ ⚪ Distance",
                        f"{dist_hypo:.0f} m" if dist_hypo is not None else "—",
                        delta=(
                            "⚠️ COLLISION" if hypo_conflict else None
                        ),
                        delta_color="inverse",
                    )

                st.caption(
                    f"Ambulance dispatched at t = {reroute['ambulance_dispatch_offset_s']:.0f} s "
                    "(after car departure). "
                    "On the right, the car — if it had *not* rerouted — would have crossed "
                    "the ambulance's path. The yellow highlight marks the meeting."
                )

        if play:
            for tt in range(0, max_t + 1, step_s):
                _render_frame(tt)
                time.sleep(0.15)
        else:
            _render_frame(t_cursor)

        # ─── LLM explanation ───
        st.divider()
        st.markdown("### 🤖 Why is this reroute needed?")
        st.caption(
            "Ask `qwen3:1.7b` (running locally via Ollama) to explain the scenario "
            "in plain English, including which streets the car will take."
        )
        if st.button("Explain and compute tokens", type="primary", key="llm_reroute_btn"):
            ollama_ok, ollama_err = onda_llm.is_available()
            if not ollama_ok:
                st.warning(
                    f"Ollama is not reachable ({ollama_err}). Start `ollama serve` "
                    "and make sure `qwen3:1.7b` has been pulled."
                )
            else:
                info = reroute["reroute_info"]

                def _fmt_streets(names: list) -> str:
                    if not names:
                        return "(no named streets available)"
                    return " → ".join(names[:15]) + (" → …" if len(names) > 15 else "")

                context = (
                    f"Scenario (car departure at {dep_hour:02d}:00, {day_kind.lower()}):\n"
                    f"- Ambulance corridor: {amb['distance_km']:.2f} km, "
                    f"{amb['time_min']:.2f} min at green-wave speed, dispatched "
                    f"{reroute['ambulance_dispatch_offset_s']:.0f} s after the car left.\n"
                    f"- Ambulance streets: {_fmt_streets(reroute['ambulance']['streets'])}\n"
                    f"- Car ORIGINAL route: {car_o['distance_km']:.2f} km, "
                    f"{car_o['time_min']:.2f} min\n"
                    f"- Car ORIGINAL streets: {_fmt_streets(reroute['car_original']['streets'])}\n"
                    f"- Car REROUTED route: {car_n['distance_km']:.2f} km, "
                    f"{car_n['time_min']:.2f} min\n"
                    f"- Car REROUTED streets: {_fmt_streets(reroute['car_rerouted']['streets'])}\n"
                    f"- Shared nodes between the two routes (spatial overlap): "
                    f"{reroute['shared_nodes_count']}\n"
                    f"- Meeting window: ±{reroute['meeting_window_s']:.0f} s\n"
                    f"- Space-time conflicts detected: {len(reroute['conflicts'])}\n"
                    f"- Protected corridor width: {corridor_buffer_m} m\n"
                    f"- Reroute happened: {reroute['car_rerouted']['reroute_happened']}\n"
                    f"- Extra time for the car: +{info['extra_time_min']:.2f} min\n"
                    f"- Extra distance for the car: +{info['extra_distance_km']:.2f} km\n"
                    f"- Equity multiplier: {info['equity_multiplier']:.1f}\n"
                    f"- Tokens earned: {info['tokens_earned']}\n"
                )
                question = (
                    "In English, 4-5 short sentences: "
                    "(a) Why is the reroute needed (or NOT needed) — refer to the "
                    "conflicts count and dispatch timing. "
                    "(b) Give the exact turn-by-turn street list the driver should now "
                    "follow (use the 'Car REROUTED streets' list — quote the actual names). "
                    "(c) State how many tokens are earned and show the formula: "
                    "round(extra_time_min × 10 × equity_multiplier). "
                    "(d) What would change if the ambulance were dispatched much earlier "
                    "or much later? Be specific about which side of the timeline avoids "
                    "the conflict."
                )
                try:
                    st.write_stream(
                        onda_llm.stream_answer(
                            question,
                            context,
                            model="qwen3:1.7b",
                            system_prompt=onda_llm.REROUTE_SYSTEM_PROMPT,
                        )
                    )
                except Exception as exc:
                    st.error(f"LLM error: {exc}")
    elif routes:
        st.subheader("Route comparison")
        rows = []
        for name, r in routes.items():
            s = r["summary"]
            rows.append(
                {
                    "Route": name,
                    "Distance (km)": round(s["distance_km"], 2),
                    "Estimated time (min)": round(s["time_min"], 1),
                    "Fuel (L)": round(s["fuel_liters"], 2),
                    "CO₂ (kg)": round(s["co2_kg"], 2),
                    "Avg congestion (0–1)": round(s["avg_congestion"], 2),
                }
            )
        route_df = pd.DataFrame(rows)

        route_colors = {"Fastest": "#1f78b4", "Balanced": "#ff7f00", "Eco": "#33a02c"}
        legend_cols = st.columns(3)
        for col, (name, color) in zip(legend_cols, route_colors.items()):
            col.markdown(
                f"<div style='background:{color};padding:8px;border-radius:6px;"
                f"color:white;text-align:center;font-weight:600'>{name}</div>",
                unsafe_allow_html=True,
            )
        st.dataframe(route_df, use_container_width=True, hide_index=True)

        st.caption(
            f"Departure at **{dep_hour:02d}:00** ({day_kind.lower()}). "
            "Fuel and CO₂ estimated assuming an average vehicle (baseline 0.08 L/km, "
            "190 g CO₂/km) modulated by the observed emission/congestion intensity "
            "on the traversed segments in that hour slot."
        )
    elif is_amb_mode and car_ready and not amb_ready:
        st.info("Now click 2 more times to set the ambulance start and end.")
    elif not car_ready:
        st.info("Click on the map to set the car's start and end.")

# ─────────────────────────── ONDA SCENARIOS TAB ───────────────────────────
with tab_onda:
    scenarios = onda_dash.load_scenarios()
    if not scenarios:
        st.error(
            "`out/scenarios.json` not found. Generate it with:\n\n"
            "```bash\npython -m onda.fake_scenarios\n```"
        )
        st.stop()

    st.subheader("Can smart mobility be efficient without being unfair?")
    st.caption(
        "Three scenarios on the same 100 NYC intersections, "
        f"snapshot **{scenarios['meta']['hour']:02d}:00 {scenarios['meta']['day_kind']}**. "
        "**A** = today. **B** = efficiency-only optimizer (the villain). "
        "**C** = ONDA (adaptive signals + diversion under a hard *no zone gets worse* constraint)."
    )

    scen_col, alpha_col = st.columns([1, 2])
    with scen_col:
        current_scenario = st.radio(
            "Scenario",
            ["A", "B", "C"],
            format_func=lambda k: f"{k} — {scenarios['scenarios'][k]['label']}",
            index=2,
            horizontal=False,
        )
    with alpha_col:
        st.markdown("**KPI vs baseline (A)**")
        onda_dash.render_kpi_row(scenarios, current_scenario)

    st.divider()

    left, right = st.columns([3, 2])
    with left:
        st.markdown("### 🗺️ Wait per intersection (Δ vs today)")
        onda_dash.render_map(scenarios, current_scenario)
    with right:
        st.markdown("### 📊 Wait per zone")
        onda_dash.render_per_zone_chart(scenarios, current_scenario)

    st.divider()

    st.markdown("### 🎯 The trade-off chart")
    st.caption(
        "This is the one chart that answers the challenge question. "
        "No other team will show its own failure mode."
    )
    onda_dash.render_efficiency_chart(scenarios)

    st.divider()

    pol_col, rew_col = st.columns(2)
    with pol_col:
        st.markdown("### 🏛️ Policy inputs (jury-editable)")
        alpha, vuln_df = onda_dash.render_vulnerability_editor(scenarios)
    with rew_col:
        st.markdown("### 🎟️ Rewards — 'whoever makes room gets paid'")
        onda_dash.render_rewards_panel(scenarios, current_scenario)

    st.divider()

    st.markdown("### 🚑 Emergency corridor")
    onda_dash.render_emergency_summary(scenarios)

    st.divider()

    st.markdown("### 🤖 Explain this scenario (local LLM)")
    st.caption(
        "Ask a question about scenarios A, B, C. The model is grounded in the "
        "numbers you see above — it cannot invent metrics."
    )
    available, err = onda_llm.is_available()
    if not available:
        st.warning(
            "Ollama server is not reachable. Install it from https://ollama.com/download, "
            "then run:\n\n"
            "```bash\nollama pull qwen2.5:7b\nollama serve\n```"
        )
        st.caption(f"(underlying error: `{err}`)")
    else:
        models = onda_llm.list_models()
        default_idx = 0
        for i, m in enumerate(models):
            if m.startswith(("qwen3", "qwen2.5", "llama3.1", "llama3", "mistral", "phi3")):
                default_idx = i
                break
        model = st.selectbox(
            "Model",
            models or [onda_llm.DEFAULT_MODEL],
            index=default_idx if models else 0,
        )
        question = st.text_input(
            "Question",
            value="Why is scenario C better than B for the periphery?",
        )
        if st.button("Ask", type="primary"):
            summary = onda_dash.scenario_summary_text(scenarios)
            try:
                st.write_stream(onda_llm.stream_answer(question, summary, model=model))
            except Exception as exc:
                st.error(f"LLM error: {exc}")
