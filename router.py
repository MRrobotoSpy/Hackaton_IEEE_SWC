import math
from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point

PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)
GRAPH_CACHE = CACHE_DIR / "nyc_drive.graphml"

BASELINE_FUEL_L_PER_KM = 0.08
BASELINE_CO2_KG_PER_KM = 0.190


def load_road_graph(bbox: tuple[float, float, float, float], pad: float = 0.005) -> nx.MultiDiGraph:
    if GRAPH_CACHE.exists():
        return ox.load_graphml(GRAPH_CACHE)

    south, west, north, east = bbox
    G = ox.graph_from_bbox(
        bbox=(west - pad, south - pad, east + pad, north + pad),
        network_type="drive",
        simplify=True,
    )
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    ox.save_graphml(G, GRAPH_CACHE)
    return G


DEFAULT_FALLBACK_SPEED_KPH = 30.0


def enrich_graph(
    G: nx.MultiDiGraph,
    intersections: pd.DataFrame,
    global_ranges: dict | None = None,
) -> nx.MultiDiGraph:
    """For every edge (u, v), attach:
    - `cong_norm` / `emit_norm` / `fuel_norm`: nearest-intersection intensity at midpoint.
    - `obs_speed_kph`: observed avg speed at the intersection nearest to origin node `u`
       (so the edge inherits the speed of the intersection you just left, until the next node).
    """
    cols = ["congestion_score", "emission_estimate", "fuel_waste_estimate"]
    metrics = intersections[cols].to_numpy()
    if global_ranges:
        lo = np.array([global_ranges[c][0] for c in cols])
        hi = np.array([global_ranges[c][1] for c in cols])
    else:
        lo = metrics.min(axis=0)
        hi = metrics.max(axis=0)
    span = np.where(hi - lo == 0, 1, hi - lo)
    norm = np.clip((metrics - lo) / span, 0.0, 1.0)

    coords = intersections[["latitude", "longitude"]].to_numpy()
    tree = cKDTree(coords)

    has_speed = "average_speed" in intersections.columns
    speeds = intersections["average_speed"].to_numpy() if has_speed else None

    edge_updates = {}
    for u, v, k, d in G.edges(keys=True, data=True):
        mid_lat = (G.nodes[u]["y"] + G.nodes[v]["y"]) / 2
        mid_lng = (G.nodes[u]["x"] + G.nodes[v]["x"]) / 2
        _, idx_mid = tree.query([mid_lat, mid_lng])
        _, idx_u = tree.query([G.nodes[u]["y"], G.nodes[u]["x"]])
        update = {
            "cong_norm": float(norm[idx_mid, 0]),
            "emit_norm": float(norm[idx_mid, 1]),
            "fuel_norm": float(norm[idx_mid, 2]),
        }
        if speeds is not None:
            s = float(speeds[idx_u])
            update["obs_speed_kph"] = s if s > 1.0 else DEFAULT_FALLBACK_SPEED_KPH
        edge_updates[(u, v, k)] = update
    nx.set_edge_attributes(G, edge_updates)
    return G


def add_weights(
    G: nx.MultiDiGraph,
    alpha_time: float = 0.5,
    alpha_cong: float = 0.25,
    alpha_eco: float = 0.25,
) -> nx.MultiDiGraph:
    """Compute per-edge weights.

    Both car and ambulance use the OBSERVED speed from the dataset at the
    intersection the vehicle just left (congestion is already baked in).
    The ambulance's "priority corridor" is spatial only — cars step aside —
    not a speed boost.
    """
    for u, v, k, d in G.edges(keys=True, data=True):
        length_m = float(d.get("length", 0) or 0)
        length_km = length_m / 1000.0
        osm_time_s = float(
            d.get("travel_time", length_m / (30 / 3.6)) or length_m / (30 / 3.6)
        )
        obs_speed_kph = float(d.get("obs_speed_kph", DEFAULT_FALLBACK_SPEED_KPH))
        if obs_speed_kph <= 1.0:
            obs_speed_kph = DEFAULT_FALLBACK_SPEED_KPH
        car_time_s = length_m * 3.6 / obs_speed_kph
        # Ambulance travels at the same observed speed as the car on each edge.
        # The "priority corridor" now means only that the car will step aside
        # (physical reroute) — not that the ambulance moves faster.
        amb_corridor_time_s = car_time_s

        emit = float(d.get("emit_norm", 0.5))
        fuel = float(d.get("fuel_norm", 0.5))
        cong = float(d.get("cong_norm", 0.5))

        edge_fuel_l = length_km * BASELINE_FUEL_L_PER_KM * (1.0 + 3.0 * fuel)
        edge_co2_kg = length_km * BASELINE_CO2_KG_PER_KM * (1.0 + 3.0 * emit)
        eco_total = edge_fuel_l + edge_co2_kg

        d["osm_time_s"] = osm_time_s
        d["car_time_s"] = car_time_s
        d["amb_corridor_time_s"] = amb_corridor_time_s

        d["w_fast"] = car_time_s
        d["w_eco"] = eco_total
        d["w_balanced"] = (
            alpha_time * car_time_s
            + alpha_cong * car_time_s * (2.0 * cong)
            + alpha_eco * eco_total * 300.0
        )
    return G


def _summarize_path(G: nx.MultiDiGraph, path: list, corridor: bool = False) -> dict:
    """Summarize a path.

    - Car (corridor=False): uses `car_time_s` per edge (observed speed).
    - Ambulance (corridor=True): uses `amb_corridor_time_s` per edge (OSM posted
      speed × green-wave speedup).
    """
    time_s = 0.0
    dist_m = 0.0
    congs, emits, fuels, speeds = [], [], [], []
    for u, v in zip(path[:-1], path[1:]):
        edata = G.get_edge_data(u, v)
        d = min(edata.values(), key=lambda x: x.get("length", float("inf")))
        length_m = float(d.get("length", 0) or 0)
        dist_m += length_m
        if corridor:
            time_s += float(
                d.get("amb_corridor_time_s",
                      d.get("osm_time_s", length_m / (30 / 3.6)))
            )
        else:
            time_s += float(
                d.get("car_time_s", length_m / (30 / 3.6))
            )
        congs.append(float(d.get("cong_norm", 0.5)))
        emits.append(float(d.get("emit_norm", 0.5)))
        fuels.append(float(d.get("fuel_norm", 0.5)))
        speeds.append(float(d.get("obs_speed_kph", DEFAULT_FALLBACK_SPEED_KPH)))

    distance_km = dist_m / 1000
    avg_emit = float(np.mean(emits)) if emits else 0.5
    avg_fuel = float(np.mean(fuels)) if fuels else 0.5
    return {
        "distance_km": distance_km,
        "time_min": time_s / 60,
        "avg_congestion": float(np.mean(congs)) if congs else 0.5,
        "avg_emission_intensity": avg_emit,
        "avg_fuel_intensity": avg_fuel,
        "avg_observed_speed_kph": float(np.mean(speeds)) if speeds else DEFAULT_FALLBACK_SPEED_KPH,
        "fuel_liters": distance_km * BASELINE_FUEL_L_PER_KM * (1.0 + avg_fuel),
        "co2_kg": distance_km * BASELINE_CO2_KG_PER_KM * (1.0 + avg_emit),
    }


def compute_routes(
    G: nx.MultiDiGraph,
    start_lat: float,
    start_lng: float,
    end_lat: float,
    end_lng: float,
) -> dict:
    orig = ox.distance.nearest_nodes(G, X=start_lng, Y=start_lat)
    dest = ox.distance.nearest_nodes(G, X=end_lng, Y=end_lat)

    routes = {}
    for name, weight in [("Fastest", "w_fast"), ("Balanced", "w_balanced"), ("Eco", "w_eco")]:
        try:
            path = nx.shortest_path(G, orig, dest, weight=weight)
        except nx.NetworkXNoPath:
            continue
        coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in path]
        summary = _summarize_path(G, path)
        routes[name] = {"path": path, "coords": coords, "summary": summary}
    return routes


REROUTE_PENALTY = 1000.0
TOKENS_PER_EXTRA_MINUTE = 10.0
DEFAULT_CORRIDOR_BUFFER_M = 150.0
DEFAULT_MEETING_WINDOW_S = 30.0


def _edge_speeds_kph(G: nx.MultiDiGraph, path: list, corridor: bool = False) -> list[float]:
    speeds = []
    for u, v in zip(path[:-1], path[1:]):
        edata = G.get_edge_data(u, v)
        d = min(edata.values(), key=lambda x: x.get("length", float("inf")))
        length_m = float(d.get("length", 0) or 0)
        if corridor:
            t_s = float(
                d.get("amb_corridor_time_s", d.get("osm_time_s", length_m / (30 / 3.6)))
            )
        else:
            t_s = float(d.get("car_time_s", length_m / (30 / 3.6)))
        speeds.append(length_m * 3.6 / t_s if t_s > 0 else 0.0)
    return speeds


def _extract_street_names(G: nx.MultiDiGraph, path: list) -> list[str]:
    """Ordered list of unique street names traversed by a path."""
    names: list[str] = []
    for u, v in zip(path[:-1], path[1:]):
        edata = G.get_edge_data(u, v)
        d = min(edata.values(), key=lambda x: x.get("length", float("inf")))
        raw = d.get("name")
        if isinstance(raw, list):
            raw = raw[0] if raw else None
        if raw and (not names or names[-1] != raw):
            names.append(str(raw))
    return names


def position_at_time(
    G: nx.MultiDiGraph,
    path: list,
    times: list[float],
    t: float,
) -> tuple[float, float] | None:
    """Interpolate (lat, lng) along `path` at absolute time `t`.

    Returns None if `t` is before the path starts or after it ends.
    `times` are cumulative arrival seconds at each node (as returned by
    _cumulative_times), possibly shifted by a dispatch offset.
    """
    if not path or not times or len(path) != len(times):
        return None
    if t <= times[0]:
        return (G.nodes[path[0]]["y"], G.nodes[path[0]]["x"])
    if t >= times[-1]:
        return (G.nodes[path[-1]]["y"], G.nodes[path[-1]]["x"])
    for i in range(len(times) - 1):
        if times[i] <= t <= times[i + 1]:
            span = times[i + 1] - times[i]
            frac = 0.0 if span <= 0 else (t - times[i]) / span
            y0, x0 = G.nodes[path[i]]["y"], G.nodes[path[i]]["x"]
            y1, x1 = G.nodes[path[i + 1]]["y"], G.nodes[path[i + 1]]["x"]
            return (y0 + (y1 - y0) * frac, x0 + (x1 - x0) * frac)
    return None


def _cumulative_times(G: nx.MultiDiGraph, path: list, corridor: bool = False) -> list[float]:
    """Seconds elapsed at each node along the path, starting at t=0.

    Car: uses `car_time_s` per edge (observed speed at the previous intersection).
    Ambulance in green corridor: uses `amb_corridor_time_s` (OSM posted × speedup).
    """
    times = [0.0]
    for u, v in zip(path[:-1], path[1:]):
        edata = G.get_edge_data(u, v)
        d = min(edata.values(), key=lambda x: x.get("length", float("inf")))
        length_m = float(d.get("length", 0) or 0)
        if corridor:
            step = float(
                d.get("amb_corridor_time_s",
                      d.get("osm_time_s", length_m / (30 / 3.6)))
            )
        else:
            step = float(d.get("car_time_s", length_m / (30 / 3.6)))
        times.append(times[-1] + step)
    return times


def _path_edge_set(path: list) -> set:
    edges = set()
    for u, v in zip(path[:-1], path[1:]):
        edges.add((u, v))
        edges.add((v, u))
    return edges


def _blocked_edges_within_buffer(
    G: nx.MultiDiGraph,
    amb_path: list,
    buffer_m: float,
) -> set:
    """Return set of (u, v) edges whose midpoint lies within `buffer_m` metres
    of any node on the ambulance path. Blocks the corridor plus a buffer."""
    if buffer_m <= 0:
        return _path_edge_set(amb_path)

    amb_coords = np.array(
        [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in amb_path]
    )
    lat0 = float(amb_coords[:, 0].mean())
    m_per_deg_lat = 111_320.0
    m_per_deg_lng = 111_320.0 * math.cos(math.radians(lat0))
    proj = np.column_stack(
        [amb_coords[:, 0] * m_per_deg_lat, amb_coords[:, 1] * m_per_deg_lng]
    )
    tree = cKDTree(proj)

    blocked = set()
    for u, v in G.edges():
        mid_lat = (G.nodes[u]["y"] + G.nodes[v]["y"]) / 2
        mid_lng = (G.nodes[u]["x"] + G.nodes[v]["x"]) / 2
        d, _ = tree.query([mid_lat * m_per_deg_lat, mid_lng * m_per_deg_lng])
        if d <= buffer_m:
            blocked.add((u, v))
            blocked.add((v, u))
    return blocked


def compute_ambulance_reroute(
    G: nx.MultiDiGraph,
    car_start: tuple[float, float],
    car_end: tuple[float, float],
    amb_start: tuple[float, float],
    amb_end: tuple[float, float],
    equity_multiplier: float = 1.0,
    corridor_buffer_m: float = DEFAULT_CORRIDOR_BUFFER_M,
    meeting_window_s: float = DEFAULT_MEETING_WINDOW_S,
    ambulance_dispatch_offset_s: float = 0.0,
) -> dict:
    """Compute three artefacts:

    1. Ambulance route (fastest, using base OSM travel time — the green corridor
       clears local congestion so travel_time is un-multiplied).
    2. Car's original route (fastest, no ambulance in the picture).
    3. Car's rerouted path — same weight as the original, but every road
       segment on the ambulance corridor is penalised so the car takes a detour.

    Returns extra time / distance / tokens earned by the rerouted driver.
    """
    car_orig_node = ox.distance.nearest_nodes(G, X=car_start[1], Y=car_start[0])
    car_dest_node = ox.distance.nearest_nodes(G, X=car_end[1], Y=car_end[0])
    amb_orig_node = ox.distance.nearest_nodes(G, X=amb_start[1], Y=amb_start[0])
    amb_dest_node = ox.distance.nearest_nodes(G, X=amb_end[1], Y=amb_end[0])

    amb_path = nx.shortest_path(G, amb_orig_node, amb_dest_node, weight="w_fast")
    amb_coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in amb_path]
    amb_summary = _summarize_path(G, amb_path, corridor=True)
    amb_summary_traffic = _summarize_path(G, amb_path, corridor=False)

    car_orig_path = nx.shortest_path(G, car_orig_node, car_dest_node, weight="w_fast")
    car_orig_coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in car_orig_path]
    car_orig_summary = _summarize_path(G, car_orig_path)

    amb_edges = _path_edge_set(amb_path)
    blocked_edges = _blocked_edges_within_buffer(G, amb_path, corridor_buffer_m)

    car_orig_edges = list(zip(car_orig_path[:-1], car_orig_path[1:]))
    orig_shares_exact_edge = any((u, v) in amb_edges for u, v in car_orig_edges)
    orig_path_in_corridor = any((u, v) in blocked_edges for u, v in car_orig_edges)

    amb_times_local = _cumulative_times(G, amb_path, corridor=True)
    amb_times = [t + ambulance_dispatch_offset_s for t in amb_times_local]
    car_times = _cumulative_times(G, car_orig_path, corridor=False)
    amb_node_to_time = dict(zip(amb_path, amb_times))
    car_node_to_time = dict(zip(car_orig_path, car_times))
    shared_nodes = [n for n in car_orig_path if n in amb_node_to_time]

    conflicts = []
    for n in shared_nodes:
        t_amb_clock = amb_node_to_time[n]
        t_car_clock = car_node_to_time[n]
        dt = t_car_clock - t_amb_clock
        if abs(dt) < meeting_window_s:
            conflicts.append({
                "node": n,
                "coords": (G.nodes[n]["y"], G.nodes[n]["x"]),
                "t_amb_s": t_amb_clock,
                "t_car_s": t_car_clock,
                "dt_s": dt,
            })

    if orig_path_in_corridor:
        for u, v, k, d in G.edges(keys=True, data=True):
            base = float(d.get("w_fast", 0.0) or 0.0)
            d["w_reroute"] = base * REROUTE_PENALTY if (u, v) in blocked_edges else base
        try:
            car_new_path = nx.shortest_path(
                G, car_orig_node, car_dest_node, weight="w_reroute"
            )
            car_new_coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in car_new_path]
            car_new_summary = _summarize_path(G, car_new_path)
            rerouted = True
        except nx.NetworkXNoPath:
            car_new_path = car_orig_path
            car_new_coords = car_orig_coords
            car_new_summary = car_orig_summary
            rerouted = False
    else:
        car_new_path = car_orig_path
        car_new_coords = car_orig_coords
        car_new_summary = car_orig_summary
        rerouted = False

    extra_time_min = max(0.0, car_new_summary["time_min"] - car_orig_summary["time_min"])
    extra_distance_km = max(0.0, car_new_summary["distance_km"] - car_orig_summary["distance_km"])
    tokens_earned = (
        round(extra_time_min * TOKENS_PER_EXTRA_MINUTE * equity_multiplier)
        if rerouted and extra_time_min > 0.01
        else 0
    )

    car_new_times = (
        _cumulative_times(G, car_new_path, corridor=False) if rerouted else list(car_times)
    )

    amb_streets = _extract_street_names(G, amb_path)
    car_orig_streets = _extract_street_names(G, car_orig_path)
    car_new_streets = _extract_street_names(G, car_new_path) if rerouted else list(car_orig_streets)

    amb_edge_speeds = _edge_speeds_kph(G, amb_path, corridor=True)
    car_orig_edge_speeds = _edge_speeds_kph(G, car_orig_path, corridor=False)
    car_new_edge_speeds = (
        _edge_speeds_kph(G, car_new_path, corridor=False)
        if rerouted else list(car_orig_edge_speeds)
    )

    return {
        "ambulance": {
            "path": amb_path,
            "coords": amb_coords,
            "times": amb_times,
            "edge_speeds_kph": amb_edge_speeds,
            "streets": amb_streets,
            "summary": amb_summary,
            "summary_no_corridor": amb_summary_traffic,
        },
        "car_original": {
            "path": car_orig_path,
            "coords": car_orig_coords,
            "times": car_times,
            "edge_speeds_kph": car_orig_edge_speeds,
            "streets": car_orig_streets,
            "summary": car_orig_summary,
            "uses_amb_corridor": orig_shares_exact_edge,
            "crosses_corridor_buffer": orig_path_in_corridor,
        },
        "car_rerouted": {
            "path": car_new_path,
            "coords": car_new_coords,
            "times": car_new_times,
            "edge_speeds_kph": car_new_edge_speeds,
            "streets": car_new_streets,
            "summary": car_new_summary,
            "reroute_happened": rerouted,
        },
        "reroute_info": {
            "extra_time_min": extra_time_min,
            "extra_distance_km": extra_distance_km,
            "tokens_earned": tokens_earned,
            "equity_multiplier": equity_multiplier,
            "amb_time_saved_min": max(
                0.0,
                amb_summary_traffic["time_min"] - amb_summary["time_min"],
            ),
        },
        "conflicts": conflicts,
        "shared_nodes_count": len(shared_nodes),
        "meeting_window_s": meeting_window_s,
        "ambulance_dispatch_offset_s": ambulance_dispatch_offset_s,
    }


def compute_corridor_polygon(
    G: nx.MultiDiGraph,
    amb_path: list,
    buffer_m: float,
) -> list[tuple[float, float]]:
    """Return the buffer polygon around the ambulance path as a list of
    (lat, lng) pairs suitable for folium.Polygon.

    Approximation: 1 degree of latitude ≈ 111.32 km. This ignores latitude-
    dependent stretch on longitude, but for a small buffer (<2 km) at NYC's
    latitude the visual error is negligible.
    """
    if not amb_path:
        return []
    line = LineString([(G.nodes[n]["x"], G.nodes[n]["y"]) for n in amb_path])
    buffer_deg = buffer_m / 111_320.0
    poly = line.buffer(buffer_deg)
    if poly.is_empty:
        return []
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda p: p.area)
    return [(y, x) for x, y in poly.exterior.coords]


def simulate_corridor_traffic(
    G: nx.MultiDiGraph,
    amb_path: list,
    buffer_m: float,
    n_cars: int = 12,
    random_state: int = 42,
    equity_multiplier: float = 1.0,
) -> dict:
    """Sample random OD pairs and return only the cars whose fastest route
    crosses the ambulance corridor, along with their rerouted alternative.

    Cars are placed such that origin OR destination lies inside the corridor
    polygon — this virtually guarantees the fastest route crosses the corridor
    and needs a diversion.
    """
    if not amb_path:
        return {"cars": [], "n_nodes_in_corridor": 0}

    line = LineString([(G.nodes[n]["x"], G.nodes[n]["y"]) for n in amb_path])
    buffer_deg = buffer_m / 111_320.0
    corridor = line.buffer(buffer_deg)
    if corridor.is_empty:
        return {"cars": [], "n_nodes_in_corridor": 0}

    nodes_in_corridor = [
        n for n in G.nodes()
        if corridor.contains(Point(G.nodes[n]["x"], G.nodes[n]["y"]))
    ]
    if len(nodes_in_corridor) < 2:
        return {"cars": [], "n_nodes_in_corridor": len(nodes_in_corridor)}

    all_nodes = list(G.nodes())
    rng = np.random.default_rng(random_state)

    blocked_edges = _blocked_edges_within_buffer(G, amb_path, buffer_m)
    for u, v, k, d in G.edges(keys=True, data=True):
        base = float(d.get("w_fast", 0.0) or 0.0)
        d["w_reroute_ambient"] = (
            base * REROUTE_PENALTY if (u, v) in blocked_edges else base
        )

    cars = []
    attempts = 0
    max_attempts = max(n_cars * 8, 40)
    while len(cars) < n_cars and attempts < max_attempts:
        attempts += 1
        if rng.random() < 0.5:
            o = nodes_in_corridor[int(rng.integers(0, len(nodes_in_corridor)))]
            d = all_nodes[int(rng.integers(0, len(all_nodes)))]
        else:
            o = all_nodes[int(rng.integers(0, len(all_nodes)))]
            d = nodes_in_corridor[int(rng.integers(0, len(nodes_in_corridor)))]
        if o == d:
            continue
        try:
            orig_path = nx.shortest_path(G, o, d, weight="w_fast")
        except nx.NetworkXNoPath:
            continue
        crosses = any(
            (u, v) in blocked_edges
            for u, v in zip(orig_path[:-1], orig_path[1:])
        )
        if not crosses:
            continue
        try:
            new_path = nx.shortest_path(G, o, d, weight="w_reroute_ambient")
        except nx.NetworkXNoPath:
            new_path = orig_path

        orig_coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in orig_path]
        new_coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in new_path]
        orig_summary = _summarize_path(G, orig_path)
        new_summary = _summarize_path(G, new_path)
        extra_time_min = max(0.0, new_summary["time_min"] - orig_summary["time_min"])
        extra_distance_km = max(
            0.0, new_summary["distance_km"] - orig_summary["distance_km"]
        )
        tokens = round(
            extra_time_min * TOKENS_PER_EXTRA_MINUTE * equity_multiplier
        )
        cars.append({
            "orig_coords": orig_coords,
            "new_coords": new_coords,
            "orig_time_min": orig_summary["time_min"],
            "new_time_min": new_summary["time_min"],
            "orig_distance_km": orig_summary["distance_km"],
            "new_distance_km": new_summary["distance_km"],
            "extra_time_min": extra_time_min,
            "extra_distance_km": extra_distance_km,
            "tokens_earned": tokens,
        })

    return {
        "cars": cars,
        "n_nodes_in_corridor": len(nodes_in_corridor),
        "total_tokens": sum(c["tokens_earned"] for c in cars),
        "total_extra_time_min": sum(c["extra_time_min"] for c in cars),
        "total_extra_distance_km": sum(c["extra_distance_km"] for c in cars),
    }
