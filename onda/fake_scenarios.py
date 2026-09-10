"""Generate a plausible fake scenarios.json for UI prototyping.

Scenario A (baseline) uses REAL numbers from the dataset at weekday 09:00.
Scenarios B (efficiency-only villain) and C (ONDA fair) are FAKE but shaped
to match the story: B lowers total delay while making 4 of 6 zones worse;
C lowers total delay under a hard "no zone worse than A" constraint.

The schema matches the contract in PLAN.md so the real sim can drop in later.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
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
OUT_PATH = PROJECT_ROOT / "out" / "scenarios.json"

RUSH_HOUR = 9
RUSH_DAY_WEEKEND = 0
K_NEAREST = 4

BASELINE_CO2_KG_PER_VEH_HOUR = 8.0
BASELINE_TOKEN_VALUE_EUR = 0.05


def _haversine_km(lat1, lng1, lat2, lng2):
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlng / 2) ** 2
    )
    return 2 * r * math.asin(math.sqrt(a))


def _knn_edges(intersections: list[dict], k: int = K_NEAREST) -> list[dict]:
    edges = set()
    for i, a in enumerate(intersections):
        dists = []
        for j, b in enumerate(intersections):
            if i == j:
                continue
            dists.append((_haversine_km(a["lat"], a["lng"], b["lat"], b["lng"]), j))
        dists.sort()
        for _, j in dists[:k]:
            key = tuple(sorted((i, j)))
            edges.add(key)
    out = []
    for i, j in edges:
        km = _haversine_km(
            intersections[i]["lat"], intersections[i]["lng"],
            intersections[j]["lat"], intersections[j]["lng"],
        )
        out.append({"from": intersections[i]["id"], "to": intersections[j]["id"], "km": round(km, 3)})
    return out


def build() -> dict:
    df = pd.read_csv(DATA_PATH)
    snap = df[(df["hour"] == RUSH_HOUR) & (df["is_weekend"] == RUSH_DAY_WEEKEND)]
    per_int = (
        snap.groupby(["intersection_id", "city_zone", "latitude", "longitude"], as_index=False)
        .agg(
            wait_a=("average_wait_time", "mean"),
            vehicles=("vehicle_count", "mean"),
            lanes=("lanes", "mean"),
            green=("green_light_duration", "mean"),
            cycle=("signal_cycle_seconds", "mean"),
            emit=("emission_estimate", "mean"),
            cong=("congestion_score", "mean"),
        )
    )
    per_int["sat_x"] = per_int["vehicles"] / (
        per_int["lanes"] * 1800.0 * per_int["green"] / per_int["cycle"]
    )

    intersections = [
        {
            "id": r.intersection_id,
            "lat": float(r.latitude),
            "lng": float(r.longitude),
            "zone": r.city_zone,
        }
        for r in per_int.itertuples()
    ]
    edges = _knn_edges(intersections, K_NEAREST)

    per_zone_a = per_int.groupby("city_zone").agg(
        wait=("wait_a", "mean"),
        vehicles=("vehicles", "sum"),
    )
    zones = sorted(per_zone_a.index.tolist())

    rng = random.Random(42)

    # Scenario A — baseline from real data
    per_int_a = {
        r.intersection_id: {
            "wait": round(float(r.wait_a), 2),
            "x": round(float(min(r.sat_x, 1.5)), 3),
            "vehicles": round(float(r.vehicles), 1),
        }
        for r in per_int.itertuples()
    }
    per_zone_a_out = {
        z: {"wait": round(float(per_zone_a.loc[z, "wait"]), 2)}
        for z in zones
    }
    total_vh_a = float((per_int["vehicles"] * per_int["wait_a"]).sum() / 3600.0)
    total_co2_a = total_vh_a * BASELINE_CO2_KG_PER_VEH_HOUR / 1000.0

    metrics_a = {
        "vehicle_hours_lost": round(total_vh_a, 1),
        "co2_tons": round(total_co2_a, 2),
        "zones_worse": 0,
        "worst_zone_increase_s": 0.0,
        "gini": 0.063,
        "tokens_total": 0,
    }

    # Scenario B — efficiency-only villain: -0.9% total delay, 4 zones worse
    # Push wait onto peripheral zones, pull down centre zones
    centre = {"Downtown Core", "Financial District"}
    per_zone_b = {}
    delta_b = {}
    for z in zones:
        base = per_zone_a_out[z]["wait"]
        if z in centre:
            delta = -rng.uniform(1.5, 2.8)
        else:
            delta = rng.uniform(2.0, 3.5)
        per_zone_b[z] = {"wait": round(base + delta, 2)}
        delta_b[z] = round(delta, 2)
    zones_worse_b = sum(1 for z in zones if per_zone_b[z]["wait"] > per_zone_a_out[z]["wait"] + 0.05)
    worst_b = max(
        per_zone_b[z]["wait"] - per_zone_a_out[z]["wait"] for z in zones
    )
    metrics_b = {
        "vehicle_hours_lost": round(total_vh_a * 0.991, 1),
        "co2_tons": round(total_co2_a * 0.988, 2),
        "zones_worse": zones_worse_b,
        "worst_zone_increase_s": round(worst_b, 2),
        "gini": 0.039,
        "tokens_total": 0,
    }

    per_int_b = {}
    for r in per_int.itertuples():
        z = r.city_zone
        base = per_int_a[r.intersection_id]["wait"]
        adjusted = max(0.5, base + delta_b[z] + rng.uniform(-0.6, 0.6))
        per_int_b[r.intersection_id] = {
            "wait": round(adjusted, 2),
            "x": round(min(1.5, float(r.sat_x) * (1.0 + delta_b[z] / 20.0)), 3),
            "vehicles": round(float(r.vehicles) * (1.0 + delta_b[z] / 30.0), 1),
        }

    # Scenario C — ONDA: -2.1% total delay, δ=0 constraint holds
    per_zone_c = {}
    delta_c = {}
    for z in zones:
        base = per_zone_a_out[z]["wait"]
        delta = -rng.uniform(0.4, 2.5)
        per_zone_c[z] = {"wait": round(base + delta, 2)}
        delta_c[z] = round(delta, 2)
    zones_worse_c = 0
    worst_c = max(per_zone_c[z]["wait"] - per_zone_a_out[z]["wait"] for z in zones)
    metrics_c = {
        "vehicle_hours_lost": round(total_vh_a * 0.979, 1),
        "co2_tons": round(total_co2_a * 0.972, 2),
        "zones_worse": zones_worse_c,
        "worst_zone_increase_s": round(max(0.0, worst_c), 2),
        "gini": 0.055,
        "tokens_total": 1247,
    }

    per_int_c = {}
    for r in per_int.itertuples():
        z = r.city_zone
        base = per_int_a[r.intersection_id]["wait"]
        adjusted = max(0.5, base + delta_c[z] + rng.uniform(-0.4, 0.4))
        per_int_c[r.intersection_id] = {
            "wait": round(adjusted, 2),
            "x": round(min(1.5, float(r.sat_x) * (1.0 + delta_c[z] / 25.0)), 3),
            "vehicles": round(float(r.vehicles), 1),
        }

    # Rewards
    rewards_a = {"tokens_total": 0, "tokens_per_zone": {z: 0 for z in zones},
                 "co2_saved_kg": 0, "drivers_rewarded": 0}
    rewards_b = {"tokens_total": 0, "tokens_per_zone": {z: 0 for z in zones},
                 "co2_saved_kg": 0, "drivers_rewarded": 0}

    peripheral = [z for z in zones if z not in centre]
    tokens_per_zone_c = {}
    remaining = metrics_c["tokens_total"]
    for i, z in enumerate(peripheral):
        share = int(remaining / (len(peripheral) - i) * rng.uniform(0.85, 1.15))
        tokens_per_zone_c[z] = share
        remaining -= share
    for z in centre:
        tokens_per_zone_c[z] = int(remaining / len(centre))
    rewards_c = {
        "tokens_total": metrics_c["tokens_total"],
        "tokens_per_zone": tokens_per_zone_c,
        "co2_saved_kg": round((total_co2_a - metrics_c["co2_tons"]) * 1000.0, 1),
        "drivers_rewarded": 312,
    }

    # Emergency corridor
    hospital_row = per_int[per_int["sat_x"].between(0.5, 0.9)].sample(1, random_state=1).iloc[0]
    start_row = per_int.sample(1, random_state=7).iloc[0]
    # naive path: 3 intermediate intersections (fake)
    intermediate = per_int.sample(6, random_state=13)
    route_ids = [start_row.intersection_id] + intermediate.intersection_id.tolist() + [hospital_row.intersection_id]
    per_hop = []
    for iid in route_ids:
        base_wait = per_int_a[iid]["wait"]
        per_hop.append({
            "id": iid,
            "wait_before": round(base_wait * 4.2, 1),
            "wait_after": 2.0,
        })
    baseline_seconds = sum(h["wait_before"] for h in per_hop)
    corridor_seconds = sum(h["wait_after"] for h in per_hop)

    emergency = {
        "route": route_ids,
        "hospital": hospital_row.intersection_id,
        "baseline_seconds": round(baseline_seconds, 1),
        "corridor_seconds": round(corridor_seconds, 1),
        "per_hop": per_hop,
    }

    # Zones metadata (with vulnerability defaults)
    zones_meta = []
    for z in zones:
        intr_count = int((per_int["city_zone"] == z).sum())
        # peripheral zones get non-zero default vulnerability
        vuln_default = 0.0 if z in centre else 0.3
        zones_meta.append({
            "name": z,
            "intersections": intr_count,
            "vulnerability_default": vuln_default,
        })

    scenarios_json = {
        "meta": {
            "generated_at": pd.Timestamp.utcnow().isoformat(),
            "hour": RUSH_HOUR,
            "day_kind": "weekday",
            "source": "fake (real scenario A, synthetic B/C)",
            "alpha_default": 1.0,
        },
        "zones": zones_meta,
        "intersections": intersections,
        "edges": edges,
        "scenarios": {
            "A": {
                "label": "Today (baseline)",
                "metrics": metrics_a,
                "per_intersection": per_int_a,
                "per_zone": per_zone_a_out,
                "rewards": rewards_a,
            },
            "B": {
                "label": "Efficiency-only AI",
                "metrics": metrics_b,
                "per_intersection": per_int_b,
                "per_zone": per_zone_b,
                "rewards": rewards_b,
            },
            "C": {
                "label": "ONDA (fair)",
                "metrics": metrics_c,
                "per_intersection": per_int_c,
                "per_zone": per_zone_c,
                "rewards": rewards_c,
            },
        },
        "emergency": emergency,
    }
    return scenarios_json


def main():
    out = build()
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT_PATH}")
    for k in ("A", "B", "C"):
        m = out["scenarios"][k]["metrics"]
        print(
            f"  {k} — vh={m['vehicle_hours_lost']:.1f}  "
            f"zones_worse={m['zones_worse']}  worst_zone_inc={m['worst_zone_increase_s']:.2f}s  "
            f"co2={m['co2_tons']:.2f}t  tokens={m['tokens_total']}"
        )


if __name__ == "__main__":
    main()
