"""Rendering helpers for the ONDA Scenarios tab.

Loads scenarios.json (the contract from PLAN.md) and provides Streamlit
render functions for: KPI row, per-zone bar chart, 3D map, the
efficiency-vs-worst-zone chart, rewards panel, and the vulnerability table.
"""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import pydeck as pdk
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCENARIOS_PATH = PROJECT_ROOT / "out" / "scenarios.json"


@st.cache_data
def load_scenarios(path: Path = SCENARIOS_PATH) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _delta_color(delta: float, threshold: float = 0.05) -> str:
    if delta > threshold:
        return "🔴"
    if delta < -threshold:
        return "🟢"
    return "⚪"


def render_kpi_row(scenarios: dict, current: str, baseline: str = "A") -> None:
    m = scenarios["scenarios"][current]["metrics"]
    m_base = scenarios["scenarios"][baseline]["metrics"]

    def d(k):
        return m[k] - m_base[k]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        "Vehicle-hours lost / day",
        f"{m['vehicle_hours_lost']:.0f}",
        delta=f"{d('vehicle_hours_lost'):+.0f}" if current != baseline else None,
        delta_color="inverse",
    )
    c2.metric(
        "CO₂ / day (tons)",
        f"{m['co2_tons']:.2f}",
        delta=f"{d('co2_tons'):+.2f}" if current != baseline else None,
        delta_color="inverse",
    )
    c3.metric(
        "Zones worse than today",
        int(m["zones_worse"]),
        delta=None,
        help="Number of city zones with higher mean wait than baseline (A). ONDA guarantees 0.",
    )
    c4.metric(
        "Worst zone increase (s)",
        f"{m['worst_zone_increase_s']:+.1f}",
        delta=None,
    )
    c5.metric(
        "Tokens distributed",
        int(m["tokens_total"]),
        delta=None,
        help="Rewards paid to drivers who accepted a diversion or off-peak shift.",
    )


def render_per_zone_chart(scenarios: dict, current: str, baseline: str = "A") -> None:
    zones_a = scenarios["scenarios"][baseline]["per_zone"]
    zones_x = scenarios["scenarios"][current]["per_zone"]

    rows = []
    for z, va in zones_a.items():
        vb = zones_x[z]["wait"]
        rows.append({"zone": z, "scenario": baseline, "wait": va["wait"]})
        rows.append({"zone": z, "scenario": current, "wait": vb})
    df = pd.DataFrame(rows)

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("zone:N", sort="-y", title=None),
            y=alt.Y("wait:Q", title="Mean wait (s)"),
            color=alt.Color(
                "scenario:N",
                scale=alt.Scale(
                    domain=[baseline, current],
                    range=["#7f8c8d", "#e74c3c" if current == "B" else "#27ae60"],
                ),
                title=None,
            ),
            xOffset="scenario:N",
            tooltip=["zone", "scenario", alt.Tooltip("wait", format=".2f")],
        )
        .properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)


def render_efficiency_chart(scenarios: dict) -> None:
    """The money chart: x = vehicle-hours saved vs A, y = worst-zone increase (s).

    A sits at origin (0,0). B goes right (saves time) but up (pays a fairness cost).
    C goes right and stays at y=0. Nobody else will have this chart.
    """
    a = scenarios["scenarios"]["A"]["metrics"]
    rows = []
    color_map = {"A": "#7f8c8d", "B": "#e74c3c", "C": "#27ae60"}
    for k in ("A", "B", "C"):
        m = scenarios["scenarios"][k]["metrics"]
        rows.append({
            "scenario": k,
            "label": f"{k} — {scenarios['scenarios'][k]['label']}",
            "vh_saved": a["vehicle_hours_lost"] - m["vehicle_hours_lost"],
            "worst_zone_increase": m["worst_zone_increase_s"],
        })
    df = pd.DataFrame(rows)

    base = alt.Chart(df).encode(
        x=alt.X("vh_saved:Q", title="Vehicle-hours saved vs today (higher = more efficient)"),
        y=alt.Y("worst_zone_increase:Q", title="Worst zone extra wait (s)  ← lower is fairer"),
    )
    points = base.mark_circle(size=400, opacity=0.85).encode(
        color=alt.Color(
            "scenario:N",
            scale=alt.Scale(domain=list(color_map.keys()), range=list(color_map.values())),
            legend=None,
        ),
        tooltip=["label", alt.Tooltip("vh_saved", format=".0f"),
                 alt.Tooltip("worst_zone_increase", format=".2f")],
    )
    labels = base.mark_text(dy=-20, fontSize=14, fontWeight="bold").encode(text="scenario:N")
    zero_x = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(strokeDash=[4, 3], color="#666").encode(y="y:Q")

    st.altair_chart((zero_x + points + labels).properties(height=340), use_container_width=True)
    st.caption(
        "**A** sits at the origin (today's baseline).  "
        "**B** goes right (efficiency gain) but *up* — some zone pays for it.  "
        "**C** goes right and stays at **y = 0** — no zone is worse off."
    )


def render_map(scenarios: dict, current: str, baseline: str = "A") -> None:
    ints = scenarios["intersections"]
    per_int_now = scenarios["scenarios"][current]["per_intersection"]
    per_int_base = scenarios["scenarios"][baseline]["per_intersection"]

    rows = []
    for it in ints:
        w_now = per_int_now[it["id"]]["wait"]
        w_base = per_int_base[it["id"]]["wait"]
        delta = w_now - w_base
        rows.append({
            "id": it["id"],
            "zone": it["zone"],
            "latitude": it["lat"],
            "longitude": it["lng"],
            "wait": w_now,
            "wait_base": w_base,
            "delta": delta,
        })
    df = pd.DataFrame(rows)

    max_delta = max(1.0, df["delta"].abs().max())
    df["r"] = (
        (df["delta"].clip(lower=0) / max_delta * 220 + (df["delta"] < 0) * 30).astype(int)
    )
    df["g"] = (
        ((-df["delta"]).clip(lower=0) / max_delta * 220 + (df["delta"] > 0) * 30).astype(int)
    )
    df["b"] = 60
    df["a"] = 210

    layer = pdk.Layer(
        "ColumnLayer",
        data=df,
        get_position=["longitude", "latitude"],
        get_elevation="wait",
        elevation_scale=25,
        radius=60,
        get_fill_color=["r", "g", "b", "a"],
        pickable=True,
        auto_highlight=True,
    )
    view_state = pdk.ViewState(
        latitude=float(df["latitude"].mean()),
        longitude=float(df["longitude"].mean()),
        zoom=11,
        pitch=45,
        bearing=15,
    )
    tooltip = {
        "html": (
            "<b>{id}</b><br/>"
            "Zone: {zone}<br/>"
            "Wait (this scenario): {wait} s<br/>"
            "Wait (baseline): {wait_base} s<br/>"
            "Δ vs baseline: {delta} s"
        ),
    }
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style="road",
        tooltip=tooltip,
    )
    st.pydeck_chart(deck, use_container_width=True, height=520)
    st.caption("Column height = mean wait (s). Green = better than baseline, red = worse.")


def render_rewards_panel(scenarios: dict, current: str) -> None:
    r = scenarios["scenarios"][current]["rewards"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Tokens distributed", int(r["tokens_total"]))
    c2.metric("Drivers rewarded", int(r["drivers_rewarded"]))
    c3.metric("CO₂ saved (kg)", f"{r['co2_saved_kg']:.0f}")

    if r["tokens_total"] == 0:
        st.info(
            "No tokens paid in this scenario.  "
            "Whoever makes room for others must be compensated — otherwise the "
            "periphery absorbs the cost of the optimization for free."
        )
        return

    tokens_df = pd.DataFrame(
        [{"zone": z, "tokens": t} for z, t in r["tokens_per_zone"].items()]
    ).sort_values("tokens", ascending=False)
    st.dataframe(tokens_df, use_container_width=True, hide_index=True)
    st.caption(
        f"Redeemable **only** for public transport passes and parking discounts. "
        f"1 token = 1 minute of wait saved to other drivers, multiplied by "
        f"(1 + α × vulnerability(zone))."
    )


def render_vulnerability_editor(scenarios: dict) -> tuple[float, pd.DataFrame]:
    """Return (alpha, vulnerability_table) for downstream use.

    In the current UI the table is display-only — the real sim will read it back.
    """
    st.caption(
        "Policy inputs held by the city. **α** trades city-wide efficiency for zone "
        "equity. **vulnerability** is the per-zone weight (0 = no protection, 1 = maximum)."
    )
    alpha = st.slider("α (equity weight)", 0.0, 3.0, scenarios["meta"]["alpha_default"], 0.1)

    zones = scenarios["zones"]
    default_df = pd.DataFrame(
        [{"zone": z["name"], "intersections": z["intersections"],
          "vulnerability": z["vulnerability_default"]}
         for z in zones]
    )
    edited = st.data_editor(
        default_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "vulnerability": st.column_config.NumberColumn(
                "vulnerability",
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                format="%.2f",
            ),
            "intersections": st.column_config.NumberColumn(disabled=True),
            "zone": st.column_config.TextColumn(disabled=True),
        },
        key="vulnerability_editor",
    )
    return alpha, edited


def render_emergency_summary(scenarios: dict) -> None:
    e = scenarios.get("emergency")
    if not e:
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Route hops", len(e["route"]))
    c2.metric("Baseline signal delay", f"{e['baseline_seconds']:.0f} s")
    c3.metric(
        "With green corridor",
        f"{e['corridor_seconds']:.0f} s",
        delta=f"−{e['baseline_seconds'] - e['corridor_seconds']:.0f} s",
        delta_color="inverse",
    )
    hop_df = pd.DataFrame(e["per_hop"])
    hop_df["saved_s"] = hop_df["wait_before"] - hop_df["wait_after"]
    st.dataframe(hop_df, use_container_width=True, hide_index=True)


def scenario_summary_text(scenarios: dict) -> str:
    """Compact text summary for feeding into the LLM."""
    lines = []
    lines.append(f"Snapshot: hour {scenarios['meta']['hour']:02d}:00 {scenarios['meta']['day_kind']}.")
    lines.append(f"Zones: {', '.join(z['name'] for z in scenarios['zones'])}.")
    lines.append("")
    for k in ("A", "B", "C"):
        s = scenarios["scenarios"][k]
        m = s["metrics"]
        lines.append(f"[{k}] {s['label']}")
        lines.append(
            f"  vehicle_hours_lost={m['vehicle_hours_lost']}  "
            f"co2_tons={m['co2_tons']}  "
            f"zones_worse={m['zones_worse']}  "
            f"worst_zone_increase_s={m['worst_zone_increase_s']}  "
            f"tokens={m['tokens_total']}"
        )
        for z, v in s["per_zone"].items():
            base = scenarios["scenarios"]["A"]["per_zone"][z]["wait"]
            lines.append(f"    zone={z} wait={v['wait']:.2f}s (Δ vs A: {v['wait'] - base:+.2f}s)")
    e = scenarios.get("emergency")
    if e:
        lines.append("")
        lines.append(
            f"Emergency corridor: {len(e['route'])} hops, "
            f"signal delay {e['baseline_seconds']:.0f}s → {e['corridor_seconds']:.0f}s."
        )
    return "\n".join(lines)
