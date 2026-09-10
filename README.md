# Hackaton_IEEE_SWC — CityFlow 3D Traffic Explorer + ONDA Fairness Dashboard

An interactive Streamlit app built for the **IEEE Smart World Congress hackathon**. It answers a single, uncomfortable question that the organisers put to the challenge:

> *Can we build smart urban mobility services without making them unfair?*

Most attempts to make traffic "smarter" quietly re-route the overflow onto neighbourhoods that had nothing to do with the problem, or move ambulances faster by making everyone else wait. **ONDA** (this project) tries to measure that hidden cost and refuse to pay it.

Concretely, the app does three things on the same 100 NYC intersections from the Kaggle **CityFlow: Smart Urban Mobility & Traffic IoT** dataset:

1. **Explores the raw data in 3D** so a human can see where and when things go bad.
2. **Plans routes** for a private car — and, in ambulance mode, shows how *other* drivers must step aside when an emergency crosses their path, and how they get compensated for it.
3. **Compares three signal-timing scenarios** side-by-side (today, a "greedy efficiency" villain, and ONDA's fair alternative) so a jury can see the trade-off between total delay and per-zone fairness.

This README is the entire tour. If you have never seen the app before, read from top to bottom; if you've seen it before and want a specific detail, use the table of contents.

---

## Table of contents

1. [The problem in one paragraph](#the-problem-in-one-paragraph)
2. [What the app looks like](#what-the-app-looks-like)
3. [Setup — from a fresh clone to a running app](#setup--from-a-fresh-clone-to-a-running-app)
4. [Tab 1 — 🗺️ Explorer 3D](#tab-1--️-explorer-3d)
5. [Tab 2 — 🚗 Route Planner](#tab-2--route-planner)
   - [Car-only mode](#car-only-mode)
   - [Car + ambulance mode](#car--ambulance-mode)
   - [How travel time is computed (data model)](#how-travel-time-is-computed-data-model)
   - [How the reroute is actually computed](#how-the-reroute-is-actually-computed)
   - [Why it works (and why it feels "magic")](#why-it-works-and-why-it-feels-magic)
   - [The trigger rule (spatial, not temporal)](#the-trigger-rule-spatial-not-temporal)
   - [How ambient traffic is generated](#how-ambient-traffic-is-generated)
   - [Tokens — how much the driver earns](#tokens--how-much-the-driver-earns)
   - [Concrete example (from the "Example" button)](#concrete-example-from-the-example-button)
6. [Tab 3 — 🌊 ONDA Scenarios](#tab-3--onda-scenarios)
7. [Local LLM (Ollama) — optional but nice](#local-llm-ollama--optional-but-nice)
8. [The dataset in detail](#the-dataset-in-detail)
9. [Project layout](#project-layout)
10. [Key files, function by function](#key-files-function-by-function)
11. [Suggested demo script for the pitch](#suggested-demo-script-for-the-pitch)
12. [Troubleshooting](#troubleshooting)
13. [Roadmap / open work](#roadmap--open-work)
14. [Credits](#credits)

---

## The problem in one paragraph

Smart traffic AI reduces the city's total delay by quietly *taxing* the neighbourhoods with the least mobility power. Any routing algorithm that only optimises the aggregate — average commute time, total vehicle-hours, city-wide CO₂ — will happily push overflow traffic into peripheral zones that had no say in the decision. The same applies to green-corridor systems for emergency vehicles: they can save a few minutes for the ambulance while imposing dozens of minutes of collective wait on drivers who happen to be in the wrong place at the wrong time. **ONDA measures that tax explicitly and refuses to pay it** — via a hard "no zone gets worse than today" constraint in the ONDA scenario, and via a token-based reward for every driver who accepts a diversion in the Route Planner. All numbers are grounded in a real-world dataset, not in synthetic gridworlds.

---

## What the app looks like

Three tabs at the top of a single Streamlit page:

| Tab | Purpose |
|---|---|
| **🗺️ Explorer 3D** | Understand where and when the city is stuck. |
| **🚗 Route Planner** | Simulate a specific trip, and in ambulance mode watch other drivers being asked to step aside — and rewarded for it. |
| **🌊 ONDA Scenarios** | See the trade-off between efficiency and fairness for the *system* as a whole, at rush hour. |

The Streamlit sidebar hosts filters for the Explorer tab (city zone, hour range, day of week, metric, layer style, temporal aggregation, basemap, ranking controls). The other two tabs have their own controls inside the tab body.

---

## Setup — from a fresh clone to a running app

The project targets **Python 3.11**. If you have a different Python major version, install 3.11 first (`brew install python@3.11`, `pyenv install 3.11`, or your package manager of choice).

```bash
# 1. Create and activate the virtual environment
python3 -m venv .venv
source .venv/bin/activate         # macOS / Linux
# on Windows:  .venv\Scripts\activate

# 2. Install all Python dependencies (pinned in requirements.txt)
pip install -r requirements.txt

# 3. Download the CityFlow dataset (~50 MB) into ./data
python import.py
```

The `import.py` script wraps [kagglehub](https://pypi.org/project/kagglehub/) but redirects its cache to the local `./data/` folder so the whole team works from the same on-disk layout. You'll be prompted for Kaggle credentials the first time; follow the on-screen link if needed.

```bash
# 4. Run the app
streamlit run app.py
```

The app opens at http://localhost:8501. The first time you open the **Route Planner** tab it will download the OpenStreetMap drive network for New York City (~30–60 s, cached to `cache/nyc_drive.graphml`). Subsequent runs load the cached graph in under a second.

### Optional but recommended: local LLM

Two panels ("Why is this reroute needed?" and "Explain this scenario") stream answers from a local [Ollama](https://ollama.com/) model. If you want them to work:

```bash
# https://ollama.com/download
ollama pull qwen3:1.7b   # ~1 GB, fast; alternatives: qwen2.5:7b, llama3.1:8b
ollama serve             # in a separate terminal, if not already a service
```

The app degrades gracefully: if Ollama is not reachable, the LLM panels show a friendly hint and everything else keeps working. Do **not** pull the model on the hackathon venue's wifi — pre-download at home.

---

## Tab 1 — 🗺️ Explorer 3D

### Goal

Look at the raw dataset in a way that makes patterns obvious. In particular: which intersections are the worst, when they are the worst, and by which metric.

### Controls (in the sidebar)

- **City zone** — multi-select. NYC is split by the dataset into six zones (Downtown Core, Financial District, Suburban North, Tech Park, Residential West, Industrial East). Deselect a zone to hide its intersections and re-compute all downstream stats.
- **Hour range** — restrict to a slice of the 24-hour clock (e.g., 8-9 for the morning peak).
- **Day of week** — multi-select from Mon-Sun (defaults to all).
- **Metric (column height)** — the field to visualise on the 3D map. Choices: `vehicle_count`, `congestion_score`, `average_wait_time`, `queue_length`, `emission_estimate`, `air_quality_index`.
- **3D layer type** — how the metric is rendered on the map:
  - *Columns per intersection* — one 3D pillar at each of the 100 intersections, height ∝ metric.
  - *Precise scatter* — one dot per intersection at exact coordinates, radius ∝ metric (useful when you want to see coordinates without vertical clutter).
  - *Aggregated hexagon* — bins many intersections into hex cells for wide-area overview (deck.gl `HexagonLayer`).
- **Time aggregation per intersection** — how the metric across the filtered hours is collapsed into a single value per intersection: Mean / Max / Sum / Last value.
- **Basemap** — background tiles. `Road (Carto)` is the default because it strikes a nice balance between visible streets and low ink. `OpenStreetMap` uses osm.org tiles directly (heaviest but most detailed). `Satellite` is aerial. `Light` and `Dark` are minimalist.

### Ranking panel (in the sidebar)

- **Highlight worst/best on the map** — checkbox.
- **How many to highlight (per side)** — slider 3-20.

When enabled, the map draws a thick red ring around the top-N *worst* intersections and a thick green ring around the top-N *best*, ranked by a composite **badness score**.

**Badness score** (per intersection, over the filtered slice) is:

```
badness = mean of {
    congestion_score_norm,       # 0..1 min-max
    emission_estimate_norm,      # 0..1 min-max
    fuel_waste_estimate_norm,    # 0..1 min-max
    air_quality_index_norm,      # 0..1 min-max (higher AQI = worse)
    average_wait_time_norm       # 0..1 min-max
}
```

Higher = worse. Below the map two ranked tables show *Worst N* (Reds gradient) and *Best N* (Greens gradient) with the underlying metrics side by side. Column `badness` is the composite; the other columns are the raw values that fed into it.

### Reading the map

- Column height ∝ your chosen metric.
- Column colour: red end of the spectrum for high values, green for low.
- If ranking is on, thick coloured rings mark the extremes.
- Hover a column to see the intersection ID, zone, and current value.
- Drag with the **right mouse button** to tilt/rotate (Google Maps-style 3D).

Below the map you also get four "top-line" numbers: total records in the filter, unique intersections, average congestion score, average vehicle speed.

---

## Tab 2 — 🚗 Route Planner

The router is the interactive heart of the app. It answers two very different questions depending on the mode:

- **Car-only mode** — "What's my best route from A to B, given today's observed traffic?"
- **Car + ambulance mode** — "An ambulance has been dispatched and its corridor overlaps with my planned route. What happens?"

### Car-only mode

**Step 1.** Set two points on the map (car start and car end). The default click-to-place mechanism uses a small radio selector — *"the next map click will update"* — that auto-advances to the next empty slot, so you don't have to click Reset every time.

**Step 2.** Adjust the departure hour and day type at the top of the tab. These drive which slice of the dataset is used to weight the graph.

**Step 3.** Read the results:

- The map shows three coloured routes:
  - 🟦 **Fastest** — minimises observed-speed travel time (see the data model below).
  - 🟧 **Balanced** — weighted blend: 50 % time + 25 % congestion + 25 % eco cost.
  - 🟩 **Eco** — minimises total (fuel + CO₂).
- A comparison table underneath reports: distance, time, average speed, fuel (L), CO₂ (kg), average congestion (0-1) for each route.

Why are the three routes often very close on a Manhattan grid? Because the parallel avenues have similar observed speeds, and the FDR Drive / West Side Highway pair provides two nearly-equivalent express corridors. You'll see more divergence in off-peak hours where highways win on speed.

### Car + ambulance mode

Toggle "🚗 + 🚑 Car with ambulance" at the top. Now you place **four** points (car start, car end, ambulance start, ambulance end). The slot selector still auto-advances but you can pick any slot and click on the map to move only that point.

Three new sliders appear:

- **🚑 Protected corridor width (m)** — 100 to 2000, default 1200. A buffer polygon is drawn around the ambulance path with this radius. Any car whose route enters the polygon is considered inside the "corridor" and must step aside.
- **⏱️ Ambulance dispatched after (s)** — 0 to 1800, default 700. How many seconds after the car left the ambulance is called. This lets you explore "if the ambulance is dispatched 5 minutes after the driver started, would they meet?"
- **🎯 Conflict window (s)** — 10 to 180, default 60. Tolerance for the space-time conflict check. If car and ambulance would arrive at the same intersection within ±this many seconds of each other, that intersection is flagged as a *conflict*.

There's also an **ambient traffic simulation** toggle (default on) with a slider for the number of sampled cars (default 15). More on that in a moment.

#### What you see on the map

- 🔴 A translucent **corridor bubble** — the buffer polygon around the ambulance path. The bubble expands/contracts live as you drag the width slider.
- 🚑 **Ambulance route** — thick red line.
- 🚗 **Car — rerouted** — solid blue.
- ⚪ **Car — original** — dashed grey (what the car *would* have taken without the ambulance).
- 🟡 **Yellow dots** at intersections where car and ambulance would have been at the same time (space-time conflicts).
- 🟢 **Green paths** — ambient traffic that was rerouted around the corridor.
- ⚪ **Grey dashed** ambient paths — the original routes those ambient cars would have taken.

Markers:

- 🟢 flag start / dark-green flag end for the car
- 🔴 plus start / dark-red heart for the ambulance destination (hospital)
- The marker of the currently selected slot has a yellow icon; the others are white.

#### The reroute rule (important)

The car reroutes **whenever its original route enters the protected corridor**, regardless of whether an actual space-time conflict occurs. This is a *precautionary* rule: if an ambulance is coming through, all cars in its corridor step aside, even the ones that would have missed it in time.

The space-time conflict count is still displayed and used for context — "how many intersections would you have physically shared with the ambulance?" — but it does not gate the reroute. So you can slide the corridor width alone and watch the reroute switch on and off in real time.

#### Ambient traffic simulation

Toggle it on (it's on by default) and the sampler will:

1. Compute the corridor buffer polygon.
2. Find every OSM node inside the polygon (typically ~800–1000 on a 1.2 km buffer in Manhattan).
3. Sample **N** random OD pairs (default 15) where either the origin or the destination is one of those nodes.
4. Compute each car's fastest route.
5. Keep only the cars whose fastest route physically crosses the corridor buffer.
6. Recompute the fastest route with corridor edges heavily penalised (a "detour").
7. Report per-car: original time, rerouted time, Δ time, Δ distance, tokens earned.

The panel below the map summarises: cars rerouted, intersections in the corridor, total extra time in minutes, total tokens paid. A per-car table lists every simulated driver's compensation.

**Why this matters for the pitch.** The main "hero" car you clicked is one anecdote. The ambient sim shows that *dozens* of anonymous drivers are affected by any given ambulance corridor. Multiplied by the number of ambulance calls per day, the collective cost is what makes the token compensation a real feature and not a gimmick.

#### Space-time animation

Scroll below the ambient panel to the animation section. Two maps sit side by side, both driven by the same time slider (and a `▶ Play` button that walks the slider forward):

- **✅ ACTUAL — car reroutes.** You see the ambulance moving along its corridor and the blue car following its detour. They never occupy the same point.
- **⚠️ HYPOTHETICAL — car ignores reroute.** Same ambulance, but the car keeps its original path. When the two dots get within 150 m of each other, both markers turn yellow and the metric row flashes **⚠️ COLLISION**.

Live metrics under each map: current speed of each vehicle (kph) and their instantaneous distance (m).

This is the visualisation that answers the "would they actually meet?" question in the most obvious way.

#### Explain this reroute (local LLM)

Below the animation, a button asks a local [Ollama](https://ollama.com/) model (default `qwen3:1.7b`) to write a 4-5 sentence explanation in English. The prompt is **grounded**: the model receives the exact numbers of your scenario — distance, time, speed, conflicts, tokens — and the ordered list of OSM street names for both routes. It answers:

1. Why the reroute was (or was not) needed, referring to the conflict count and dispatch timing.
2. The exact turn-by-turn street list the driver should now follow (quoting real street names).
3. How many tokens the driver earned, with the formula `round(extra_time_min × 10 × equity_multiplier)`.
4. What would change if the ambulance were dispatched earlier or later.

A dedicated system prompt (`REROUTE_SYSTEM_PROMPT` in `onda/llm.py`) forbids the model from inventing multipliers or units and enforces the token formula.

### How travel time is computed (data model)

This is the section your reviewer will care about. The routing does not use OpenStreetMap posted speed limits directly. Instead:

1. The OSM drive network for the NYC bounding box is downloaded once and cached (~30 000 edges, ~13 000 nodes).
2. For every **edge** `(u → v)`, we look up the **CityFlow intersection** closest to the origin node `u` and copy its `average_speed_kph` (observed at the currently selected hour + day type) onto the edge.
3. Travel time for the edge is then `length_m × 3.6 / observed_speed_kph`. Congestion is therefore already inside the number — a segment leaving a rush-hour Downtown intersection gets ~20 kph, while the same segment at 3 am gets ~55 kph. No extra multiplier is added.
4. The **ambulance travels at the same observed speed as the car** on each edge. Its "priority corridor" is spatial only (other cars step aside), not a speed boost. This is the honest choice: our dataset does not measure emergency-vehicle speeds, so pretending the ambulance goes 60 % faster would be theatre.
5. `Fastest` weight = `car_time_s`. `Eco` weight = `length_m × BASELINE_FUEL_L_PER_KM × (1 + 3 × fuel_norm) + length_m × BASELINE_CO2_KG_PER_KM × (1 + 3 × emit_norm)`. `Balanced` = weighted sum.

Fuel and CO₂ are estimated from a baseline vehicle profile (0.08 L/km, 190 g CO₂/km) modulated by the local emission / fuel-waste intensity (min-max normalised across the dataset). The multipliers `(1 + 3 × norm)` give a fuel value that ranges from 100 % to 400 % of the baseline depending on how bad the local corridor is.

### How the reroute is actually computed

The reroute is a **weighted shortest-path** trick. Four steps happen inside [`router.py:compute_ambulance_reroute()`](router.py) every time the four points are placed.

#### 1. Snap the four clicks to the OSM graph

```python
car_orig_node = ox.distance.nearest_nodes(G, X=car_start[1], Y=car_start[0])
car_dest_node = ox.distance.nearest_nodes(G, X=car_end[1],   Y=car_end[0])
amb_orig_node = ox.distance.nearest_nodes(G, X=amb_start[1], Y=amb_start[0])
amb_dest_node = ox.distance.nearest_nodes(G, X=amb_end[1],   Y=amb_end[0])
```

Your click at (40.7075, −74.0113) does not have to be exactly on an intersection — `osmnx` finds the nearest real graph node.

#### 2. Compute both routes independently, as if the ambulance did not exist

```python
amb_path      = nx.shortest_path(G, amb_orig_node, amb_dest_node, weight="w_fast")
car_orig_path = nx.shortest_path(G, car_orig_node, car_dest_node, weight="w_fast")
```

`w_fast` is `car_time_s` = `length_m × 3.6 / observed_speed_kph`, i.e. the observed-speed travel time on each edge (see the previous section).

#### 3. Build the "corridor" — the set of edges that must be cleared

```python
blocked_edges = _blocked_edges_within_buffer(G, amb_path, corridor_buffer_m)
```

This is where the `🚑 Protected corridor width (m)` slider actually applies. The function walks every edge in the graph and asks: *is the midpoint of this edge within `corridor_buffer_m` metres of any node on the ambulance path?* If yes, the edge is added to `blocked_edges`. Distances are computed by projecting lat/lng to metres via a local flat-earth approximation, then queried through a `scipy.spatial.cKDTree` — fast on 30 000 edges.

At buffer = 1200 m in the NYC example this gives ~2 000 blocked edges — a "no-go bubble" wrapped around the ambulance's actual path. The exact same polygon is what `folium.Polygon` draws on the map, via `compute_corridor_polygon()` (which uses `shapely.geometry.LineString.buffer` on the same path with the same buffer).

#### 4. Check the trigger, then re-route

```python
orig_path_in_corridor = any(
    (u, v) in blocked_edges
    for u, v in zip(car_orig_path[:-1], car_orig_path[1:])
)

if orig_path_in_corridor:
    # Multiply the cost of every corridor edge by REROUTE_PENALTY (=1000).
    # Everything else keeps its normal w_fast cost.
    for u, v, k, d in G.edges(keys=True, data=True):
        base = float(d.get("w_fast", 0.0))
        d["w_reroute"] = base * REROUTE_PENALTY if (u, v) in blocked_edges else base

    # Re-run Dijkstra with the new weights.
    car_new_path = nx.shortest_path(G, car_orig_node, car_dest_node, weight="w_reroute")
```

The 1000× penalty makes going through any corridor edge 1000 times more expensive than that same edge's normal cost. Any alternative route that stays outside the corridor is therefore cheaper, so `nx.shortest_path` picks it. The corridor edges are technically still traversable (if there is truly no alternative — say, the destination is inside the bubble — the path still uses them), but only as a last-resort fallback.

### Why it works (and why it feels "magic")

`networkx.shortest_path` runs Dijkstra on the modified weight. Dijkstra returns the **globally optimal** path given whatever cost function you hand it. So the reroute is not a hand-crafted "turn left here" heuristic — it's the fastest possible detour on the real NYC street network *given the constraint that corridor edges are prohibitively expensive*.

That's why on Manhattan a buffer of 1200 m often forces the car onto a different avenue *pair*: FDR Drive gets blocked, so the algorithm switches to the West Side Highway. The car goes from Broadway → FDR to Broadway → Battery Place → West Street → 12th Avenue → West Side Highway — a physically longer route that Dijkstra picks because every FDR-side edge is now 1000× its real cost.

### The trigger rule (spatial, not temporal)

The reroute fires **whenever the car's original path crosses the corridor buffer**, i.e. `orig_path_in_corridor == True`. This is a **spatial** rule, not a temporal one:

- If the car and ambulance would be at the same intersection at the same time (a *space-time conflict*), the reroute is obviously needed.
- If the ambulance is dispatched much earlier or much later than the car, they never meet in time — but the corridor still needs to be clear when the ambulance moves through it. The car reroutes anyway, as a precaution.

Sliding just the corridor-width slider (leaving dispatch and window untouched) is enough to switch the reroute on or off. Space-time conflicts still get computed and shown as extra context ("your reroute avoided N actual meeting points"), but they no longer gate anything.

Previously the trigger was `if conflicts:`, which required temporal overlap. That produced the confusing behaviour of "🛰️ Intersections in corridor is 961 but the car isn't rerouted" for scenarios where the ambulance had already passed by the time the car arrived. The current spatial rule matches the intuitive mental model — "if you're in the bubble, you move" — and is what real emergency-corridor systems do in practice.

### How ambient traffic is generated

`simulate_corridor_traffic()` is the same trick applied N times:

1. Compute the corridor polygon (via `shapely.geometry.LineString.buffer`).
2. Find every OSM node that falls inside the polygon (~800–1000 for a 1.2 km buffer in Manhattan). This is the `n_nodes_in_corridor` metric shown in the panel.
3. Sample N random OD pairs where either the origin or the destination is one of those nodes.
4. Compute each car's `w_fast` route (its natural fastest path).
5. Keep only the cars whose fastest route actually enters `blocked_edges`.
6. Apply the same 1000× penalty and re-run Dijkstra to get each car's detour.
7. Compute per-car: original time, rerouted time, Δ time, Δ distance, tokens earned.

The green polylines you see on the map are N independent reroutes, all computed with the same rule as the hero car. The panel aggregates: cars rerouted, total extra time, total tokens paid.

### Tokens — how much the driver earns

```python
tokens_earned = round(extra_time_min × TOKENS_PER_EXTRA_MINUTE × equity_multiplier)
```

Where `TOKENS_PER_EXTRA_MINUTE = 10` (a hackathon choice, easy to swap) and `equity_multiplier` defaults to 1.0 (extensible: the α slider from the ONDA tab is meant to flow into this later, so drivers from vulnerable zones would earn more per minute of extra travel).

If the reroute produces zero extra time (the alternative is exactly as fast), tokens = 0. Fair play: we compensate only for time actually lost. In the "Example" scenario the hero driver earns 31 tokens for 3.05 minutes of extra travel; the ambient simulation typically distributes 400–700 tokens across 15 drivers.

### Concrete example (from the "Example" button)

- **Car**: Financial District → Central Park.
- **Ambulance**: mid-Manhattan → uptown.
- **Original car route**: Broadway → Whitehall → Bridge → Broad → South → FDR → East 61st → 1st Ave → …, 13.53 km, uses FDR Drive.
- **Corridor** at 1200 m buffer: covers most of the east-side avenues down to about 3rd Avenue and up 79th Street. `n_nodes_in_corridor` ≈ 961.
- **`orig_path_in_corridor`** is `True` because FDR is inside the bubble.
- **`w_reroute`** = `w_fast × 1000` for every east-side avenue edge and every FDR edge; unchanged for the west side and mid-Manhattan cross streets.
- **Dijkstra with `w_reroute`** returns: Broadway → Battery Place → West Street → 11th Avenue → 12th Avenue → West Side Highway → Henry Hudson Parkway → West 79th Street. 14.62 km. That's the blue solid line on the map.
- **Delta**: 14.62 − 13.53 = 1.09 km, +3.05 min at ~20 kph observed speed.
- **Tokens**: `round(3.05 × 10 × 1.0) = 31`.

The algorithm never explicitly says "avoid FDR" or "take West Side Highway". It just says "FDR edges cost 1000× more today" and Dijkstra figures out the rest.

---

## Tab 3 — 🌊 ONDA Scenarios

The Explorer tells you *where* the city is stuck. The Route Planner tells you *what happens to one trip*. This tab asks: *what would a smarter signal-timing policy look like, and would it be fair?*

Three scenarios computed on the same 100 NYC intersections at rush hour (weekday 09:00, the busiest slice):

- **A — Today.** Baseline. Signal timings from the dataset, no intervention.
- **B — Efficiency-only AI (the villain).** A greedy diversion optimiser that reduces total city-wide delay by re-routing traffic between neighbouring intersections. It saves ~1 % of vehicle-hours but pushes 4 out of 6 zones above their baseline wait — the classic "Waze rat-running" effect writ large.
- **C — ONDA (the hero).** Max-pressure signal reallocation at each intersection, plus the same diversion loop as B but with a hard constraint: **no zone can end up worse than in A**. Whoever accepts a diversion or shifts departure out of the peak earns **tokens** redeemable for public transport passes and parking discounts.

### What the tab shows

- **Scenario picker** — three radio buttons (A / B / C).
- **KPI row** — vehicle-hours lost per day, CO₂ per day, zones worse than baseline (the "money metric"), worst-zone increase in seconds, tokens distributed.
- **3D map** of the 100 intersections. Each column's height is the mean wait time under the selected scenario. Columns are coloured **green** where the scenario is *better* than baseline A, and **red** where it is *worse*.
- **Per-zone bar chart** comparing the selected scenario to A for each of the six zones.
- **🎯 The trade-off chart.** X-axis: vehicle-hours saved vs today (higher is more efficient). Y-axis: the *worst zone's* extra wait in seconds (lower is fairer). A sits at the origin. B goes right and up. C goes right and stays at y = 0. This is the single most important chart in the pitch — it visualises the fairness/efficiency trade-off in one glance.
- **🏛️ Policy inputs — jury-editable.** A slider for α (the equity weight, default 1.0) and a table where each row is a city zone and the editable column is the *vulnerability* (0–1, higher = more protection). This is a *hand-off*: the jury decides who to protect; ONDA guarantees the protection is respected.
- **🎟️ Rewards panel.** Total tokens distributed (0 for B by construction), drivers rewarded, CO₂ saved. A per-zone breakdown table if C is selected.
- **🚑 Emergency corridor summary.** For a representative ambulance route: baseline signal delay (~4 min), corridor signal delay (~16 s), and a per-hop table.
- **🤖 Explain this scenario (LLM).** Ask a plain-English question and get an answer grounded in the numbers of `out/scenarios.json`.

### Where the numbers come from

The current values live in `out/scenarios.json` — a placeholder generated by:

```bash
python -m onda.fake_scenarios
```

This script writes a JSON with the **real** Scenario A computed from the dataset (mean wait per intersection at weekday 09:00) and **plausible synthetic** Scenarios B and C shaped to match the story (B lowers total delay but pushes 4 zones above A; C lowers total delay under δ = 0). The real simulator (WIP, planned to replace this) will emit the exact same JSON contract so no UI change is needed when the real numbers arrive.

The JSON contract is documented as the intended `scenarios.json` schema at the top of `onda/fake_scenarios.py`. Key sections:

```jsonc
{
  "meta": { "hour": 9, "day_kind": "weekday", "alpha_default": 1.0 },
  "zones": [ { "name": "Downtown Core", "intersections": 22, "vulnerability_default": 0.0 }, ... ],
  "intersections": [ { "id": "INT_001", "lat": 40.75, "lng": -73.99, "zone": "..." }, ... ],
  "edges": [ { "from": "INT_001", "to": "INT_007", "km": 0.77 }, ... ],
  "scenarios": {
    "A": { "label": "...", "metrics": { ... }, "per_zone": { ... }, "per_intersection": { ... }, "rewards": { ... } },
    "B": { ... },
    "C": { ... }
  },
  "emergency": { "route": [ ... ], "baseline_seconds": 240, "corridor_seconds": 16, "per_hop": [ ... ] }
}
```

Any real simulator that respects this schema will drop into the UI unchanged.

---

## Local LLM (Ollama) — optional but nice

The two "Explain" panels (Route Planner and ONDA Scenarios) call a local Ollama model. The client is a thin wrapper around the official [`ollama`](https://pypi.org/project/ollama/) Python package.

**Why local?** No API cost, no data leaves the machine (relevant for the hackathon venue and for a demo where the jury might want reproducibility), and inference on `qwen3:1.7b` runs at ~50 tokens/second on an Apple Silicon Mac — fast enough for a live streaming answer.

**Two dedicated system prompts** (see `onda/llm.py`):

- `SYSTEM_PROMPT` — used for the ONDA scenario explanations. Forbids inventing metrics, instructs the model to base every claim on the numbers provided, calls the fairness constraint by name.
- `REROUTE_SYSTEM_PROMPT` — used for the reroute explanations. Enforces the exact token formula, instructs the model to quote the real street names, and tells it that no reroute happens when there are zero conflicts (used to be true; now updated to "no reroute happens when the car's route does not enter the corridor buffer").

**If Ollama is not running**, both panels show a hint with the exact `ollama pull` and `ollama serve` commands to run. The rest of the app is unaffected.

---

## The dataset in detail

- **Source.** [mobeenfatimah/cityflow-smart-urban-mobility-and-traffic-iot](https://www.kaggle.com/datasets/mobeenfatimah/cityflow-smart-urban-mobility-and-traffic-iot) on Kaggle.
- **Shape.** 204,000 rows × 47 columns.
- **Grid.** 100 intersections × 2,040 hours = 204,000 rows. Time range: 2026-01-01 → 2026-03-26 (2 months, hourly).
- **Zones.** 6 city zones, unequal counts: Financial District (24), Downtown Core (22), Suburban North (26), Tech Park (11), Residential West (9), Industrial East (8).
- **Coordinates.** NYC — latitude 40.63–40.79, longitude −74.08 to −73.93 (Manhattan + parts of Brooklyn / Bronx).
- **Fields we use.** `latitude`, `longitude`, `city_zone`, `intersection_id`, `hour`, `day_of_week`, `is_weekend`, `average_speed`, `congestion_score`, `emission_estimate`, `fuel_waste_estimate`, `average_wait_time`, `air_quality_index`, `vehicle_count`, `lanes`, `green_light_duration`, `signal_cycle_seconds`.

### Two dataset traps we caught (worth mentioning in the pitch)

- **`traffic_flow_rate` is identical to `vehicle_count`** in 100 % of rows (correlation 1.0). It's a duplicate column; we drop it.
- **`air_quality_index` correlates only 0.034 with `vehicle_count`**, meaning it's essentially noise in this dataset. `emission_estimate`, on the other hand, correlates 0.924 with vehicle count — so we use `emission_estimate` as the impact metric, not AQI. Mentioning this on stage shows the jury we read the data critically.

### The saturation → wait curve (used by the fake ONDA scenarios)

An empirical relation computed from the dataset itself:

```
x = vehicle_count / (lanes × 1800 × green_light_duration / signal_cycle_seconds)
```

| Saturation x | Mean wait (s) | Rows |
|--------------|---------------|------|
| < 0.3 | 1.7 | 131,407 |
| 0.3 – 0.5 | 19.4 | 48,013 |
| 0.5 – 0.7 | 51.4 | 22,373 |
| 0.7 – 0.85 | 82.0 | 2,195 |
| > 0.85 | 101.0 | 12 |

This curve is what a real ONDA simulator would use: more green light → lower x → shorter wait. Simple and data-backed.

---

## Project layout

```
.
├── app.py                    # Streamlit UI: Explorer + Route Planner + ONDA tabs (~1100 lines)
├── router.py                 # OSM graph loading, metric enrichment, weighted routing,
│                             # ambulance reroute, corridor polygon, multi-car sim
├── import.py                 # Dataset download via kagglehub (local cache in ./data)
├── onda/
│   ├── __init__.py
│   ├── fake_scenarios.py     # Generates out/scenarios.json (placeholder sim)
│   ├── dashboard.py          # Streamlit render helpers for the ONDA tab
│   └── llm.py                # Ollama helper for "Explain this scenario / reroute"
├── out/
│   └── scenarios.json        # The A/B/C contract consumed by the UI
├── requirements.txt          # All Python deps, version-pinned
├── data/                     # CityFlow dataset (git-ignored, downloaded by import.py)
├── cache/                    # OSM graph cache (git-ignored, auto-created)
├── .venv/                    # Virtual env (git-ignored)
├── .gitignore
└── README.md                 # (this file)
```

---

## Key files, function by function

### `router.py`

The routing engine. Pure functions; no Streamlit imports.

- **`load_road_graph(bbox, pad=0.005)`** — downloads and caches the OSM drive graph for the bounding box (with `osmnx.graph_from_bbox`). Adds OSM-derived edge speeds and travel times.
- **`enrich_graph(G, intersections, global_ranges=None)`** — for each edge attaches:
  - `cong_norm`, `emit_norm`, `fuel_norm` — 0–1 normalisations of the three "impact" metrics from the nearest CityFlow intersection to the edge *midpoint*.
  - `obs_speed_kph` — observed average speed at the CityFlow intersection nearest to the edge *origin* node.
- **`add_weights(G, alpha_time=0.5, alpha_cong=0.25, alpha_eco=0.25)`** — computes per-edge `car_time_s`, `amb_corridor_time_s`, `w_fast`, `w_eco`, `w_balanced`. See "How travel time is computed" above.
- **`compute_routes(G, start_lat, start_lng, end_lat, end_lng)`** — returns the three alternative routes (Fastest / Balanced / Eco) with paths, coordinates and summaries.
- **`compute_ambulance_reroute(G, car_start, car_end, amb_start, amb_end, corridor_buffer_m, ambulance_dispatch_offset_s, meeting_window_s, equity_multiplier)`** — the big one. Returns a dict with the ambulance path, car original path, car rerouted path, space-time conflicts, tokens earned, per-edge speeds, and cumulative time arrays needed by the animation.
- **`compute_corridor_polygon(G, amb_path, buffer_m)`** — returns the buffer polygon around the ambulance path as `(lat, lng)` tuples ready for `folium.Polygon`. Uses `shapely.geometry.LineString.buffer`.
- **`simulate_corridor_traffic(G, amb_path, buffer_m, n_cars=12, random_state=42)`** — samples N random cars whose OD sits inside the corridor bubble, keeps only those whose fastest route actually crosses the corridor, and computes their detours + tokens.

### `app.py`

Streamlit UI wiring. All user-facing strings are in English. Structure:

1. Imports and helper functions (`_pos_at` for interpolating a vehicle position between path nodes, `_speed_at` for the current edge speed, `_haversine_m` for map distances).
2. Cached data loaders: `load_data()`, `load_base_graph(bbox)`, `enrich_graph_for_time(bbox, hour, day_kind)`, `global_metric_ranges()`.
3. Sidebar (Explorer filters + Ranking).
4. Three tabs, one big `with` block each.

The Route Planner tab is the biggest chunk (~600 lines). It defines two inner helper functions:

- `_build_side_deck(car_coords, car_pos, car_color, amb_pos, highlight_conflict)` — builds a pydeck for one side of the animation panel.
- `_render_frame(t)` — renders both animation frames for time cursor `t`.

### `onda/dashboard.py`

Pure render helpers for the ONDA Scenarios tab.

- `load_scenarios()` — reads `out/scenarios.json` (cached).
- `render_kpi_row(scenarios, current, baseline="A")` — the five headline metrics with vs-baseline deltas.
- `render_per_zone_chart(scenarios, current, baseline="A")` — grouped bar chart.
- `render_efficiency_chart(scenarios)` — **the money chart** (efficiency vs worst-zone increase).
- `render_map(scenarios, current, baseline="A")` — 3D pydeck with colour-coded delta.
- `render_rewards_panel(scenarios, current)` — token distribution table.
- `render_vulnerability_editor(scenarios)` — α slider + editable per-zone vulnerability table.
- `render_emergency_summary(scenarios)` — ambulance corridor mini-table.
- `scenario_summary_text(scenarios)` — compact text summary used as LLM context.

### `onda/llm.py`

Thin Ollama client with two system prompts and a `stream_answer` generator that streams the model's response chunk by chunk (used by `st.write_stream`).

### `onda/fake_scenarios.py`

Standalone script (`python -m onda.fake_scenarios`) that writes `out/scenarios.json`. Scenario A uses real dataset values; B and C are plausible synthetics shaped by hand.

---

## Suggested demo script for the pitch

Roughly 5 minutes end-to-end. Test it twice with a stopwatch before going on stage.

1. **Open 🗺️ Explorer 3D.** Filter to hour 8–9 weekday. Metric = `congestion_score`. Layer = *Columns per intersection*. Enable ranking, top-10. Point at the map: Downtown Core is red, Suburban North is green. *"The centre is worse today — that's not the fairness problem we're fixing."*
2. **Open 🚗 Route Planner, ambulance mode.** Click **📍 Example (car + ambulance)**. Point at the corridor bubble. Drag the corridor width slider from 400 to 1500 m — watch the bubble expand and the ambient cars turn from grey to green as they're forced into detours. *"On a 1200 m corridor, 15 sampled drivers are affected — total 633 tokens paid."*
3. **Scroll to the animation, press ▶ Play.** On the right map (hypothetical) the car eventually reaches the ambulance's position — badge flashes ⚠️ COLLISION. On the left (actual), the car stays clear. *"This is what the reroute avoids."*
4. **Click "Explain and compute tokens".** The local LLM prints the turn-by-turn streets (Broadway → Battery Place → West Street → …) and the token formula. *"No API, no data leaves the room, ~2 seconds to answer."*
5. **Open 🌊 ONDA Scenarios.** Point at the trade-off chart. Flip A → B → C. *"A sits at the origin. B saves 0.9 % of vehicle-hours but four zones pay for it. C saves 2.1 % and no zone is worse off."*
6. **Ask a jury question** in the "Explain this scenario" panel. Something like *"why is C better than B for peripheral zones?"* — the LLM answers grounded in the numbers on screen.

**Fallback:** record a 90-second screen capture of steps 2-3-5 before the pitch. If the laptop dies on stage, play the video.

---

## Troubleshooting

- **The dataset didn't download.** Check that `kagglehub` can reach Kaggle (needs credentials the first time). Fall back: manually download from Kaggle and place under `data/datasets/mobeenfatimah/cityflow-smart-urban-mobility-and-traffic-iot/versions/1/smart_city_traffic_mobility.csv`.
- **Route Planner takes forever to open the first time.** It's downloading the NYC OSM drive network (~50 MB). Subsequent opens are cached.
- **Ambulance mode shows "The car's original route does not enter the protected corridor — no reroute needed"** even though I clicked points near the ambulance path. Increase the corridor width slider. On Manhattan you often need ≥ 1200 m to force a real detour because of parallel highways.
- **`StreamlitWidgetAlreadyInstantiatedError` on target_slot.** Should be fixed; if it comes back, the culprit is a write to `st.session_state.target_slot` after `st.radio(key="target_slot")` runs. Move the write above the widget.
- **Ollama panels say "Ollama is not reachable".** Start `ollama serve` in a separate terminal and `ollama pull qwen3:1.7b`. Re-open the panel.
- **Streamlit shows the app but the map is blank.** Refresh the page. The pydeck WebGL context sometimes fails to initialise on the first render.
- **`ImportError: scikit-learn must be installed as an optional dependency...`** from osmnx. `pip install scikit-learn` (it's already in `requirements.txt`).

---

## Roadmap / open work

The synthetic ONDA scenarios are placeholders. The plan is to replace `onda/fake_scenarios.py` with a real simulator under `onda/sim.py` that:

1. Builds a k-nearest-neighbour graph over the 100 intersections (haversine, k = 4).
2. Runs **max-pressure signal reallocation** at each intersection (this alone hurts nobody).
3. Runs a **greedy diversion loop** on the kNN graph. Two versions: unconstrained (B) and constrained to `mean_wait_z(C) ≤ mean_wait_z(A) + δ, δ = 0` for every zone z (C).
4. Emits the same `scenarios.json` schema.
5. Optional: LightGBM 1h-ahead forecast for the "Insights" tab.

Nothing in the UI will change when the real simulator lands — just re-run `python -m onda.sim` in place of `python -m onda.fake_scenarios`.

The Route Planner is essentially feature-complete for the demo. Nice-to-haves:

- Real drag-and-drop markers (currently blocked by `streamlit-folium` not exposing `dragend` events; workaround via the slot selector + click).
- Multi-agent animation showing all N ambient cars moving, not only the hero.
- Wire the α slider from the ONDA tab into the token formula (`equity_multiplier`), so protected zones earn more per minute of extra travel.

---

## Credits

- Dataset — [Mobeen Fatimah on Kaggle](https://www.kaggle.com/datasets/mobeenfatimah/cityflow-smart-urban-mobility-and-traffic-iot).
- Base map tiles — [Carto](https://carto.com/) and [OpenStreetMap](https://www.openstreetmap.org/) contributors.
- Road network — [OpenStreetMap](https://www.openstreetmap.org/) via [OSMnx](https://osmnx.readthedocs.io/).
- 3D rendering — [deck.gl](https://deck.gl/) via [pydeck](https://pydeck.gl/), and [Folium](https://python-visualization.github.io/folium/) via [streamlit-folium](https://github.com/randyzwitch/streamlit-folium).
- Local LLM — [Ollama](https://ollama.com/) with `qwen3:1.7b` from [Alibaba's Qwen team](https://huggingface.co/Qwen).
- UI — [Streamlit](https://streamlit.io/).

Built for the **IEEE Smart World Congress 2026 hackathon**.
