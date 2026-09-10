# Hackaton_IEEE_SWC — CityFlow 3D Traffic Explorer

Interactive 3D visualizer for the **CityFlow: Smart Urban Mobility & Traffic IoT** Kaggle dataset. Explore per-intersection congestion, emissions and air quality on a tiltable 3D map (Google-Maps-style perspective).

## Setup

```bash
# 1. Create the virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset (stored in ./data, not in the user cache)
python import.py
```

## Run the app

```bash
streamlit run app.py
```

The app opens at http://localhost:8501.

## Features

- **3D tiltable map** (right-click drag to rotate/pitch) with selectable basemap: Road, OpenStreetMap, Satellite, Light, Dark.
- **Three visualization modes**:
  - *Columns per intersection* — one 3D pillar per intersection, height = selected metric.
  - *Precise scatter* — one dot per intersection at its exact coordinates.
  - *Aggregated hexagon* — hex binning for a wide-area overview.
- **Filters**: city zone, hour range, day of week, temporal aggregation (mean / max / sum / last).
- **Metrics**: `vehicle_count`, `congestion_score`, `average_wait_time`, `queue_length`, `emission_estimate`, `air_quality_index`.
- **Intersection ranking**: composite *badness* score (normalized mean of congestion, emissions, fuel waste, AQI, wait time). The top-N worst intersections are highlighted in red and the top-N best in green on the map, with side-by-side comparison tables underneath.

## Project layout

```
.
├── app.py              # Streamlit + pydeck app
├── import.py           # Dataset download via kagglehub (local cache in ./data)
├── requirements.txt
├── data/               # Dataset (git-ignored)
└── .venv/              # Virtual env (git-ignored)
```

## Dataset

- Source: [mobeenfatimah/cityflow-smart-urban-mobility-and-traffic-iot](https://www.kaggle.com/datasets/mobeenfatimah/cityflow-smart-urban-mobility-and-traffic-iot)
- 204,000 hourly records across 100 NYC intersections, including traffic, weather, air quality and emission metrics.
- Local path: `data/datasets/mobeenfatimah/cityflow-smart-urban-mobility-and-traffic-iot/versions/1/`
