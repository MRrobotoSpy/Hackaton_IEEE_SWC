# Hackaton_IEEE_SWC — CityFlow 3D Traffic Explorer

Visualizzatore 3D interattivo del dataset **CityFlow: Smart Urban Mobility & Traffic IoT** (Kaggle), pensato per esplorare congestione, emissioni e qualità dell'aria delle intersezioni di una città con una mappa 3D stile Google Maps.

## Setup

```bash
# 1. Crea il virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Installa le dipendenze
pip install -r requirements.txt

# 3. Scarica il dataset (viene salvato in ./data/, non nella cache utente)
python import.py
```

## Avvio dell'app

```bash
streamlit run app.py
```

L'app si apre su http://localhost:8501.

## Cosa fa

- **Mappa 3D** tiltabile (drag col tasto destro) con basemap selezionabile: Road, OpenStreetMap, Satellite, Light, Dark.
- **Tre modalità di visualizzazione**:
  - *Colonne per intersezione* — un pilastro 3D per ogni intersezione, altezza = metrica scelta.
  - *Scatter preciso* — un cerchio per intersezione alle coordinate esatte.
  - *Hexagon aggregato* — binning esagonale per overview su aree ampie.
- **Filtri**: city zone, fascia oraria, giorno della settimana, tipo di aggregazione temporale (media / max / somma / ultimo valore).
- **Metriche disponibili**: `vehicle_count`, `congestion_score`, `average_wait_time`, `queue_length`, `emission_estimate`, `air_quality_index`.
- **Ranking intersezioni**: score composito di "badness" (media normalizzata di congestione, emissioni, spreco carburante, AQI, tempo d'attesa) con evidenziazione delle top-N peggiori (rosso) e migliori (verde) sulla mappa + tabelle di confronto side-by-side.

## Struttura del progetto

```
.
├── app.py              # Streamlit + pydeck app
├── import.py           # Download dataset via kagglehub (cache locale in ./data)
├── requirements.txt
├── data/               # Dataset (ignorato da git)
└── .venv/              # Virtual env (ignorato da git)
```

## Dataset

- Fonte: [mobeenfatimah/cityflow-smart-urban-mobility-and-traffic-iot](https://www.kaggle.com/datasets/mobeenfatimah/cityflow-smart-urban-mobility-and-traffic-iot)
- 204.000 record orari su 100 intersezioni di NYC, con metriche di traffico, meteo, qualità dell'aria ed emissioni.
- Cartella locale: `data/datasets/mobeenfatimah/cityflow-smart-urban-mobility-and-traffic-iot/versions/1/`
