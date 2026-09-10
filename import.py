import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ["KAGGLEHUB_CACHE"] = str(PROJECT_ROOT / "data")

import kagglehub

path = kagglehub.dataset_download("mobeenfatimah/cityflow-smart-urban-mobility-and-traffic-iot")

print("Path to dataset files:", path)
