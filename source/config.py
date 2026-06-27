# source/config.py
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
SOURCE_DIR = ROOT / "source"
OUT_DIR = ROOT / "out"
INPUT_DIR = ROOT / "inputs"
PLOT_DIR = ROOT / "plot"
MODEL_DIR = SOURCE_DIR / "models"
