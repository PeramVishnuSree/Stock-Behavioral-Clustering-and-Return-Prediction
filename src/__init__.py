"""
Stock Behavioral Clustering & Return Prediction
DATA 255 Final Project — Source modules.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR    = PROJECT_ROOT / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)
