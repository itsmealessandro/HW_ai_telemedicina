"""
config.py - Configurazione centralizzata del sistema di telemedicina.
"""

from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DB_DIR = PROJECT_ROOT / "data"
DB_PATH = str(DB_DIR / "telemedicina.db")

LOG_DIR = str(PROJECT_ROOT / "data" / "logs")
