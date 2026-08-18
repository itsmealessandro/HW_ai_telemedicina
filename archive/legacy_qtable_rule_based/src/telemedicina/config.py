"""
config.py - Configurazione centralizzata del sistema di telemedicina.
"""

from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DB_DIR = PROJECT_ROOT / "data"
DB_PATH = str(DB_DIR / "telemedicina.db")

LOG_DIR = str(PROJECT_ROOT / "data" / "logs")

# RL (Reinforcement Learning)
RL_QTABLE_PATH = str(PROJECT_ROOT / "data" / "models" / "q_table.pkl")
RL_ALPHA = 0.1
RL_GAMMA = 0.9
RL_EPSILON_INIT = 1.0
RL_EPSILON_DECAY = 0.99996
RL_EPSILON_MIN = 0.05
RL_EPISODI = 100000
