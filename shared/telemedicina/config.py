import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DB_DIR = PROJECT_ROOT / "data"
DB_PATH = str(DB_DIR / "telemedicina.db")

LOG_DIR = str(PROJECT_ROOT / "data" / "logs")

RL_QTABLE_PATH = str(PROJECT_ROOT / "data" / "models" / "q_table.pkl")
RL_ALPHA = 0.1
RL_GAMMA = 0.9
RL_EPSILON_INIT = 1.0
RL_EPSILON_DECAY = 0.99996
RL_EPSILON_MIN = 0.05
RL_EPISODI = 100000
RL_REWARDS_PATH = str(PROJECT_ROOT / "data" / "models" / "training_rewards.npy")
RL_QTABLES_HISTORY_PATH = str(PROJECT_ROOT / "data" / "models" / "qtables_history.npy")

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
