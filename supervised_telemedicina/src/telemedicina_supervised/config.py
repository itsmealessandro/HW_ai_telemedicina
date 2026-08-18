"""Configurazione condivisa della CLI e del training."""

from pathlib import Path
from typing import Optional

from telemedicina_supervised.ml.mlp import MLP

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "data" / "models"
MODEL_ARTIFACT_PATH = MODELS_DIR / "mlp_telemedicina.npz"
REPORT_PATH = MODELS_DIR / "report.json"
SEED_DEFAULT = 41
SOGLIA_INCERTEZZA = 0.6

# Copia immutabile dei default della griglia di training, esposta per la CLI.
TRAINING_HYPERPARAMETERS = (
    {"n_hidden": 16, "lr": 0.01, "batch_size": 32, "epoche": 25},
    {"n_hidden": 16, "lr": 0.05, "batch_size": 32, "epoche": 25},
    {"n_hidden": 32, "lr": 0.01, "batch_size": 32, "epoche": 25},
    {"n_hidden": 32, "lr": 0.05, "batch_size": 32, "epoche": 25},
    {"n_hidden": 64, "lr": 0.01, "batch_size": 64, "epoche": 25},
    {"n_hidden": 64, "lr": 0.05, "batch_size": 64, "epoche": 25},
)
GRID = TRAINING_HYPERPARAMETERS


def carica_modello(path: Optional[Path] = None) -> Optional[MLP]:
    """Carica un artifact valido; per qualunque errore restituisce ``None``."""
    try:
        modello = MLP.carica(Path(path) if path is not None else MODEL_ARTIFACT_PATH)
        if tuple(modello.classi) != ("basso", "medio", "alto"):
            return None
        return modello
    except Exception:
        return None
