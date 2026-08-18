"""
telemedicina_supervised/ml/scaler.py — Standardizzazione feature.

Media e deviazione standard sono calcolate SOLO sul train (mai sul test:
sarebbe leakage) e vengono salvate insieme al modello nell'artifact, così il
SupervisedAgent applica in inferenza la stessa trasformazione vista in
training. Nessuna soglia clinica qui.
"""

from typing import Optional

import numpy as np


class StandardScaler:
    """Scaler per colonna: (x - media) / deviazione (ddof=0)."""

    def __init__(self) -> None:
        self.media: Optional[np.ndarray] = None
        self.dev: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        """Stima media/deviazione su X (chiamare SOLO sul train)."""
        X = np.asarray(X, dtype=np.float64)
        media = X.mean(axis=0)
        dev = X.std(axis=0)
        # Evita divisione per zero su feature costanti.
        dev[dev == 0.0] = 1.0
        self.media = media
        self.dev = dev
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.media is None or self.dev is None:
            raise RuntimeError("StandardScaler non addestrato: chiamare fit() prima")
        return (np.asarray(X, dtype=np.float64) - self.media) / self.dev

    def inverso(self, X_norm: np.ndarray) -> np.ndarray:
        """Riporta i dati standardizzati nello spazio originale."""
        if self.media is None or self.dev is None:
            raise RuntimeError("StandardScaler non addestrato: chiamare fit() prima")
        return (
            np.asarray(X_norm, dtype=np.float64) * self.dev + self.media
        )

    def to_dict(self) -> dict:
        if self.media is None or self.dev is None:
            raise RuntimeError("StandardScaler non addestrato: chiamare fit() prima")
        return {"media": self.media, "dev": self.dev}

    @classmethod
    def from_dict(cls, dati: dict) -> "StandardScaler":
        scaler = cls()
        scaler.media = np.asarray(dati["media"], dtype=np.float64)
        scaler.dev = np.asarray(dati["dev"], dtype=np.float64)
        return scaler
