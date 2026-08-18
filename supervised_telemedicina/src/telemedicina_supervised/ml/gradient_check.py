"""
telemedicina_supervised/ml/gradient_check.py — Verifica della backpropagation.

Confronta i gradienti analitici (chain rule) con differenze finite centrali
su una rete piccola: criterio oggettivo di correttezza del backward pass.
Nessuna soglia clinica qui.
"""

from typing import Dict

import numpy as np

from telemedicina_supervised.ml.mlp import MLP


def _gradiente_numerico(
    modello: MLP,
    X: np.ndarray,
    y: np.ndarray,
    eps: float = 1e-6,
) -> Dict[str, np.ndarray]:
    """Gradienti della loss con differenze finite centrali (per parametro)."""
    gradi: Dict[str, np.ndarray] = {}
    for nome in ("W1", "b1", "W2", "b2"):
        parametro = getattr(modello, nome)
        grad = np.zeros_like(parametro)
        for indice in np.ndindex(parametro.shape):
            originale = parametro[indice]

            parametro[indice] = originale + eps
            loss_piu = modello.loss(X, y)

            parametro[indice] = originale - eps
            loss_meno = modello.loss(X, y)

            parametro[indice] = originale
            grad[indice] = (loss_piu - loss_meno) / (2.0 * eps)
        gradi[nome] = grad
    return gradi


def errore_massimo_relativo(
    modello: MLP,
    X: np.ndarray,
    y: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Errore relativo massimo tra gradienti analitici e numerici.

    Errore relativo per parametro: |analitico - numerico| /
    max(1, |analitico|, |numerico|). Restituisce il massimo su tutti i
    parametri. Valori sotto ~1e-5 indicano backward corretto.
    """
    analitici = modello.gradienti(X, y)
    numerici = _gradiente_numerico(modello, X, y, eps=eps)
    massimo = 0.0
    for nome in ("W1", "b1", "W2", "b2"):
        a = analitici[nome]
        n = numerici[nome]
        denominatore = np.maximum(1.0, np.maximum(np.abs(a), np.abs(n)))
        errore = np.max(np.abs(a - n) / denominatore)
        massimo = max(massimo, float(errore))
    return massimo