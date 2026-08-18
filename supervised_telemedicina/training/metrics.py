"""
training/metrics.py — Metriche di valutazione in numpy puro.

Nessuna dipendenza da sklearn: accuracy, precision/recall/F1 per classe,
macro-F1, matrice di confusione, Cohen's kappa, analisi degli errori per
distanza dalle soglie (derivata da safety_rules, fonte unica) e metriche
di sicurezza (mancato riconoscimento della classe 'alto' e dei pattern
critici). Nessun literal di soglia è duplicato qui.
"""

from typing import Any, Dict, List

import numpy as np

from telemedicina_supervised.safety.safety_rules import (
    FEATURE_ORDER,
    RANGE_NORMALI,
    SOGLIE_CRITICHE,
    SOGLIE_MODERATE,
)


def accuratezza(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Frazione di predizioni corrette."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def matrice_confusione(
    y_true: np.ndarray, y_pred: np.ndarray, n_classi: int
) -> np.ndarray:
    """Matrice n_classi x n_classi: righe = vere, colonne = predette."""
    matrice = np.zeros((n_classi, n_classi), dtype=np.int64)
    for vero, predetto in zip(y_true, y_pred):
        matrice[vero, predetto] += 1
    return matrice


def metriche_per_classe(
    y_true: np.ndarray, y_pred: np.ndarray, n_classi: int
) -> Dict[str, Dict[str, float]]:
    """Precision, recall e F1 per ogni classe (indici 0..n_classi-1)."""
    matrice = matrice_confusione(y_true, y_pred, n_classi)
    risultato: Dict[str, Dict[str, float]] = {}
    for classe in range(n_classi):
        veri_positivi = matrice[classe, classe]
        predetti_positivi = int(matrice[:, classe].sum())
        reali_positivi = int(matrice[classe, :].sum())
        precision = veri_positivi / predetti_positivi if predetti_positivi else 0.0
        recall = veri_positivi / reali_positivi if reali_positivi else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0.0
            else 0.0
        )
        risultato[str(classe)] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
    return risultato


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classi: int) -> float:
    """Media delle F1 per classe (macro)."""
    per_classe = metriche_per_classe(y_true, y_pred, n_classi)
    return float(
        np.mean([per_classe[str(c)]["f1"] for c in range(n_classi)])
    )


def kappa_cohen(y_true: np.ndarray, y_pred: np.ndarray, n_classi: int) -> float:
    """Cohen's kappa: concordanza corretta per il caso (accordo atteso)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = y_true.size
    if n == 0:
        return 0.0
    matrice = matrice_confusione(y_true, y_pred, n_classi)
    accordo_osservato = np.trace(matrice) / n
    righe = matrice.sum(axis=1) / n
    colonne = matrice.sum(axis=0) / n
    accordo_atteso = float(np.sum(righe * colonne))
    if accordo_atteso == 1.0:
        return 1.0 if accordo_osservato == 1.0 else 0.0
    return float((accordo_osservato - accordo_atteso) / (1.0 - accordo_atteso))


def distanza_dalle_soglie(X: np.ndarray) -> np.ndarray:
    """Distanza minima di ogni campione dai confini di classificazione.

    Per ogni feature si considera la distanza dal confine più vicino tra
    soglie critiche/moderate (alta e bassa) e range normali, NORMALIZZATA
    per l'ampiezza del range normale della feature (unità: "ampiezze di
    range normale"). Senza normalizzazione la temperatura (range normale
    stretto) dominerebbe la distanza minima di quasi tutti i campioni,
    rendendo l'analisi errori priva di discriminazione. Un campione
    esattamente su un confine ha distanza 0; valori lontani da ogni
    confine hanno distanza grande. Usata per l'analisi degli errori:
    gli errori vicini ai confini sono attesi (confini arbitrari), quelli
    lontani sono più preoccupanti.
    """
    X = np.asarray(X, dtype=np.float64)
    distanze = np.full(X.shape[0], np.inf)
    for i, nome in enumerate(FEATURE_ORDER):
        valori = X[:, i]
        min_n, max_n = RANGE_NORMALI[nome]
        ampiezza = max_n - min_n
        confini: List[float] = []
        for tabella in (SOGLIE_CRITICHE, SOGLIE_MODERATE):
            for soglia in tabella[nome].values():
                if soglia is not None:
                    confini.append(float(soglia))
        confini.extend([min_n, max_n])
        for confine in confini:
            distanze = np.minimum(
                distanze, np.abs(valori - confine) / ampiezza
            )
    return distanze


def metriche_sicurezza(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classi: int,
    indice_alto: int,
) -> Dict[str, float]:
    """Metriche di sicurezza: mancato riconoscimento della classe 'alto'.

    - recall_alto: frazione di veri 'alto' riconosciuti;
    - mancati_alto: numero di 'alto' classificati come non-alto;
    - declassati_critici: 'alto' predetti come qualunque classe non-alto
      (peggiore errore: anche il declassamento a 'medio' è critico).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    veri_alto = y_true == indice_alto
    n_alto = int(veri_alto.sum())
    riconosciuti = int((veri_alto & (y_pred == indice_alto)).sum())
    declassati_basso = int(
        (veri_alto & (y_pred < indice_alto)).sum()
    )
    return {
        "recall_alto": float(riconosciuti / n_alto) if n_alto else 0.0,
        "mancati_alto": n_alto - riconosciuti,
        "declassati_critici": declassati_basso,
    }


def riepilogo_metriche(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classi: int,
    indice_alto: int,
) -> Dict[str, Any]:
    """Riepilogo completo delle metriche per il report."""
    return {
        "accuracy": accuratezza(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred, n_classi),
        "kappa_cohen": kappa_cohen(y_true, y_pred, n_classi),
        "per_classe": metriche_per_classe(y_true, y_pred, n_classi),
        "matrice_confusione": matrice_confusione(
            y_true, y_pred, n_classi
        ).tolist(),
        "sicurezza": metriche_sicurezza(
            y_true, y_pred, n_classi, indice_alto
        ),
    }