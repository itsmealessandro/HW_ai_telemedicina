"""
telemedicina_supervised/ml/labels.py — Conversione stabile label <-> indici.

Il MLP lavora con indici interi; il dataset salvato da Fase 2 contiene label
come stringhe ('basso'/'medio'/'alto'). L'ordine degli indici è quello della
sequenza `classi` passata dal chiamante (il contratto la definisce una sola
volta; qui non è duplicata alcuna soglia).
"""

from typing import Dict, Sequence, Union

import numpy as np


def to_indici(y: Union[Sequence[str], np.ndarray], classi: Sequence[str]) -> np.ndarray:
    """Converte un array di label (stringhe) in indici interi (int64)."""
    mappa = {classe: i for i, classe in enumerate(classi)}
    try:
        return np.array([mappa[etichetta] for etichetta in y], dtype=np.int64)
    except KeyError as errore:
        raise ValueError(
            f"Label sconosciuta {errore}: attese {list(mappa)}"
        ) from errore


def to_etichette(indici: np.ndarray, classi: Sequence[str]) -> np.ndarray:
    """Converte un array di indici interi in label (stringhe).

    La dtype è derivata dalla lunghezza massima dei nomi di classe, così
    nessuna etichetta viene troncata silenziosamente.
    """
    indici = np.asarray(indici, dtype=np.int64)
    if indici.size and (indici.min() < 0 or indici.max() >= len(classi)):
        raise ValueError("Indice fuori intervallo rispetto alla sequenza classi")
    larghezza = max(len(classe) for classe in classi)
    return np.array([classi[i] for i in indici], dtype=f"U{larghezza}")


def mappa_indici(classi: Sequence[str]) -> Dict[str, int]:
    """Dizionario classe -> indice (utile per predizioni a livello agente)."""
    return {classe: i for i, classe in enumerate(classi)}
