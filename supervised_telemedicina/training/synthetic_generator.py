"""
training/synthetic_generator.py — Generatore sintetico a profili correlati.

Genera combinazioni di parametri vitali fisiologicamente plausibili e le
etichetta con il teacher rule-based (training/teacher_rules.py), che riusa
esclusivamente `safety_rules` (fonte unica delle soglie).

VINCOLI DI PROGETTO (dal piano, Fase 2):
  - Campionamento CORRELATO: la pressione diastolica è campionata
    condizionata alla sistolica (vincolo strutturale `sistolica > diastolica`
    con differenza minima DIFF_MIN_SIST_DIAST, scelta di plausibilità
    documentata). Il campionamento indipendente produrrebbe combinazioni
    fisiologicamente assurde.
  - Limiti di campionamento derivati da RANGE_FISIOLOGICI di safety_rules:
    nessuna soglia numerica è duplicata in questo modulo (guard test
    tests/test_single_source.py).
  - Buffer zone: con `buffer_zone=True` vengono scartati i campioni che
    cadono entro MARGINI_BUFFER da una qualunque soglia di confine
    (SOGLIE_CRITICHE, SOGLIE_MODERATE, RANGE_NORMALI), per ridurre il rumore
    di label sul gradino della classificazione.
  - Seed: vengono settati ENTRAMBI i generatori (`random.seed` per la scelta
    del profilo, `np.random.default_rng` per i valori), come da piano ⚠️.
  - I campioni con label 'errore' (input invalidi) vengono scartati e
    contati: il dataset di training non contiene mai 'errore'.

I valori medi/deviazione dei profili sono PARAMETRI DI DISTRIBUZIONE, non
soglie cliniche: definiscono dove campionare, non come classificare.
"""

import random
from typing import Dict, List, Tuple

import numpy as np

from telemedicina_supervised.safety.safety_rules import (
    FEATURE_ORDER,
    RANGE_FISIOLOGICI,
    RANGE_NORMALI,
    SOGLIE_CRITICHE,
    SOGLIE_MODERATE,
)
from training.teacher_rules import label

# Differenza minima sistolica-diastolica (mmHg): scelta di plausibilità
# fisiologica del generatore, NON una soglia clinica.
DIFF_MIN_SIST_DIAST: float = 10.0

# Margini della buffer zone per parametro (unità di misura del parametro).
# Un campione con un valore entro `margine` da una soglia di confine viene
# scartato (solo con buffer_zone=True).
MARGINI_BUFFER: Dict[str, float] = {
    "pressione_sistolica": 2.0,   # mmHg
    "pressione_diastolica": 2.0,  # mmHg
    "frequenza_cardiaca": 1.0,    # bpm
    "temperatura": 0.2,           # °C
    "saturazione_ossigeno": 1.0,  # %
    "glicemia": 2.0,              # mg/dL
}

# Cap di tentativi per campione richiesto (rejection sampling): evita loop
# infiniti se la buffer zone scarta quasi tutto.
MAX_TENTATIVI_PER_CAMPIONE: int = 99

# ----------------------------------------------------------------------
# Medie di temperatura dei profili (PARAMETRI DI DISTRIBUZIONE, non soglie
# cliniche). I valori coincidono con soglie moderate di safety_rules: il
# marcatore noqa documenta che qui NON sono soglie, ma medie gaussiane.
# ----------------------------------------------------------------------
TEMP_NORMALE: float = 36.8  # noqa: soglia derivata — media di profilo
TEMP_LEGGERA: float = 36.5  # noqa: soglia derivata — media di profilo
TEMP_FEBBRE: float = 38.6  # noqa: soglia derivata — media di profilo

# ----------------------------------------------------------------------
# Profili fisiologici (mixture).
# Ogni profilo: 'peso' (intero; la somma è normalizzata da random.choices)
# + per ogni feature una coppia (media, deviazione standard) di una
# gaussiana, poi clip ai range fisiologici. I pesi realizzano
# l'oversampling delle regioni anomale.
# ----------------------------------------------------------------------
PROFILI: Dict[str, Dict[str, object]] = {
    "normale": {
        "peso": 25,
        "pressione_sistolica": (118.0, 10.0),
        "pressione_diastolica": (76.0, 8.0),
        "frequenza_cardiaca": (74.0, 8.0),
        "temperatura": (TEMP_NORMALE, 0.3),
        "saturazione_ossigeno": (98.0, 1.0),
        "glicemia": (96.0, 12.0),
    },
    "ipertensione_lieve": {
        "peso": 10,
        "pressione_sistolica": (148.0, 8.0),
        "pressione_diastolica": (94.0, 6.0),
        "frequenza_cardiaca": (78.0, 8.0),
        "temperatura": (TEMP_NORMALE, 0.3),
        "saturazione_ossigeno": (98.0, 1.0),
        "glicemia": (96.0, 12.0),
    },
    "ipertensione_severa": {
        "peso": 8,
        "pressione_sistolica": (184.0, 8.0),
        "pressione_diastolica": (112.0, 6.0),
        "frequenza_cardiaca": (84.0, 10.0),
        "temperatura": (TEMP_NORMALE, 0.3),
        "saturazione_ossigeno": (97.0, 1.5),
        "glicemia": (96.0, 12.0),
    },
    "ipotensione": {
        "peso": 8,
        "pressione_sistolica": (86.0, 6.0),
        "pressione_diastolica": (56.0, 5.0),
        "frequenza_cardiaca": (72.0, 8.0),
        "temperatura": (TEMP_LEGGERA, 0.3),
        "saturazione_ossigeno": (97.0, 1.5),
        "glicemia": (91.0, 10.0),
    },
    "tachicardia": {
        "peso": 8,
        "pressione_sistolica": (118.0, 10.0),
        "pressione_diastolica": (76.0, 8.0),
        "frequenza_cardiaca": (114.0, 8.0),
        "temperatura": (TEMP_NORMALE, 0.3),
        "saturazione_ossigeno": (98.0, 1.0),
        "glicemia": (96.0, 12.0),
    },
    "bradicardia": {
        "peso": 6,
        "pressione_sistolica": (118.0, 10.0),
        "pressione_diastolica": (76.0, 8.0),
        "frequenza_cardiaca": (52.0, 4.0),
        "temperatura": (TEMP_LEGGERA, 0.3),
        "saturazione_ossigeno": (97.0, 1.5),
        "glicemia": (96.0, 12.0),
    },
    "febbre": {
        "peso": 8,
        "pressione_sistolica": (118.0, 10.0),
        "pressione_diastolica": (76.0, 8.0),
        "frequenza_cardiaca": (96.0, 10.0),
        "temperatura": (TEMP_FEBBRE, 0.4),
        "saturazione_ossigeno": (97.0, 1.5),
        "glicemia": (96.0, 12.0),
    },
    "ipossia": {
        "peso": 6,
        "pressione_sistolica": (118.0, 10.0),
        "pressione_diastolica": (76.0, 8.0),
        "frequenza_cardiaca": (86.0, 10.0),
        "temperatura": (TEMP_NORMALE, 0.3),
        "saturazione_ossigeno": (89.0, 3.0),
        "glicemia": (96.0, 12.0),
    },
    "ipoglicemia": {
        "peso": 5,
        "pressione_sistolica": (118.0, 10.0),
        "pressione_diastolica": (76.0, 8.0),
        "frequenza_cardiaca": (82.0, 10.0),
        "temperatura": (TEMP_NORMALE, 0.3),
        "saturazione_ossigeno": (98.0, 1.0),
        "glicemia": (56.0, 5.0),
    },
    "iperglicemia": {
        "peso": 5,
        "pressione_sistolica": (118.0, 10.0),
        "pressione_diastolica": (76.0, 8.0),
        "frequenza_cardiaca": (82.0, 10.0),
        "temperatura": (TEMP_NORMALE, 0.3),
        "saturazione_ossigeno": (98.0, 1.0),
        "glicemia": (218.0, 18.0),
    },
    "shock": {
        "peso": 3,
        "pressione_sistolica": (82.0, 6.0),
        "pressione_diastolica": (54.0, 5.0),
        "frequenza_cardiaca": (118.0, 8.0),
        "temperatura": (TEMP_LEGGERA, 0.3),
        "saturazione_ossigeno": (94.0, 2.0),
        "glicemia": (91.0, 10.0),
    },
    "febbre_tachicardia": {
        "peso": 8,  # combinazione multi-anomalia (febbre + tachicardia)
        "pressione_sistolica": (118.0, 10.0),
        "pressione_diastolica": (76.0, 8.0),
        "frequenza_cardiaca": (114.0, 8.0),
        "temperatura": (TEMP_FEBBRE, 0.4),
        "saturazione_ossigeno": (97.0, 1.5),
        "glicemia": (96.0, 12.0),
    },
}

# Somma dei pesi = 25+10+8+8+8+6+8+6+5+5+3+8 (normalizzata da random.choices).
_NOMI_PROFILI: List[str] = list(PROFILI.keys())
_PESI: List[int] = [int(PROFILI[n]["peso"]) for n in _NOMI_PROFILI]  # type: ignore[arg-type]

# Cache delle soglie di confine per parametro (critiche + moderate + normali).
_SOGLIE_CONFINE: Dict[str, Tuple[float, ...]] = {}


def _soglie_di_confine(nome: str) -> Tuple[float, ...]:
    """Soglie di confine del parametro: critiche, moderate e range normali."""
    if nome not in _SOGLIE_CONFINE:
        valori: List[float] = []
        for tabella in (SOGLIE_CRITICHE, SOGLIE_MODERATE):
            for v in tabella[nome].values():
                if v is not None:
                    valori.append(float(v))
        min_n, max_n = RANGE_NORMALI[nome]
        valori.extend([min_n, max_n])
        _SOGLIE_CONFINE[nome] = tuple(sorted(set(valori)))
    return _SOGLIE_CONFINE[nome]


def _in_buffer_zone(campione: Dict[str, float]) -> bool:
    """
    True se un valore del campione cade entro MARGINI_BUFFER da una soglia
    di confine (critica, moderata o limite del range normale).
    """
    for nome in FEATURE_ORDER:
        valore = campione[nome]
        margine = MARGINI_BUFFER[nome]
        for soglia in _soglie_di_confine(nome):
            if abs(valore - soglia) <= margine:
                return True
    return False


def _campiona_profilo(rng: np.random.Generator, nome_profilo: str) -> Dict[str, float]:
    """
    Campiona i 6 parametri dal profilo (gaussiane clip ai range fisiologici)
    e impone il vincolo strutturale sistolica > diastolica.
    """
    profilo = PROFILI[nome_profilo]
    campione: Dict[str, float] = {}
    for nome in FEATURE_ORDER:
        media, dev = profilo[nome]  # type: ignore[misc]
        valore = float(rng.normal(media, dev))
        min_v, max_v = RANGE_FISIOLOGICI[nome]
        campione[nome] = float(np.clip(valore, min_v, max_v))

    sistolica = campione["pressione_sistolica"]
    diastolica = campione["pressione_diastolica"]
    if sistolica - diastolica < DIFF_MIN_SIST_DIAST:
        # Ricampiona la diastolica condizionata alla sistolica, entro il
        # range fisiologico e sotto il vincolo di differenza minima.
        min_d, _ = RANGE_FISIOLOGICI["pressione_diastolica"]
        limite_sup = sistolica - DIFF_MIN_SIST_DIAST
        campione["pressione_diastolica"] = float(rng.uniform(min_d, limite_sup))
    return campione


def genera_batch(
    seed: int, n: int, buffer_zone: bool = True
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Genera un batch di `n` campioni etichettati dal teacher.

    Args:
        seed: seed per ENTRAMBI i generatori (random e numpy).
        n: numero di campioni VALIDI richiesti.
        buffer_zone: se True, scarta i campioni entro i margini dalle soglie.

    Returns:
        (X, y, scartati): X shape (n, 6) nell'ordine FEATURE_ORDER, y array
        di stringhe ('basso'/'medio'/'alto'), scartati = campioni rifiutati
        (buffer zone o invalidi).
    """
    random.seed(seed)
    rng = np.random.default_rng(seed)

    X = np.empty((n, len(FEATURE_ORDER)), dtype=np.float64)
    # dtype stringa a larghezza fissa (max label: 'errore' = 6 char):
    # caricabile con np.load senza allow_pickle=True.
    y = np.empty(n, dtype="U6")
    scartati = 0
    i = 0
    tentativi = 0

    while i < n:
        tentativi += 1
        if tentativi > n * MAX_TENTATIVI_PER_CAMPIONE:
            raise RuntimeError(
                "genera_batch: troppi scarti (buffer zone o invalidi); "
                f"generati {i}/{n} campioni"
            )
        nome_profilo = random.choices(_NOMI_PROFILI, weights=_PESI, k=1)[0]
        campione = _campiona_profilo(rng, nome_profilo)
        if buffer_zone and _in_buffer_zone(campione):
            scartati += 1
            continue
        classe = label(campione)
        if classe == "errore":
            scartati += 1
            continue
        X[i] = [campione[nome] for nome in FEATURE_ORDER]
        y[i] = classe
        i += 1

    return X, y, scartati