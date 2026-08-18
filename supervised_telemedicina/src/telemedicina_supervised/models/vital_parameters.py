"""
telemedicina_supervised/models/vital_parameters.py — Modello dati dei
parametri vitali (nuovo progetto supervised).

Modello dati indipendente dal legacy: una dataclass con i 6 campi
nell'ordine ufficiale fissato dal contratto (FEATURE_ORDER in
safety_rules.py). Ogni dato strutturato (validazione inclusa) vive in
`safety_rules`: qui NON ci sono soglie duplicate, c'è solo il riuso.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from telemedicina_supervised.safety.safety_rules import (
    FEATURE_ORDER,
    valida_parametri,
)


@dataclass
class VitalParameters:
    """
    Parametri vitali di un paziente.

    Campi nell'ordine ufficiale del contratto:
      1. pressione_sistolica   (mmHg)
      2. pressione_diastolica  (mmHg)
      3. frequenza_cardiaca    (bpm)
      4. temperatura           (°C)
      5. saturazione_ossigeno  (%)
      6. glicemia              (mg/dL)

    Il costruttore SEGNALA gli input fisiologicamente invalidi invece di
    scartarli in silenzio: dopo la costruzione gli attributi `valido` e
    `motivi` espongono l'esito della validazione delegata a
    `safety_rules.valida_parametri` (vincolo sistolica > diastolica
    incluso). Il flusso 'errore' dell'agente/servizio si basa proprio su
    questa segnalazione. In Fase 1 il costruttore non lancia eccezioni:
    un input invalido deve poter essere ANALIZZATO (e rispondere
    'errore'), non solo rifiutato.
    """

    pressione_sistolica: float
    pressione_diastolica: float
    frequenza_cardiaca: float
    temperatura: float
    saturazione_ossigeno: float
    glicemia: float

    # Campi non ordinari: esito della validazione, calcolati in
    # __post_init__ delegando a safety_rules.
    valido: bool = field(init=False)
    motivi: List[str] = field(init=False)

    def __post_init__(self) -> None:
        self.valido, self.motivi = valida_parametri(self)

    # ------------------------------------------------------------------
    # Costruzione alternativa
    # ------------------------------------------------------------------
    @classmethod
    def da_dict(cls, valori: Dict[str, float]) -> "VitalParameters":
        """
        Costruisce l'istanza da un mapping con le chiavi ufficiali
        (ordine del contratto), accettando anche chiavi non ufficiali
        purché le 6 richieste siano presenti.
        """
        return cls(**{nome: valori[nome] for nome in FEATURE_ORDER})

    # ------------------------------------------------------------------
    # Accesso ai valori
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, float]:
        """Dizionario con le chiavi ufficiali nell'ordine del contratto."""
        return {nome: getattr(self, nome) for nome in FEATURE_ORDER}

    def to_vector(self) -> np.ndarray:
        """
        Vettore numpy delle 6 feature nell'ORDINE UFFICIALE del
        contratto: [pressione_sistolica, pressione_diastolica,
        frequenza_cardiaca, temperatura, saturazione_ossigeno, glicemia].

        Questo è l'input dell'MLP in Fase 3+; l'ordine qui è parte del
        contratto e viene verificato dai test.
        """
        return np.array([getattr(self, nome) for nome in FEATURE_ORDER], dtype=float)

    def valida_parametri(self) -> Tuple[bool, List[str]]:
        """
        Riuso esplicito della validazione di safety_rules (fonte unica).
        Include il vincolo pressione_sistolica > pressione_diastolica.
        """
        return valida_parametri(self)

    def __str__(self) -> str:
        return (
            f"VitalParameters(sistolica={self.pressione_sistolica}, "
            f"diastolica={self.pressione_diastolica}, "
            f"fc={self.frequenza_cardiaca}, "
            f"temperatura={self.temperatura}, "
            f"spo2={self.saturazione_ossigeno}, "
            f"glicemia={self.glicemia})"
        )
