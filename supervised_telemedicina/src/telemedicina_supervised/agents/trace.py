"""Tracciatura completa della pipeline di analisi (pagina interattiva).

Espone ``trace_analysis``: specchia ``SupervisedAgent.predict`` ma cattura i
valori intermedi (validazione, safety gate rule-based, passaggi interni
dell'MLP, soglia di incertezza) così il frontend può mostrarli passo-passo.

Il risultato finale è quello autoritativo di ``SupervisedAgent.predict`` (nessuna
logica di classificazione duplicata): i campi di ``rete`` servono solo alla
visualizzazione. Non persiste su DB (usa l'agente diretto, non AnalysisService)
per non inquinare il tab "Analisi Live" con input demo.
"""

from typing import Any, Optional

import numpy as np

from telemedicina_supervised.agents.supervised_agent import SupervisedAgent
from telemedicina_supervised.config import SOGLIA_INCERTEZZA
from telemedicina_supervised.safety.safety_rules import (
    CLASSI,
    FEATURE_ORDER,
    RANGE_FISIOLOGICI,
    analizza,
    valida_parametri,
)


def trace_analysis(parametri: Any, agent: Optional[SupervisedAgent] = None) -> dict:
    """Restituisce la traccia completa della pipeline di valutazione.

    Args:
        parametri: mapping (dict) o oggetto con i 6 attributi ufficiali.
        agent: ``SupervisedAgent`` opzionale (riusato per cache del modello).

    Returns:
        dict con le chiavi: ``input``, ``validazione``, ``safety_gate``,
        ``rete``, ``soglia_incertezza``, ``risposta`` e, se il flusso si
        interrompe prima, ``stopped_at`` (``"validazione"`` | ``"gate"``).
    """
    agent = agent or SupervisedAgent()

    valido, errori = valida_parametri(parametri)
    trace = {
        "input": {k: parametri.get(k) for k in FEATURE_ORDER},
        "validazione": {
            "valido": valido,
            "errori": list(errori),
            "range_fisiologici": {
                k: list(RANGE_FISIOLOGICI[k]) for k in FEATURE_ORDER
            },
        },
    }

    if not valido:
        trace["stopped_at"] = "validazione"
        trace["risposta"] = {
            "classe": "errore",
            "messaggio": "Parametri non validi: " + "; ".join(errori),
            "allerta_medico": True,
            "percorso": "errore",
        }
        return trace

    dettagli = analizza(parametri)
    trace["safety_gate"] = {
        "classe_regola": dettagli["classe"],
        "anomalie": dettagli["anomalie"],
        "pattern": dettagli["pattern"],
    }
    if dettagli["classe"] == "alto":
        trace["stopped_at"] = "gate"
        outcome = agent.predict(parametri)
        trace["risposta"] = _risposta(outcome, "gate")
        return trace

    # Non critico: calcola i valori interni della rete per la visualizzazione.
    modello, _motivo = agent._carica()
    rete = None
    if modello is not None and modello.scaler is not None:
        valori = [parametri[n] for n in FEATURE_ORDER]
        vettore = np.asarray([valori], dtype=float)
        normalizzato = modello.scaler.transform(vettore)
        fwd = modello.forward(normalizzato)
        probs = fwd["probs"][0]
        rete = {
            "input_normalizzato": [float(x) for x in normalizzato[0]],
            "z1": [float(x) for x in fwd["z1"][0]],
            "a1": [float(x) for x in fwd["a1"][0]],
            "z2": [float(x) for x in fwd["z2"][0]],
            "probabilita": {c: float(p) for c, p in zip(CLASSI, probs)},
        }
    trace["rete"] = rete

    outcome = agent.predict(parametri)
    conf = outcome.confidenza
    soglia = SOGLIA_INCERTEZZA
    trace["soglia_incertezza"] = {
        "soglia": soglia,
        "confidenza": conf,
        "superata": None if conf is None else (conf >= soglia),
        "fallback": (conf is not None and conf < soglia),
    }

    if outcome.fallback and outcome.motivo_fallback and (
        outcome.motivo_fallback.lower().startswith("safety gate")
    ):
        percorso = "gate"
    elif outcome.fallback:
        percorso = "fallback"
    elif outcome.modello_usato:
        percorso = "mlp"
    else:
        percorso = "regole"
    trace["risposta"] = _risposta(outcome, percorso)
    return trace


def _risposta(outcome: Any, percorso: str) -> dict:
    """Normalizza l'esito dell'agente per il JSON di risposta."""
    return {
        "classe": outcome.classe,
        "messaggio": outcome.messaggio,
        "allerta_medico": outcome.classe == "alto" or outcome.errore,
        "percorso": percorso,
        "probabilita": outcome.probabilita,
        "confidenza": outcome.confidenza,
        "classe_mlp": outcome.classe_mlp,
        "classe_regola": outcome.classe_regola,
        "override_sicurezza": outcome.override_sicurezza,
        "fallback": outcome.fallback,
        "motivo_fallback": outcome.motivo_fallback,
    }
