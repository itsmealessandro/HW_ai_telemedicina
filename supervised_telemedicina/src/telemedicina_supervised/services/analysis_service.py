"""
telemedicina_supervised/services/analysis_service.py — Contratto del
servizio di analisi.

Coordina il flusso: validazione -> SupervisedAgent -> esito.

In questa fase NON esiste ancora un database: l'esito viene loggato su
console e registrato in una struttura dati in memoria (`self.storico`).
La persistenza reale (SQLite o equivalente) arriverà nelle fasi
successive senza modificare il contratto di `analizza`.

Nessuna soglia numerica è definita qui: l'unica fonte è
`telemedicina_supervised.safety.safety_rules`.

NOTA ARCHITETTURALE (decisione accettata): questo modulo importa
`CLASSE_ERRORE` e `valida_parametri` direttamente da `safety_rules`,
attraversando il layer `agents`. È una scelta deliberata: `safety_rules`
è lo STRATO DI DOMINIO CONDIVISO del progetto (fonte unica delle soglie),
e il servizio ne ha bisogno per esporre gli errori di validazione
STRUTTURATI (lista) in `ServiceOutcome.errori`, cosa che `AgentOutcome`
(che espone solo `messaggio` testuale) non fornisce. La decisione di
classificazione resta comunque delegata al `SupervisedAgent`; il servizio
non duplica alcuna soglia né logica di classificazione.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from telemedicina_supervised.agents.supervised_agent import (
    AgentOutcome,
    SupervisedAgent,
)
from telemedicina_supervised.safety.safety_rules import CLASSE_ERRORE, valida_parametri


@dataclass
class ServiceOutcome:
    """
    Output del AnalysisService (contratto).

    Attributi:
        classe:             'basso' | 'medio' | 'alto' | 'errore'.
        messaggio_paziente: messaggio leggibile per il paziente.
        allerta_medico:     True se il caso richiede attenzione medica
                            (classe 'alto' o input invalido).
        errori:             lista dei motivi di validazione (vuota se
                            l'input è valido).
    """

    classe: str
    messaggio_paziente: str
    allerta_medico: bool
    errori: List[str] = field(default_factory=list)
    # Esito completo dell'agente (probabilità, metadati, anomalie, pattern):
    # esposto per il display della CLI senza duplicare la logica di analisi.
    esito_agente: Any = None


_MESSAGGI_PAZIENTE: Dict[str, str] = {
    "basso": "I tuoi parametri vitali risultano nella norma. Continua il monitoraggio regolare.",
    "medio": "Sono state rilevate alcune anomalie. Ti consigliamo di contattare il tuo medico e ripetere la misurazione.",
    "alto": "Sono stati rilevati parametri critici. Ti consigliamo di contattare subito un medico o recarti al pronto soccorso.",
}


class AnalysisService:
    """
    Servizio che orchestra validazione -> agente -> esito.

    Baseline Fase 1: l'agente è la versione rule-based. In Fase 5 il
    SupervisedAgent esporrà l'MLP, ma il contratto di `analizza` resta
    invariato e il safety gate mantiene la precedenza assoluta.
    """

    def __init__(self, agent: Any = None):
        self.agent: Any = agent if agent is not None else SupervisedAgent()
        # Struttura dati in memoria al posto del database (fasi successive):
        # ultima analisi + storico delle analisi.
        self.ultima: Dict[str, Any] = {}
        self.storico: List[Dict[str, Any]] = []

    def analizza(self, parametri: Any) -> ServiceOutcome:
        """
        Analizza i parametri vitali e produce l'esito per paziente/medico.

        Args:
            parametri: mapping (dict) o oggetto con i 6 attributi
                       ufficiali (es. VitalParameters).

        Returns:
            ServiceOutcome secondo il contratto.
        """
        # 1. Validazione (fonte unica: safety_rules) per gli errori strutturati.
        valido, errori = valida_parametri(parametri)

        # 2. Coordinamento all'agente (contratto definitivo).
        outcome: AgentOutcome = self.agent.predict(parametri)

        # 3. Composizione dell'esito per paziente/medico.
        if outcome.errore:
            classe = CLASSE_ERRORE
            messaggio_paziente = (
                "Le misurazioni non risultano valide: ripetere la rilevazione dei parametri."
            )
            allerta_medico = True
        else:
            classe = outcome.classe
            messaggio_paziente = _MESSAGGI_PAZIENTE.get(
                classe, "Analisi completata."
            )
            # Allerta medico: classe 'alto' (le regole hanno precedenza
            # assoluta); per 'basso'/'medio' nessuna allerta.
            allerta_medico = classe == "alto"

        risultato = ServiceOutcome(
            classe=classe,
            messaggio_paziente=messaggio_paziente,
            allerta_medico=allerta_medico,
            errori=list(errori),
            esito_agente=outcome,
        )

        # 4. Registrazione in memoria (database in fasi successive).
        record = {
            "classe": classe,
            "allerta_medico": allerta_medico,
            "messaggio_paziente": messaggio_paziente,
            "errori": list(errori),
        }
        self.ultima = record
        self.storico.append(record)

        # 5. Log su console (baseline: nessun database reale).
        print(
            f"[AnalysisService] classe={classe!r} allerta_medico={allerta_medico} "
            f"errori={len(errori)}"
        )

        return risultato

    def ottieni_storico(self) -> List[Dict[str, Any]]:
        """Storico in memoria delle analisi (placeholder del database)."""
        return list(self.storico)
