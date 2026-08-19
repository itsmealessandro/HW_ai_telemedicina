"""
telemedicina_supervised/services/analysis_service.py — Contratto del
servizio di analisi.

Coordina il flusso: validazione -> SupervisedAgent -> esito.

Dalla Fase 7 l'esito viene PERSISTITO in SQLite (`AnalisiDatabase`, path
configurabile in `config.DB_PATH` o via costruttore) e i casi critici
(classe 'alto' o input invalido) generano una notifica (DB + stderr).
Il contratto pubblico di `analizza` non cambia: la persistenza è un
comportamento aggiuntivo dietro la stessa API. Se il database non è
scrivibile l'analisi fallisce RUMOROSAMENTE (ValueError) per non perdere
record clinici.

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

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from telemedicina_supervised.agents.supervised_agent import (
    AgentOutcome,
    SupervisedAgent,
)
from telemedicina_supervised.config import DB_PATH
from telemedicina_supervised.database.analisi_db import AnalisiDatabase
from telemedicina_supervised.safety.safety_rules import CLASSE_ERRORE, valida_parametri
from telemedicina_supervised.services.notifications import notifica_critico


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
    Servizio che orchestra validazione -> agente -> esito -> persistenza.

    Dalla Fase 7 ogni analisi viene registrata su SQLite e i casi critici
    generano una notifica (DB + stderr). Il contratto di `analizza` resta
    invariato e il safety gate mantiene la precedenza assoluta.
    """

    def __init__(self, agent: Any = None, db_path: Optional[Path] = None):
        self.agent: Any = agent if agent is not None else SupervisedAgent()
        # Path del database SQLite: esplicito (test/CLI) o da config.
        self._db_path: Path = Path(db_path) if db_path is not None else DB_PATH
        self._db: Optional[AnalisiDatabase] = None
        # Struttura dati in memoria (compatibilità): ultima analisi +
        # storico. La fonte persistente è il database SQLite.
        self.ultima: Dict[str, Any] = {}
        self.storico: List[Dict[str, Any]] = []

    def _ottieni_db(self) -> AnalisiDatabase:
        """Connessione lazy: schema creato alla prima analisi."""
        if self._db is None:
            self._db = AnalisiDatabase(self._db_path)
        return self._db

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

        # 4. Registrazione in memoria (compatibilità con i consumatori
        #    esistenti; la fonte persistente è il database SQLite).
        record = {
            "classe": classe,
            "allerta_medico": allerta_medico,
            "messaggio_paziente": messaggio_paziente,
            "errori": list(errori),
        }
        self.ultima = record
        self.storico.append(record)

        # 5. Persistenza SQLite (Fase 7). Un unico timestamp UTC per
        #    analisi e notifica. Se il database non è scrivibile il
        #    servizio FALLISCE RUMOROSAMENTE: un record clinico non va
        #    mai perso in silenzio.
        timestamp = datetime.now(timezone.utc).isoformat()
        record_db = {
            "timestamp": timestamp,
            "classe": classe,
            "allerta_medico": allerta_medico,
            "errori": list(errori),
            "probabilita": outcome.probabilita,
            "metadati": outcome.metadati,
            "messaggio": messaggio_paziente,
        }
        try:
            database = self._ottieni_db()
            database.inserisci_analisi(record_db)
            if allerta_medico:
                notifica_critico(database, timestamp, classe, messaggio_paziente)
        except (sqlite3.Error, OSError) as exc:
            raise ValueError(
                f"database non scrivibile ({self._db_path}): {exc}"
            ) from exc

        # 6. Log su console.
        print(
            f"[AnalysisService] classe={classe!r} allerta_medico={allerta_medico} "
            f"errori={len(errori)}"
        )

        return risultato

    def ottieni_storico(self) -> List[Dict[str, Any]]:
        """Storico in memoria delle analisi (compatibilità)."""
        return list(self.storico)
