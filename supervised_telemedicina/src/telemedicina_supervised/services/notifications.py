"""
telemedicina_supervised/services/notifications.py — Notifiche per casi
critici (Fase 7).

Canali del progetto didattico (nessuna dipendenza esterna):
  1. database: riga nella tabella `notifiche` (via AnalisiDatabase);
  2. console: stampa su stderr con timestamp.

La notifica viene generata dal AnalysisService (mai dalla CLI): ogni
flusso applicativo futuro (CLI, API, webapp) eredita il comportamento
per costruzione.
"""

import sys

from telemedicina_supervised.database.analisi_db import AnalisiDatabase


def notifica_critico(
    database: AnalisiDatabase,
    timestamp: str,
    classe: str,
    messaggio: str,
) -> None:
    """Registra una notifica per un caso critico (DB + stderr).

    Args:
        database: repository aperto su cui scrivere la notifica.
        timestamp: ISO-8601 UTC della notifica (stessa dell'analisi).
        classe: classe che ha generato l'allerta ('alto' o 'errore').
        messaggio: testo della notifica.
    """
    database.inserisci_notifica(
        {"timestamp": timestamp, "classe": classe, "messaggio": messaggio}
    )
    print(
        f"[NOTIFICA] {timestamp} classe={classe!r} messaggio={messaggio}",
        file=sys.stderr,
    )
