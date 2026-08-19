"""
telemedicina_supervised/database/analisi_db.py — Persistenza SQLite (Fase 7).

Unico strato di persistenza del progetto (solo stdlib: sqlite3, nessuna
dipendenza nuova). Il file del database viene creato automaticamente alla
prima connessione se assente (schema lazy init); thread-safety NON
richiesto (uso single-thread: CLI o un processo per servizio).

Schema (documentato):

    analisi
        id              INTEGER PRIMARY KEY AUTOINCREMENT
        timestamp       TEXT    -- ISO-8601 UTC del momento dell'analisi
        classe          TEXT    -- 'basso' | 'medio' | 'alto' | 'errore'
        allerta_medico  INTEGER -- 0/1: richiede attenzione medica
        errori          TEXT    -- JSON (lista di stringhe; [] se valido)
        probabilita     TEXT    -- JSON (dict classe->float) oppure NULL
                                 -- se l'MLP non è stato usato (gate o
                                 -- fallback)
        metadati        TEXT    -- JSON (modello_usato, classe_mlp,
                                 -- classe_regola, confidenza, fallback,
                                 -- motivo_fallback, override_sicurezza)
        messaggio       TEXT    -- messaggio per il paziente

    notifiche
        id              INTEGER PRIMARY KEY AUTOINCREMENT
        timestamp       TEXT    -- ISO-8601 UTC
        classe          TEXT    -- classe che ha generato l'allerta
        messaggio       TEXT    -- testo della notifica

Nessuna soglia clinica vive qui: l'unica fonte resta
`telemedicina_supervised.safety.safety_rules`.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS analisi (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    classe TEXT NOT NULL,
    allerta_medico INTEGER NOT NULL,
    errori TEXT NOT NULL,
    probabilita TEXT,
    metadati TEXT NOT NULL,
    messaggio TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS notifiche (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    classe TEXT NOT NULL,
    messaggio TEXT NOT NULL
);
"""


class AnalisiDatabase:
    """Accesso SQLite con schema lazy-init e commit per operazione."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._connessione: Optional[sqlite3.Connection] = None

    def _conn(self) -> sqlite3.Connection:
        if self._connessione is None:
            self._connessione = sqlite3.connect(str(self.path))
            self._connessione.executescript(SCHEMA_SQL)
        return self._connessione

    def inserisci_analisi(self, record: Dict[str, Any]) -> int:
        """Inserisce un record di analisi; restituisce l'id assegnato."""
        conn = self._conn()
        cursore = conn.execute(
            """
            INSERT INTO analisi
                (timestamp, classe, allerta_medico, errori, probabilita,
                 metadati, messaggio)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["timestamp"],
                record["classe"],
                int(bool(record["allerta_medico"])),
                json.dumps(record["errori"], ensure_ascii=False),
                (
                    json.dumps(record["probabilita"], ensure_ascii=False)
                    if record.get("probabilita") is not None
                    else None
                ),
                json.dumps(record["metadati"], ensure_ascii=False),
                record["messaggio"],
            ),
        )
        conn.commit()
        return self._ultimo_id(cursore)

    def _ultimo_id(self, cursore: sqlite3.Cursor) -> int:
        ultimo: Optional[int] = cursore.lastrowid
        if ultimo is None:  # non raggiungibile dopo un INSERT committato
            raise RuntimeError("inserimento riuscito senza id assegnato")
        return ultimo

    def inserisci_notifica(self, record: Dict[str, Any]) -> int:
        """Inserisce una notifica per caso critico; restituisce l'id."""
        conn = self._conn()
        cursore = conn.execute(
            "INSERT INTO notifiche (timestamp, classe, messaggio) VALUES (?, ?, ?)",
            (record["timestamp"], record["classe"], record["messaggio"]),
        )
        conn.commit()
        return self._ultimo_id(cursore)

    def leggi_analisi(self, limite: int = 1000) -> List[Dict[str, Any]]:
        """Record di analisi più recenti, con i campi JSON decodificati."""
        conn = self._conn()
        righe = conn.execute(
            "SELECT id, timestamp, classe, allerta_medico, errori, probabilita,"
            " metadati, messaggio FROM analisi ORDER BY id DESC LIMIT ?",
            (int(limite),),
        ).fetchall()
        return [
            {
                "id": riga[0],
                "timestamp": riga[1],
                "classe": riga[2],
                "allerta_medico": bool(riga[3]),
                "errori": json.loads(riga[4]),
                "probabilita": json.loads(riga[5]) if riga[5] is not None else None,
                "metadati": json.loads(riga[6]),
                "messaggio": riga[7],
            }
            for riga in righe
        ]

    def leggi_notifiche(self, limite: int = 1000) -> List[Dict[str, Any]]:
        """Notifiche più recenti (id, timestamp, classe, messaggio)."""
        conn = self._conn()
        righe = conn.execute(
            "SELECT id, timestamp, classe, messaggio FROM notifiche"
            " ORDER BY id DESC LIMIT ?",
            (int(limite),),
        ).fetchall()
        return [
            {"id": riga[0], "timestamp": riga[1], "classe": riga[2], "messaggio": riga[3]}
            for riga in righe
        ]

    def chiudi(self) -> None:
        """Chiude la connessione (best effort; idempotente)."""
        if self._connessione is not None:
            self._connessione.close()
            self._connessione = None
