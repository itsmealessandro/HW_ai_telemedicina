#!/usr/bin/env python3
"""tools/genera_dashboard.py — Dashboard HTML statica del sistema supervised (Fase V4: completa).

Genera un file HTML autonomo (CSS e JS inline, nessun CDN, nessuna richiesta
di rete) con:
  - vista "1. Analisi Live": tabella delle analisi da ``analisi.db`` con
    badge distintivi (MLP / GATE / FALLBACK / NOTIFICA) e pulsante
    "Aggiorna" (funziona solo con --serve, via GET /api/analisi);
  - vista "2. Flusso decisionale": diagramma verticale del caso selezionato
    (click su una riga della Live), con i valori reali dai metadati;
  - vista "3. Teacher vs MLP": banner metriche, matrice di confusione,
    scatter degli errori e regioni di decisione affiancate (figure
    matplotlib incorporate come base64, da ``report.json`` e dati congelati);
  - vista "4. Training": curve loss, confronto grid, gradient check e
    architettura dell'MLP;
  - vista "5. Incertezza": istogramma della confidenza softmax (soglia da
    config), contatori di sistema (casi critici, override, notifiche) e
    tabella dei mancati dell'MLP ricalcolata sul test congelato;
  - vista "6. Riproducibilità": seed, split e hash da ``metadati.json``;
  - banner didattico obbligatorio e 6 tab tutti implementati.

matplotlib è importato LAZY (via ``tools/_figure.py``): senza la dipendenza
dev le figure diventano segnaposto con messaggio, mai un crash.

Uso:
    python tools/genera_dashboard.py                  # build statico
    python tools/genera_dashboard.py --out OUT        # output custom
    python tools/genera_dashboard.py --db-path DB     # database custom
    python tools/genera_dashboard.py --metadati-path M  # metadati custom
    python tools/genera_dashboard.py --report-path R  # report custom
    python tools/genera_dashboard.py --test-path D    # dir con X_test/y_test
    python tools/genera_dashboard.py --serve          # build + http.server
                                                     # (+ GET /api/analisi)

Il tool è eseguibile da qualsiasi CWD: i percorsi di default (DB e output)
sono derivati da ``__file__``, non dalla directory corrente.
"""

import argparse
import html
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Percorso assoluto a src/ (il tool vive in tools/, un livello sotto la root).
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))
# tools/ stesso: per l'import lazy di _figure (matplotlib NON a livello di
# modulo, così il tool gira anche senza la dipendenza dev).
TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

from telemedicina_supervised.config import (  # noqa: E402
    CLASSI,
    DATA_PROCESSED_DIR,
    DB_PATH,
    GRID,
    MODEL_ARTIFACT_PATH,
    PAZIENZA,
    REPORT_PATH,
    SOGLIA_INCERTEZZA,
)
from telemedicina_supervised.database.analisi_db import AnalisiDatabase  # noqa: E402
from telemedicina_supervised.agents.supervised_agent import SupervisedAgent  # noqa: E402
from telemedicina_supervised.agents.trace import trace_analysis  # noqa: E402
from telemedicina_supervised.safety.safety_rules import (  # noqa: E402
    FEATURE_ORDER,
    RANGE_FISIOLOGICI,
    RANGE_NORMALI,
    SOGLIE_CRITICHE,
    SOGLIE_MODERATE,
)

ETICHETTE: Dict[str, str] = {
    "pressione_sistolica": "Pressione sistolica",
    "pressione_diastolica": "Pressione diastolica",
    "frequenza_cardiaca": "Frequenza cardiaca",
    "temperatura": "Temperatura",
    "saturazione_ossigeno": "Saturazione O₂",
    "glicemia": "Glicemia",
}
UNITA: Dict[str, str] = {
    "pressione_sistolica": "mmHg",
    "pressione_diastolica": "mmHg",
    "frequenza_cardiaca": "bpm",
    "temperatura": "°C",
    "saturazione_ossigeno": "%",
    "glicemia": "mg/dL",
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DEFAULT = PROJECT_ROOT / "data" / "dashboard.html"
METADATI_DEFAULT = DATA_PROCESSED_DIR / "metadati.json"
REPORT_DEFAULT = REPORT_PATH
ARTIFACT_DEFAULT = MODEL_ARTIFACT_PATH

BANNER = (
    "Dati sintetici: questa dashboard mostra esempi generati artificialmente, "
    "nessuna pretesa diagnostica."
)
SOTTOTITOLO = (
    "Percorso applicativo: CLI → AnalysisService → SupervisedAgent → "
    "SQLite + notifiche"
)
VINCOLI = ("Runtime: numpy + stdlib", "Grafici: matplotlib (dev, solo tools/)")

TAB = [
    ("live", "1. Analisi Live"),
    ("flusso", "2. Flusso decisionale"),
    ("teacher", "3. Teacher vs MLP"),
    ("training", "4. Training"),
    ("incertezza", "5. Incertezza"),
    ("riproducibilita", "6. Riproducibilità"),
    ("interattiva", "7. Analisi interattiva"),
]

BADGE_LABEL = {
    "mlp": "MLP",
    "gate": "GATE",
    "fallback": "FALLBACK",
    "notifica": "NOTIFICA",
}


# ---------------------------------------------------------------------------
# Logica badge per riga
# ---------------------------------------------------------------------------

def _badge_per_riga(metadati: Dict[str, Any]) -> List[str]:
    """Badge MLP/GATE/FALLBACK per una riga (GATE e FALLBACK esclusivi).

    - MLP      -> modello_usato == True
    - GATE     -> fallback == True e motivo_fallback inizia con "safety gate"
    - FALLBACK -> fallback == True e motivo NON "safety gate" (o assente)
    """
    badge: List[str] = []
    if metadati.get("modello_usato") is True:
        badge.append("mlp")
    if metadati.get("fallback") is True:
        motivo = metadati.get("motivo_fallback")
        if isinstance(motivo, str) and motivo.lower().startswith("safety gate"):
            badge.append("gate")
        else:
            badge.append("fallback")
    return badge


def _percorso(metadati: Dict[str, Any]) -> str:
    """Classifica il percorso decisionale del caso (stessa logica dei badge).

    - "gate"     -> fallback con motivo "safety gate ..." (il gate ha
                    bypassato l'MLP: caso critico o input invalido);
    - "fallback" -> fallback per altra ragione (es. confidenza sotto soglia);
    - "mlp"      -> modello usato senza fallback;
    - "regole"   -> baseline rule-based (nessun modello, nessun fallback).
    """
    if metadati.get("fallback") is True:
        motivo = metadati.get("motivo_fallback")
        if isinstance(motivo, str) and motivo.lower().startswith("safety gate"):
            return "gate"
        return "fallback"
    if metadati.get("modello_usato") is True:
        return "mlp"
    return "regole"


# ---------------------------------------------------------------------------
# Formattazione celle (tutti i campi testuali passano da html.escape)
# ---------------------------------------------------------------------------

def _formatta_probabilita(probabilita: Optional[Dict[str, float]]) -> str:
    """Dict classe->float come 'classe: 0.9900, ...' (None -> '—')."""
    if not probabilita:
        return "—"
    return ", ".join(
        f"{html.escape(str(classe))}: {valore:.4f}"
        for classe, valore in probabilita.items()
    )


def _formatta_errori(errori: Optional[List[str]]) -> str:
    """Lista errori unita con '; ' (vuota/assente -> '—')."""
    if not errori:
        return "—"
    return "; ".join(html.escape(str(e)) for e in errori)


def _formatta_motivo(metadati: Dict[str, Any]) -> str:
    """motivo_fallback come testo (None/vuoto -> '—')."""
    motivo = metadati.get("motivo_fallback")
    if not isinstance(motivo, str) or not motivo:
        return "—"
    return html.escape(motivo)


def _override_sicurezza(metadati: Dict[str, Any]) -> str:
    """override_sicurezza come Sì/No ('—' se assente)."""
    valore = metadati.get("override_sicurezza")
    if valore is None:
        return "—"
    return "Sì" if valore else "No"


# ---------------------------------------------------------------------------
# Righe della tabella Live
# ---------------------------------------------------------------------------

def _riga_html(riga: Dict[str, Any], timestamp_notifiche: set) -> str:
    """Una riga <tr> della tabella Live, con badge e campi escapati."""
    metadati = riga.get("metadati") or {}
    badge = _badge_per_riga(metadati)
    if riga.get("timestamp") in timestamp_notifiche:
        badge.append("notifica")
    badge_html = " ".join(
        f'<span class="badge-{b}">{BADGE_LABEL[b]}</span>' for b in badge
    ) or "—"

    classe = riga.get("classe") or "errore"
    timestamp = html.escape(str(riga.get("timestamp") or ""))
    messaggio = html.escape(str(riga.get("messaggio") or ""))
    return (
        f'<tr data-id="{riga.get("id")}">'
        f"<td>{timestamp}</td>"
        f'<td><span class="classe-{html.escape(classe)}">{html.escape(classe)}</span></td>'
        f"<td>{'Sì' if riga.get('allerta_medico') else 'No'}</td>"
        f"<td>{_formatta_errori(riga.get('errori'))}</td>"
        f"<td>{_formatta_probabilita(riga.get('probabilita'))}</td>"
        f"<td>{messaggio}</td>"
        f"<td>{_formatta_motivo(metadati)}</td>"
        f"<td>{_override_sicurezza(metadati)}</td>"
        f"<td>{badge_html}</td>"
        "</tr>"
    )


def _record_json(riga: Dict[str, Any], timestamp_notifiche: set) -> Dict[str, Any]:
    """Record per il JSON incorporato e per /api/analisi.

    Aggiunge il percorso decisionale (``percorso``) e la riga HTML
    pre-renderizzata (``riga_html``): la logica dei badge e della tabella
    resta server-side, il JS non la duplica.
    """
    record = dict(riga)
    record["percorso"] = _percorso(riga.get("metadati") or {})
    record["riga_html"] = _riga_html(riga, timestamp_notifiche)
    return record


# ---------------------------------------------------------------------------
# Lettura dati con degrado grazioso (mai eccezioni verso l'utente)
# ---------------------------------------------------------------------------

def _leggi_dati(
    db_path: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Optional[str]]:
    """Legge analisi e notifiche; restituisce (analisi, notifiche, avviso).

    - DB inesistente -> avviso esplicito, liste vuote;
    - DB vuoto       -> avviso "Nessuna analisi registrata nel database";
    - errore imprevisto -> stampa su stderr e avviso di degrado.
    """
    if not Path(db_path).is_file():
        return [], [], (
            f"Database non trovato: {db_path}. Eseguire prima "
            "python main.py --run-ml ... oppure --build-ml"
        )
    try:
        db = AnalisiDatabase(db_path, read_only=True)
        try:
            analisi = db.leggi_analisi()
            notifiche = db.leggi_notifiche()
        finally:
            db.chiudi()
    except Exception as exc:
        print(f"Errore durante la lettura del database: {exc}", file=sys.stderr)
        return [], [], f"Errore durante la lettura del database: {exc}"
    if not analisi:
        return [], [], "Nessuna analisi registrata nel database"
    return analisi, notifiche, None


def _leggi_metadati(
    metadati_path: Path,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Legge metadati.json (seed, split, hash); (None, avviso) se assente."""
    if not Path(metadati_path).is_file():
        return None, (
            f"Metadati non trovati: {metadati_path}. Eseguire prima "
            "python training/generate_dataset.py"
        )
    try:
        with open(metadati_path, encoding="utf-8") as file:
            return json.load(file), None
    except Exception as exc:
        print(f"Errore durante la lettura dei metadati: {exc}", file=sys.stderr)
        return None, f"Errore durante la lettura dei metadati: {exc}"


def _leggi_report(
    report_path: Path,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Legge report.json (metriche, loss, matrice); (None, avviso) se assente."""
    if not Path(report_path).is_file():
        return None, (
            f"Report non trovato: {report_path}. Eseguire prima "
            "python training/train_model.py --build-ml"
        )
    try:
        with open(report_path, encoding="utf-8") as file:
            return json.load(file), None
    except Exception as exc:
        print(f"Errore durante la lettura del report: {exc}", file=sys.stderr)
        return None, f"Errore durante la lettura del report: {exc}"


# ---------------------------------------------------------------------------
# Shell HTML (un solo file autonomo: CSS e JS inline, nessun link esterno)
# ---------------------------------------------------------------------------

def _css() -> str:
    return """\
* { box-sizing: border-box; }
body {
  font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  margin: 0;
  background: #f5f6f8;
  color: #1c1e21;
}
header {
  background: #0d1b2a;
  color: #e0e1dd;
  padding: 24px 32px;
}
header h1 { margin: 0 0 6px; font-size: 24px; }
.sottotitolo { margin: 0 0 12px; font-size: 14px; color: #9fb3c8; }
.vincoli { margin-bottom: 12px; }
.vincolo {
  display: inline-block;
  background: #1b3a5b;
  color: #cfe0f0;
  border-radius: 12px;
  padding: 3px 10px;
  font-size: 12px;
  margin-right: 8px;
}
.banner {
  background: #fff3cd;
  border: 1px solid #ffe08a;
  color: #664d03;
  border-radius: 6px;
  padding: 10px 14px;
  font-size: 14px;
  font-weight: 600;
}
.tab-bar {
  display: flex;
  flex-wrap: wrap;
  background: #1b3a5b;
  padding: 0 16px;
}
.tab {
  background: transparent;
  border: none;
  color: #cfe0f0;
  padding: 12px 16px;
  font-size: 14px;
  cursor: pointer;
  border-bottom: 3px solid transparent;
}
.tab:hover { background: rgba(255, 255, 255, 0.08); }
.tab.active {
  color: #ffffff;
  border-bottom-color: #4fc3f7;
  font-weight: 600;
}
main { padding: 24px 32px; }
.tab-panel { display: none; }
.tab-panel.active { display: block; }
.avviso {
  background: #fdecea;
  border: 1px solid #f5c6cb;
  color: #842029;
  border-radius: 6px;
  padding: 10px 14px;
  margin-bottom: 16px;
  font-size: 14px;
}
.tabella-live {
  width: 100%;
  border-collapse: collapse;
  background: #ffffff;
  border-radius: 6px;
  overflow: hidden;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.12);
}
.tabella-live th, .tabella-live td {
  border: 1px solid #e3e6ea;
  padding: 8px 10px;
  font-size: 13px;
  text-align: left;
  vertical-align: top;
}
.tabella-live th {
  background: #eef1f5;
  font-weight: 600;
  white-space: nowrap;
}
.classe-basso { color: #2e7d32; font-weight: 600; }
.classe-medio { color: #ef6c00; font-weight: 600; }
.classe-alto { color: #c62828; font-weight: 600; }
.classe-errore { color: #757575; font-weight: 600; }
.badge-mlp {
  display: inline-block;
  background: #1565c0;
  color: #ffffff;
  border-radius: 10px;
  padding: 2px 8px;
  font-size: 11px;
  font-weight: 700;
  margin-right: 4px;
}
.badge-gate {
  display: inline-block;
  background: #c62828;
  color: #ffffff;
  border-radius: 10px;
  padding: 2px 8px;
  font-size: 11px;
  font-weight: 700;
  margin-right: 4px;
}
.badge-fallback {
  display: inline-block;
  background: #ef6c00;
  color: #ffffff;
  border-radius: 10px;
  padding: 2px 8px;
  font-size: 11px;
  font-weight: 700;
  margin-right: 4px;
}
.badge-notifica {
  display: inline-block;
  background: #6a1b9a;
  color: #ffffff;
  border-radius: 10px;
  padding: 2px 8px;
  font-size: 11px;
  font-weight: 700;
  margin-right: 4px;
}
.segnaposto { color: #6c757d; font-style: italic; }
.toolbar-live { margin-bottom: 12px; }
.toolbar-live button {
  background: #1b3a5b;
  color: #ffffff;
  border: none;
  border-radius: 4px;
  padding: 6px 14px;
  font-size: 13px;
  cursor: pointer;
}
.toolbar-live button:hover { background: #2a4d75; }
#stato-aggiorna { margin-left: 10px; font-size: 12px; color: #6c757d; }
.tabella-live tbody tr[data-id] { cursor: pointer; }
.tabella-live tbody tr[data-id]:hover { background: #f0f6fc; }
.card {
  background: #ffffff;
  border-radius: 6px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.12);
  padding: 20px 24px;
  max-width: 720px;
}
.card h2 { margin-top: 0; }
.card h3 { margin: 18px 0 8px; font-size: 14px; color: #1b3a5b; }
.badge-deterministico {
  display: inline-block;
  background: #2e7d32;
  color: #ffffff;
  border-radius: 10px;
  padding: 3px 10px;
  font-size: 12px;
  font-weight: 700;
}
.flusso { max-width: 640px; }
.nodo {
  border: 1px solid #e3e6ea;
  border-left: 4px solid #9fb3c8;
  border-radius: 6px;
  background: #ffffff;
  padding: 10px 14px;
  margin-bottom: 8px;
}
.nodo.non-raggiunto { opacity: 0.45; }
.nodo.stato-gate { border-left-color: #c62828; }
.nodo.stato-fallback { border-left-color: #ef6c00; }
.nodo.stato-notifica { border-left-color: #6a1b9a; }
.nodo.stato-ok { border-left-color: #2e7d32; }
.nodo.stato-errore { border-left-color: #757575; }
.nodo-titolo { font-weight: 700; font-size: 13px; color: #1b3a5b; }
.nodo-esito { font-size: 13px; margin-top: 4px; }
.barre-prob { margin-top: 6px; }
.barra-riga { display: flex; align-items: center; margin-bottom: 3px; font-size: 12px; }
.barra-etichetta { width: 52px; }
.barra-sfondo {
  flex: 1;
  background: #eef1f5;
  border-radius: 3px;
  height: 12px;
  margin: 0 8px;
  overflow: hidden;
}
.barra-piena { height: 100%; background: #1b3a5b; border-radius: 3px; }
.barra-valore { width: 64px; text-align: right; color: #6c757d; }
.scala-confidenza { margin-top: 8px; }
.scala-barra {
  position: relative;
  height: 14px;
  background: linear-gradient(to right, #ef6c00, #f5c6cb 60%, #2e7d32);
  border-radius: 3px;
}
.scala-punto {
  position: absolute;
  top: -3px;
  width: 4px;
  height: 20px;
  background: #0d1b2a;
  border-radius: 2px;
}
.scala-soglia {
  position: absolute;
  top: -18px;
  transform: translateX(-50%);
  font-size: 11px;
  color: #664d03;
  background: #fff3cd;
  border: 1px solid #ffe08a;
  border-radius: 3px;
  padding: 0 4px;
  white-space: nowrap;
}
footer {
  padding: 16px 32px;
  font-size: 12px;
  color: #6c757d;
}
.metriche-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin: 14px 0;
}
.metriche-card {
  flex: 1 1 180px;
  background: #f0f4f8;
  border: 1px solid #dde5ee;
  border-radius: 8px;
  padding: 12px 14px;
  text-align: center;
}
.metriche-valore {
  display: block;
  font-size: 22px;
  font-weight: 700;
  color: #0d1b2a;
}
.metriche-etichetta {
  display: block;
  font-size: 12px;
  color: #6c757d;
  margin-top: 4px;
}
.nota {
  font-size: 13px;
  color: #6c757d;
  font-style: italic;
  margin: 10px 0 0;
}
.figura { margin: 14px 0; text-align: center; }
.figura-img {
  max-width: 100%;
  height: auto;
  border: 1px solid #e3e6ea;
  border-radius: 6px;
  background: #ffffff;
}
.figura figcaption {
  font-size: 12px;
  color: #6c757d;
  margin-top: 6px;
}
/* --- Pagina 7: Analisi interattiva --- */
.form-campioni {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 12px;
  margin-bottom: 14px;
}
.campo label {
  display: block;
  font-size: 12px;
  color: #1b3a5b;
  font-weight: 600;
  margin-bottom: 4px;
}
.campo input {
  width: 100%;
  padding: 8px 10px;
  border: 1px solid #cdd6e0;
  border-radius: 6px;
  font-size: 14px;
  box-sizing: border-box;
}
.campo .hint {
  display: block;
  font-size: 11px;
  color: #6c757d;
  margin-top: 3px;
}
.toolbar-interattiva {
  display: flex;
  gap: 10px;
  align-items: center;
  margin-bottom: 16px;
}
.toolbar-interattiva button {
  background: #1b3a5b;
  color: #ffffff;
  border: none;
  border-radius: 4px;
  padding: 8px 18px;
  font-size: 14px;
  cursor: pointer;
}
.toolbar-interattiva button:hover { background: #2a4d75; }
.toolbar-interattiva button.secondario { background: #6c757d; }
.toolbar-interattiva button.secondario:hover { background: #495057; }
#stato-valuta { font-size: 12px; color: #6c757d; }
.check-gate {
  font-size: 12px;
  color: #495057;
  display: inline-flex;
  align-items: center;
  gap: 5px;
}
.step {
  opacity: 0;
  transform: translateY(8px);
  transition: opacity .45s ease, transform .45s ease;
  margin-bottom: 10px;
}
.step.show { opacity: 1; transform: none; }
.badge-errore {
  display: inline-block;
  background: #495057;
  color: #ffffff;
  border-radius: 10px;
  padding: 2px 8px;
  font-size: 11px;
  font-weight: 700;
  margin-right: 4px;
}
.vettore { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0; }
.vcell { width: 66px; text-align: center; font-size: 11px; color: #1c1e21; }
.vbar-sfondo {
  background: #eef1f5;
  border-radius: 3px;
  height: 46px;
  position: relative;
  overflow: hidden;
}
.vbar-piena {
  position: absolute;
  left: 0; right: 0; bottom: 0;
  background: #1b3a5b;
  border-radius: 3px;
}
.vbar-piena.neg { background: #c62828; }
.vval { font-size: 10px; color: #495057; margin-top: 2px; }
.vetichetta { font-size: 10px; color: #6c757d; }
.anomalia {
  border-left: 4px solid #ef6c00;
  background: #ffffff;
  padding: 6px 10px;
  border-radius: 6px;
  margin-bottom: 6px;
  font-size: 13px;
}
.anomalia.critica { border-left-color: #c62828; }
.pattern-item {
  border-left: 4px solid #6a1b9a;
  background: #ffffff;
  padding: 6px 10px;
  border-radius: 6px;
  margin-bottom: 6px;
  font-size: 13px;
}
.spiega {
  font-size: 13px;
  color: #37474f;
  margin: 6px 0 8px;
  line-height: 1.5;
}
.spiega em { color: #1b3a5b; font-style: italic; }
.barra-range { margin: 8px 0 12px; }
.br-intestazione {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  font-size: 12px;
  color: #1b3a5b;
}
.br-track {
  position: relative;
  height: 10px;
  border-radius: 5px;
  background: #e3e6ea;
  margin-top: 14px;
}
.br-seg { position: absolute; top: 0; bottom: 0; }
.br-marker {
  position: absolute;
  top: -3px;
  width: 4px;
  height: 16px;
  background: #0d1b2a;
  border-radius: 2px;
  box-shadow: 0 0 0 2px #ffffff;
}
.br-estremi {
  display: flex;
  justify-content: space-between;
  font-size: 10px;
  color: #6c757d;
  margin-top: 3px;
}
.legenda {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  font-size: 11px;
  color: #495057;
  margin: 10px 0 2px;
}
.legenda .chip {
  display: inline-block;
  width: 10px;
  height: 10px;
  border-radius: 2px;
  margin-right: 4px;
  vertical-align: -1px;
}
.rete-figura { text-align: center; margin: 10px 0; }
.rete-figura svg { max-width: 100%; height: auto; }
.rete-figcaption { font-size: 11px; color: #6c757d; margin-top: 4px; }
.nota-attivi {
  font-size: 12px;
  color: #1b3a5b;
  font-weight: 600;
  margin: 6px 0;
}
"""


def _js() -> str:
    return """\
(function () {
  var SOGLIA = null;
  var DATI = { record: [] };
  var scriptDati = document.getElementById('dati-analisi');
  if (scriptDati) {
    try {
      DATI = JSON.parse(scriptDati.textContent);
      SOGLIA = DATI.soglia_incertezza;
    } catch (e) { /* JSON assente o corrotto: dashboard statica */ }
  }

  function esc(testo) {
    return String(testo === null || testo === undefined ? '' : testo)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  function attivaTab(chiave) {
    document.querySelectorAll('.tab').forEach(function (b) {
      b.classList.toggle('active', b.getAttribute('data-tab') === chiave);
    });
    document.querySelectorAll('.tab-panel').forEach(function (p) {
      p.classList.toggle('active', p.getAttribute('data-tab') === chiave);
    });
  }

  var barra = document.querySelector('.tab-bar');
  if (barra) {
    barra.addEventListener('click', function (evento) {
      var bottone = evento.target.closest('.tab');
      if (!bottone) { return; }
      attivaTab(bottone.getAttribute('data-tab'));
    });
  }

  function barreProbabilita(probabilita) {
    if (!probabilita) { return ''; }
    var ordine = ['basso', 'medio', 'alto'];
    var html = '<div class="barre-prob">';
    ordine.forEach(function (classe) {
      var valore = probabilita[classe];
      if (valore === undefined) { return; }
      var larghezza = Math.max(0, Math.min(100, valore * 100));
      html += '<div class="barra-riga"><span class="barra-etichetta">' + classe + '</span>'
        + '<div class="barra-sfondo"><div class="barra-piena" style="width:' + larghezza.toFixed(1) + '%"></div></div>'
        + '<span class="barra-valore">' + valore.toFixed(4) + '</span></div>';
    });
    return html + '</div>';
  }

  function scalaConfidenza(confidenza) {
    if (confidenza === null || confidenza === undefined || SOGLIA === null) { return ''; }
    var pos = Math.max(0, Math.min(100, confidenza * 100));
    var posSoglia = Math.max(0, Math.min(100, SOGLIA * 100));
    return '<div class="scala-confidenza"><div class="scala-barra">'
      + '<div class="scala-punto" style="left:' + pos.toFixed(1) + '%"></div>'
      + '<div class="scala-soglia" style="left:' + posSoglia.toFixed(1) + '%">soglia ' + SOGLIA + '</div>'
      + '</div></div>';
  }

  function nodo(titolo, esito, raggiunto, stato) {
    var classe = raggiunto ? 'nodo raggiunto' : 'nodo non-raggiunto';
    if (stato) { classe += ' ' + stato; }
    return '<div class="' + classe + '"><div class="nodo-titolo">' + titolo + '</div>'
      + '<div class="nodo-esito">' + esito + '</div></div>';
  }

  function renderFlusso(record) {
    var contenitore = document.getElementById('flusso-decisionale');
    if (!contenitore) { return; }
    if (!record) {
      contenitore.innerHTML = '<p class="segnaposto">Seleziona una riga nella tabella "1. Analisi Live".</p>';
      return;
    }
    var m = record.metadati || {};
    var percorso = record.percorso || 'regole';
    var errori = record.errori || [];
    var nodi = [];
    nodi.push(nodo('1. Input', 'Caso #' + record.id + ' — ' + esc(record.timestamp), true));
    nodi.push(nodo('2. Validazione',
      errori.length ? 'Invalido: ' + esc(errori.join('; ')) : 'Valido',
      true, errori.length ? 'stato-errore' : 'stato-ok'));
    var gate = percorso === 'gate';
    nodi.push(nodo('3. Safety gate',
      gate ? 'Critico → GATE: ' + esc(m.motivo_fallback || '') : 'Non critico',
      true, gate ? 'stato-gate' : 'stato-ok'));
    var mlpRaggiunto = percorso === 'mlp' || percorso === 'fallback';
    nodi.push(nodo('4. MLP softmax',
      mlpRaggiunto ? 'Classe MLP: ' + esc(m.classe_mlp || '—') + barreProbabilita(record.probabilita)
        : 'Non raggiunto (gate o regole)',
      mlpRaggiunto));
    var sogliaRaggiunta = mlpRaggiunto;
    var esitoSoglia = '';
    var conf = m.confidenza;
    if (sogliaRaggiunta) {
      if (conf === null || conf === undefined) {
        esitoSoglia = 'Confidenza non disponibile'
          + (m.motivo_fallback ? ' (' + esc(m.motivo_fallback) + ')' : '');
      } else if (SOGLIA === null) {
        esitoSoglia = 'Confidenza ' + conf.toFixed(4) + ' (soglia non disponibile)';
      } else if (conf >= SOGLIA) {
        esitoSoglia = 'Confidenza ' + conf.toFixed(4) + ' ≥ soglia ' + SOGLIA + ' → MLP';
      } else {
        esitoSoglia = 'Confidenza ' + conf.toFixed(4) + ' < soglia ' + SOGLIA + ' → fallback rule-based';
      }
    }
    nodi.push(nodo('5. Soglia di incertezza',
      sogliaRaggiunta ? esitoSoglia + scalaConfidenza(conf) : 'Non raggiunto',
      sogliaRaggiunta,
      sogliaRaggiunta && conf !== null && conf !== undefined && conf < SOGLIA ? 'stato-fallback' : ''));
    var maxRaggiunto = percorso === 'mlp';
    nodi.push(nodo('6. Regola max(rule, MLP)',
      maxRaggiunto ? 'Override sicurezza: ' + (m.override_sicurezza ? 'Sì' : 'No') : 'Non raggiunto',
      maxRaggiunto));
    nodi.push(nodo('7. Persistenza DB + notifica',
      record.allerta_medico ? 'Allerta medica: Sì → notifica inviata' : 'Allerta medica: No',
      true, record.allerta_medico ? 'stato-notifica' : ''));
    contenitore.innerHTML = nodi.join('');
  }

  function renderTabella(record) {
    var tbody = document.querySelector('.tabella-live tbody');
    if (!tbody) { return; }
    tbody.innerHTML = record.map(function (r) { return r.riga_html; }).join('');
  }

  var tbody = document.querySelector('.tabella-live tbody');
  if (tbody) {
    tbody.addEventListener('click', function (evento) {
      var riga = evento.target.closest('tr[data-id]');
      if (!riga) { return; }
      var id = parseInt(riga.getAttribute('data-id'), 10);
      var record = null;
      for (var i = 0; i < DATI.record.length; i++) {
        if (DATI.record[i].id === id) { record = DATI.record[i]; break; }
      }
      if (record) { renderFlusso(record); attivaTab('flusso'); }
    });
  }

  var bottone = document.getElementById('btn-aggiorna');
  var stato = document.getElementById('stato-aggiorna');
  if (bottone) {
    bottone.addEventListener('click', function () {
      fetch('/api/analisi').then(function (risposta) { return risposta.json(); }).then(function (dati) {
        DATI = dati;
        SOGLIA = dati.soglia_incertezza;
        renderTabella(dati.record);
        if (stato) { stato.textContent = 'Aggiornato: ' + new Date().toLocaleTimeString(); }
      }).catch(function () {
        if (stato) { stato.textContent = 'Aggiornamento disponibile solo con --serve'; }
      });
    });
  }
})();
"""

def _js_interattiva() -> str:
    return """\
(function () {
  var FEATURES = [
    ["pressione_sistolica", "Sist.", "Pressione sistolica", "mmHg"],
    ["pressione_diastolica", "Diast.", "Pressione diastolica", "mmHg"],
    ["frequenza_cardiaca", "FC", "Frequenza cardiaca", "bpm"],
    ["temperatura", "Temp.", "Temperatura", "°C"],
    ["saturazione_ossigeno", "SpO2", "Saturazione O₂", "%"],
    ["glicemia", "Glic.", "Glicemia", "mg/dL"]
  ];

  var ESEMPI = [
    { nome: "paziente a basso rischio", valori: { pressione_sistolica: 120, pressione_diastolica: 80, frequenza_cardiaca: 75, temperatura: 36.5, saturazione_ossigeno: 98, glicemia: 95 } },
    { nome: "paziente a medio rischio", valori: { pressione_sistolica: 150, pressione_diastolica: 92, frequenza_cardiaca: 105, temperatura: 37.0, saturazione_ossigeno: 96, glicemia: 110 } },
    { nome: "paziente ad alto rischio", valori: { pressione_sistolica: 185, pressione_diastolica: 115, frequenza_cardiaca: 125, temperatura: 38.8, saturazione_ossigeno: 86, glicemia: 210 } }
  ];
  var indiceEsempio = 0;

  function esc(s) {
    return String(s === null || s === undefined ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;')
      .replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }
  function delay(ms) { return new Promise(function (r) { setTimeout(r, ms); }); }

  function stepNode(html) {
    var d = document.createElement('div');
    d.className = 'step nodo';
    d.innerHTML = html;
    return d;
  }
  function badge(percorso) {
    var map = { mlp: 'MLP', gate: 'GATE', fallback: 'FALLBACK', regole: 'REGOLE', errore: 'ERRORE' };
    var cls = { mlp: 'badge-mlp', gate: 'badge-gate', fallback: 'badge-fallback', regole: 'badge-notifica', errore: 'badge-errore' };
    return '<span class="' + cls[percorso] + '">' + esc(map[percorso] || percorso) + '</span>';
  }
  function vettore(vals, labels) {
    var max = 1e-9;
    vals.forEach(function (v) { max = Math.max(max, Math.abs(v)); });
    var h = '<div class="vettore">';
    vals.forEach(function (v, i) {
      var pct = Math.max(2, Math.min(100, Math.abs(v) / max * 100));
      var neg = v < 0 ? ' neg' : '';
      var lab = labels ? labels[i] : ('#' + (i + 1));
      h += '<div class="vcell"><div class="vbar-sfondo"><div class="vbar-piena' + neg + '" style="height:' + pct.toFixed(1) + '%"></div></div><div class="vval">' + v.toFixed(3) + '</div><div class="vetichetta">' + esc(lab) + '</div></div>';
    });
    return h + '</div>';
  }
  function barreProb(prob) {
    if (!prob) return '';
    var ordine = ['basso', 'medio', 'alto'];
    var h = '<div class="barre-prob">';
    ordine.forEach(function (c) {
      var v = prob[c]; if (v === undefined) return;
      var w = Math.max(0, Math.min(100, v * 100));
      h += '<div class="barra-riga"><span class="barra-etichetta">' + c + '</span><div class="barra-sfondo"><div class="barra-piena" style="width:' + w.toFixed(1) + '%"></div></div><span class="barra-valore">' + v.toFixed(4) + '</span></div>';
    });
    return h + '</div>';
  }
  function scalaConf(conf, soglia) {
    if (conf === null || conf === undefined) return '';
    var pos = Math.max(0, Math.min(100, conf * 100));
    var ps = Math.max(0, Math.min(100, soglia * 100));
    return '<div class="scala-confidenza"><div class="scala-barra"><div class="scala-punto" style="left:' + pos.toFixed(1) + '%"></div><div class="scala-soglia" style="left:' + ps.toFixed(1) + '%">soglia ' + soglia + '</div></div></div>';
  }

  var COLORI_ZONA = { normale: '#2e7d32', moderata: '#ef6c00', critica: '#c62828', fuori: '#757575' };

  function zona(v, s) {
    var fis = s.fisiologico, cri = s.critica, mod = s.moderata;
    if (v < fis[0] || v > fis[1]) return 'fuori';
    if ((cri.alta !== null && v >= cri.alta) || (cri.bassa !== null && v <= cri.bassa)) return 'critica';
    if ((mod.alta !== null && v >= mod.alta) || (mod.bassa !== null && v <= mod.bassa)) return 'moderata';
    return 'normale';
  }

  function barraRange(chiave, valore, s, meta) {
    if (!s) return '';
    var fis = s.fisiologico;
    var val = Number(valore);
    var bs = [fis[0], s.critica.bassa, s.moderata.bassa, s.normale[0], s.normale[1], s.moderata.alta, s.critica.alta, fis[1]]
      .filter(function (x) { return x !== null && x !== undefined; })
      .filter(function (x, i, arr) { return arr.indexOf(x) === i; })
      .sort(function (a, b) { return a - b; });
    var segs = '';
    for (var i = 0; i < bs.length - 1; i++) {
      var mid = (bs[i] + bs[i + 1]) / 2;
      var left = (bs[i] - fis[0]) / (fis[1] - fis[0]) * 100;
      var larg = (bs[i + 1] - bs[i]) / (fis[1] - fis[0]) * 100;
      segs += '<div class="br-seg" style="left:' + left.toFixed(2) + '%;width:' + larg.toFixed(2) + '%;background:' + COLORI_ZONA[zona(mid, s)] + '"></div>';
    }
    var pos = Math.max(0, Math.min(100, (val - fis[0]) / (fis[1] - fis[0]) * 100));
    return '<div class="barra-range">'
      + '<div class="br-intestazione"><span>' + esc(meta[2]) + '</span><span><strong>' + String(valore) + '</strong> ' + esc(meta[3]) + '</span></div>'
      + '<div class="br-track">' + segs + '<div class="br-marker" style="left:' + pos.toFixed(2) + '%"></div></div>'
      + '<div class="br-estremi"><span>' + fis[0] + '</span><span>' + fis[1] + '</span></div>'
      + '</div>';
  }

  function legendaZone() {
    return '<div class="legenda">'
      + '<span><span class="chip" style="background:#2e7d32"></span>nella norma</span>'
      + '<span><span class="chip" style="background:#ef6c00"></span>anomalia moderata</span>'
      + '<span><span class="chip" style="background:#c62828"></span>anomalia critica</span>'
      + '<span><span class="chip" style="background:#757575"></span>fuori range fisiologico</span>'
      + '</div>';
  }

  function schemaRete(rete, input, labels) {
    var a1 = rete.a1;
    var nIn = 6, nH = a1.length, nOut = 3;
    var passoH = 13;
    var passoIn = Math.min(40, (nH * passoH) / nIn);
    var passoOut = Math.min(52, (nH * passoH) / nOut);
    var yTop = 36, margineInf = 16;
    var h = Math.max(yTop + nH * passoH + margineInf + 10, 240);
    var w = 380;
    var xIn = 92, xHid = w / 2, xOut = w - 74;
    var yBot = h - margineInf;
    function centri(n, passo) {
      var tot = (n - 1) * passo;
      var start = (yTop + yBot - tot) / 2;
      var arr = [];
      for (var i = 0; i < n; i++) arr.push(start + i * passo);
      return arr;
    }
    var yIn = centri(nIn, passoIn), yHid = centri(nH, passoH), yOut = centri(nOut, passoOut);
    var classeOrdine = ['basso', 'medio', 'alto'];
    var predIdx = 0, bestP = -1;
    classeOrdine.forEach(function (cl, i) {
      var p = rete.probabilita ? rete.probabilita[cl] : 0;
      if (p !== undefined && p > bestP) { bestP = p; predIdx = i; }
    });
    var outFill = ['#2e7d32', '#ef6c00', '#c62828'];
    var svg = '<svg width="' + w + '" height="' + h + '" viewBox="0 0 ' + w + ' ' + h + '" role="img" aria-label="Grafo della rete neurale">';
    // Connessioni (disegnate sotto i nodi): più opache verso i neuroni accesi.
    for (var i = 0; i < nIn; i++) {
      for (var j = 0; j < nH; j++) {
        var op = a1[j] > 0 ? 0.22 : 0.07;
        svg += '<line x1="' + xIn + '" y1="' + yIn[i].toFixed(1) + '" x2="' + xHid + '" y2="' + yHid[j].toFixed(1) + '" stroke="#90a4ae" stroke-width="0.6" opacity="' + op + '"/>';
      }
    }
    for (var j2 = 0; j2 < nH; j2++) {
      for (var k = 0; k < nOut; k++) {
        var op2 = a1[j2] > 0 ? 0.22 : 0.07;
        svg += '<line x1="' + xHid + '" y1="' + yHid[j2].toFixed(1) + '" x2="' + xOut + '" y2="' + yOut[k].toFixed(1) + '" stroke="#90a4ae" stroke-width="0.6" opacity="' + op2 + '"/>';
      }
    }
    // Didascalie di colonna.
    svg += '<text x="' + xIn + '" y="18" text-anchor="middle" font-size="10" font-weight="bold" fill="#1b3a5b">ingressi (6)</text>';
    svg += '<text x="' + xHid + '" y="18" text-anchor="middle" font-size="10" font-weight="bold" fill="#1b3a5b">strato nascosto (' + nH + ') — ReLU</text>';
    svg += '<text x="' + xOut + '" y="18" text-anchor="middle" font-size="10" font-weight="bold" fill="#1b3a5b">uscite (3)</text>';
    // Nodi di ingresso: etichetta + tooltip con valore grezzo e normalizzato.
    for (var i2 = 0; i2 < nIn; i2++) {
      svg += '<circle cx="' + xIn + '" cy="' + yIn[i2].toFixed(1) + '" r="7" fill="#1b3a5b">'
        + '<title>' + esc(labels[i2]) + ' — valore ' + esc(input[FEATURES[i2][0]]) + ' ' + esc(FEATURES[i2][3])
        + ' (normalizzato ' + Number(rete.input_normalizzato[i2]).toFixed(2) + ')</title></circle>';
      svg += '<text x="' + (xIn - 12) + '" y="' + (yIn[i2] + 3).toFixed(1) + '" text-anchor="end" font-size="9" fill="#37474f">' + esc(labels[i2]) + '</text>';
    }
    // Nodi nascosti: blu se accesi (a1 > 0), grigi se la ReLU li ha azzerati.
    for (var j3 = 0; j3 < nH; j3++) {
      var acceso = a1[j3] > 0;
      svg += '<circle cx="' + xHid + '" cy="' + yHid[j3].toFixed(1) + '" r="5" fill="' + (acceso ? '#1565c0' : '#cfd8dc') + '">'
        + '<title>Neurone #' + (j3 + 1) + ' — ' + (acceso ? 'acceso (a1 = ' + a1[j3].toFixed(3) + ')' : 'spento (a1 = 0: z1 era negativo, la ReLU lo ha azzerato)') + '</title></circle>';
    }
    // Nodi di uscita: colori per classe, anello sulla classe predetta.
    for (var k2 = 0; k2 < nOut; k2++) {
      var pred = k2 === predIdx;
      var prob = rete.probabilita ? rete.probabilita[classeOrdine[k2]] : null;
      svg += '<circle cx="' + xOut + '" cy="' + yOut[k2].toFixed(1) + '" r="9" fill="' + outFill[k2] + '"' + (pred ? ' stroke="#0d1b2a" stroke-width="2.5"' : '') + '>'
        + '<title>' + classeOrdine[k2] + ' — probabilità ' + (prob === null || prob === undefined ? '—' : Number(prob).toFixed(4)) + (pred ? ' ← classe predetta' : '') + '</title></circle>';
      svg += '<text x="' + (xOut + 14) + '" y="' + (yOut[k2] + 3).toFixed(1) + '" font-size="10" ' + (pred ? 'font-weight="bold" fill="#0d1b2a"' : 'fill="#6c757d"') + '>' + classeOrdine[k2] + (pred ? ' ★' : '') + '</text>';
    }
    return svg + '</svg>';
  }

  function legendaRete() {
    return '<div class="legenda">'
      + '<span><span class="chip" style="background:#1b3a5b"></span>ingresso (valore misurato)</span>'
      + '<span><span class="chip" style="background:#1565c0"></span>neurone acceso (a1 &gt; 0)</span>'
      + '<span><span class="chip" style="background:#cfd8dc"></span>neurone spento (a1 = 0)</span>'
      + '<span><span class="chip" style="background:#ffffff;border:2px solid #0d1b2a;width:8px;height:8px;border-radius:50%"></span>classe predetta</span>'
      + '</div>';
  }

  function mostraRisposta(c, dati) {
    var r = dati.risposta || {};
    var s = stepNode('<div class="nodo-titolo">5. Risposta finale</div>'
      + '<div class="spiega">La classe finale non è semplicemente quella della rete: vale la regola di sicurezza <em>max(regole, MLP)</em>. Se le regole indicano un rischio più alto di quello predetto dalla rete, vincono loro (override di sicurezza); il messaggio al paziente e l’eventuale allerta medica derivano dalla classe risultante.</div>');
    var stato = r.classe === 'alto' ? 'stato-gate' : (r.classe === 'errore' ? 'stato-errore' : 'stato-ok');
    s.innerHTML += '<div class="nodo-esito ' + stato + '">Classe: <strong>' + esc(r.classe) + '</strong> ' + badge(r.percorso) + '</div>';
    s.innerHTML += '<div class="nodo-esito">' + esc(r.messaggio || '') + '</div>';
    if (r.allerta_medico) s.innerHTML += '<div class="nodo-esito stato-notifica">Allerta medica: Sì</div>';
    if (r.override_sicurezza) s.innerHTML += '<div class="nodo-esito stato-gate">Override di sicurezza: le regole hanno prevalso sull’MLP.</div>';
    c.appendChild(s); s.classList.add('show');
  }

  async function mostra(dati) {
    var c = document.getElementById('racconto');
    c.innerHTML = '';
    var attesa = 650;
    var input = dati.input || {};
    var v = dati.validazione || {};
    var soglie = v.soglie || {};

    var s1 = stepNode('<div class="nodo-titolo">1. Validazione dei parametri</div>'
      + '<div class="spiega">Prima di qualunque modello, il sistema controlla che i valori siano <em>fisiologicamente possibili</em> per un essere umano — non "normali": possibili. Una misurazione impossibile (per esempio una pressione di 5000) non è un rischio alto: è un errore di misura e viene rifiutata senza consultare né regole né rete. Vale anche il vincolo strutturale sistolica &gt; diastolica.</div>');
    if (!v.valido) {
      var errs = (v.errori || []).map(function (e) { return esc(e); }).join('<br>');
      s1.innerHTML += '<div class="nodo-esito stato-errore">Input NON plausibile. Motivi:<br>' + errs + '</div>';
      s1.innerHTML += FEATURES.map(function (f) { return barraRange(f[0], input[f[0]], soglie[f[0]], f); }).join('');
      s1.innerHTML += legendaZone();
      c.appendChild(s1); s1.classList.add('show');
      await delay(attesa);
      mostraRisposta(c, dati);
      return;
    }
    s1.innerHTML += '<div class="nodo-esito stato-ok">Tutti i valori sono entro i range fisiologici: campione plausibile, si prosegue.</div>';
    s1.innerHTML += FEATURES.map(function (f) { return barraRange(f[0], input[f[0]], soglie[f[0]], f); }).join('');
    s1.innerHTML += legendaZone();
    c.appendChild(s1); s1.classList.add('show');
    await delay(attesa);

    var sg = dati.safety_gate || {};
    var gateDisattivato = sg.attivo === false;
    var s2 = stepNode('<div class="nodo-titolo">2. Safety gate — regole cliniche (il teacher)</div>'
      + '<div class="spiega">Un esperto ha scritto a mano le regole cliniche: range normali, soglie moderate e critiche per ogni parametro, più pattern di correlazione pericolosi (per esempio ipotensione + tachicardia = shock). Questo classificatore a regole è trasparente al 100% — ogni decisione si può leggere e spiegare — e ha <em>precedenza assoluta</em>: se dichiara il caso critico, la rete neurale non viene nemmeno consultata.</div>');
    var anomalie = (sg.anomalie || []).map(function (a) {
      var crit = a.gravita === 'critica';
      return '<div class="anomalia' + (crit ? ' critica' : '') + '">' + esc(a.parametro) + ': ' + esc(a.valore) + ' (' + esc(a.direzione) + ', gravità ' + esc(a.gravita) + ')</div>';
    }).join('') || '<div class="segnaposto">Nessuna anomalia: tutti i valori nel range normale.</div>';
    var pattern = (sg.pattern || []).map(function (p) { return '<div class="pattern-item">' + esc(p) + '</div>'; }).join('');
    s2.innerHTML += '<div class="nodo-esito">Verdetto delle regole: <strong>' + esc(sg.classe_regola || '-') + '</strong></div>' + anomalie + pattern;
    if (gateDisattivato) {
      s2.innerHTML += '<div class="nodo-esito stato-ok">Safety gate DISATTIVATO (modalità diagnostica): il verdetto delle regole è mostrato solo come confronto, la decisione finale spetta all’MLP.</div>';
    }
    c.appendChild(s2); s2.classList.add('show');
    await delay(attesa);

    if (!gateDisattivato && dati.stopped_at === 'gate') {
      var sg2 = stepNode('<div class="nodo-titolo">3. MLP bypassato (GATE)</div>'
        + '<div class="spiega">Le regole hanno rilevato una condizione critica, quindi il caso non passa dalla rete: per costruzione un caso "alto" non può mai essere declassato dall’MLP. Il richiamo di sistema sui casi critici è 1.0 — è la rete di sicurezza del progetto.</div>'
        + '<div class="nodo-esito stato-gate">Caso critico: precedenza assoluta del safety gate, MLP bypassato.</div>');
      c.appendChild(sg2); sg2.classList.add('show');
      await delay(attesa);
      mostraRisposta(c, dati);
      return;
    }

    var rete = dati.rete;
    if (!rete) {
      var sn = stepNode('<div class="nodo-titolo">3. Rete neurale (MLP)</div><div class="nodo-esito stato-errore">Modello non disponibile: viene usata la regola.</div>');
      c.appendChild(sn); sn.classList.add('show');
      await delay(attesa);
      mostraRisposta(c, dati);
      return;
    }
    var labels = FEATURES.map(function (f) { return f[1]; });

    var s3a = stepNode('<div class="nodo-titolo">3a. Input normalizzato (scaler)</div>'
      + '<div class="spiega">I 6 parametri hanno unità diverse (mmHg, bpm, °C…): così come sono non sono confrontabili tra loro. Lo scaler li standardizza con la formula (x − media) / deviazione_standard, calcolata sul solo training set. Nella scala risultante 0 significa "esattamente la media", +2 "molto sopra la media", −1.5 "sotto la media".</div>'
      + vettore(rete.input_normalizzato, labels));
    c.appendChild(s3a); s3a.classList.add('show'); await delay(attesa);

    var s3b = stepNode('<div class="nodo-titolo">3b. Strato nascosto — pre-attivazione z1</div>'
      + '<div class="spiega">Ognuno dei ' + rete.z1.length + ' neuroni nascosti calcola una somma pesata degli ingressi — i pesi W1 sono le "manopole" regolate dal training — più un proprio termine di bias: z1 = X·W1ᵀ + b1. Ogni numero è il punteggio di un neurone <em>prima</em> dell’attivazione.</div>'
      + '<div class="rete-figura">' + schemaRete(rete, input, labels) + '<div class="rete-figcaption">Grafo della rete: ogni colonna è uno strato; passa il mouse sui pallini per stato e valori. Blu = neurone acceso dalla ReLU, anello scuro = classe predetta.</div></div>'
      + legendaRete()
      + vettore(rete.z1, null));
    c.appendChild(s3b); s3b.classList.add('show'); await delay(attesa);

    var attivi = rete.a1.filter(function (x) { return x > 0; }).length;
    var s3c = stepNode('<div class="nodo-titolo">3c. Attivazione ReLU: a1 = max(0, z1)</div>'
      + '<div class="spiega">La ReLU è la funzione di attivazione: tronca i valori negativi a zero, quindi solo i neuroni con segnale positivo "si accendono" e trasferiscono il loro contributo allo strato successivo. È questa non-linearità che permette alla rete di apprendere confini decisionali complessi, non soltanto rette.</div>'
      + '<div class="nota-attivi">Neuroni accesi: ' + attivi + ' su ' + rete.a1.length + '</div>'
      + vettore(rete.a1, null));
    c.appendChild(s3c); s3c.classList.add('show'); await delay(attesa);

    var s3d = stepNode('<div class="nodo-titolo">3d. Strato di uscita — logits z2</div>'
      + '<div class="spiega">Il secondo strato combina le attivazioni dei neuroni accesi in 3 punteggi grezzi (logit), uno per classe di rischio: z2 = a1·W2ᵀ + b2. Il logit più alto indica la classe favorita; la distanza tra i logit dice quanto la rete è netta.</div>'
      + vettore(rete.z2, ['basso', 'medio', 'alto']));
    c.appendChild(s3d); s3d.classList.add('show'); await delay(attesa);

    var s3e = stepNode('<div class="nodo-titolo">3e. Softmax → probabilità</div>'
      + '<div class="spiega">La softmax eleva a esponenziale i logit e li divide per la loro somma: il risultato è una distribuzione di probabilità la cui somma fa esattamente 1. Logit molto distanti → confidenza alta; logit simili → la rete è indecisa.</div>'
      + barreProb(rete.probabilita));
    c.appendChild(s3e); s3e.classList.add('show'); await delay(attesa);

    var si = dati.soglia_incertezza || {};
    var conf = (si.confidenza === null || si.confidenza === undefined) ? 0 : si.confidenza;
    var s4 = stepNode('<div class="nodo-titolo">4. Soglia di incertezza</div>'
      + '<div class="spiega">Il sistema conosce i propri limiti: se la probabilità massima scende sotto ' + si.soglia + ', non si fida della rete e ricade sulle regole testuali (fallback). È la "zona di incertezza": meglio una risposta trasparente basata su regole che una risposta sicura ma sbagliata.</div>');
    if (si.fallback) {
      s4.innerHTML += '<div class="nodo-esito stato-fallback">Confidenza ' + Number(conf).toFixed(4) + ' &lt; soglia ' + si.soglia + ' → fallback alle regole.</div>' + scalaConf(si.confidenza, si.soglia);
    } else {
      s4.innerHTML += '<div class="nodo-esito stato-ok">Confidenza ' + Number(conf).toFixed(4) + ' ≥ soglia ' + si.soglia + ' → la risposta della rete è affidabile.</div>' + scalaConf(si.confidenza, si.soglia);
    }
    c.appendChild(s4); s4.classList.add('show'); await delay(attesa);

    mostraRisposta(c, dati);
  }

  function leggiForm() {
    var p = {};
    FEATURES.forEach(function (f) {
      var el = document.getElementById('in_' + f[0]);
      var val = parseFloat(el.value);
      p[f[0]] = isNaN(val) ? el.value : val;
    });
    return p;
  }

  function init() {
    var form = document.getElementById('form-campioni');
    if (!form) return;
    var btnEsempio = document.getElementById('btn-esempio');
    if (btnEsempio) {
      btnEsempio.addEventListener('click', function () {
        var es = ESEMPI[indiceEsempio % ESEMPI.length];
        indiceEsempio++;
        FEATURES.forEach(function (f) {
          var el = document.getElementById('in_' + f[0]);
          if (el) el.value = es.valori[f[0]];
        });
        var posizione = ((indiceEsempio - 1) % ESEMPI.length) + 1;
        var stato = document.getElementById('stato-valuta');
        if (stato) stato.textContent = 'Esempio ' + posizione + '/' + ESEMPI.length + ': ' + es.nome + '. Premi Valuta per analizzarlo.';
      });
    }
    form.addEventListener('submit', function (ev) {
      ev.preventDefault();
      var stato = document.getElementById('stato-valuta');
      var c = document.getElementById('racconto');
      c.innerHTML = '<p class="segnaposto">Analisi in corso…</p>';
      if (stato) stato.textContent = 'Valutazione…';
      fetch('/api/valuta', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ parametri: leggiForm(), gate: document.getElementById('gateAttivo').checked })
      }).then(function (r) { return r.json(); }).then(function (dati) {
        if (stato) stato.textContent = '';
        mostra(dati);
      }).catch(function () {
        if (stato) stato.textContent = 'Errore di rete (serve attivo con --serve?)';
      });
    });
  }

  if (document.readyState !== 'loading') init();
  else document.addEventListener('DOMContentLoaded', init);
})();
"""


def _pannello_riproducibilita(
    metadati: Optional[Dict[str, Any]],
    avviso_metadati: Optional[str],
) -> str:
    """Vista 6: seed, split e hash da metadati.json (degrado grazioso)."""
    if avviso_metadati or not isinstance(metadati, dict):
        return (
            '<section class="tab-panel" data-tab="riproducibilita">'
            f'<div class="avviso">{html.escape(avviso_metadati or "Metadati non disponibili")}</div>'
            '<p class="segnaposto">Eseguire python training/generate_dataset.py '
            "per generare il dataset e i metadati.</p>"
            "</section>"
        )
    blocchi = metadati.get("blocchi") or {}
    righe_split = "".join(
        f"<tr><td>{nome}</td><td>{html.escape(str(blocco.get('n', '—')))}</td>"
        f"<td>{html.escape(str(blocco.get('seed', '—')))}</td></tr>"
        for nome, chiave, blocco in (
            ("Train", "train", blocchi.get("train") or {}),
            ("Validazione", "val", blocchi.get("val") or {}),
            ("Test", "test", blocchi.get("test") or {}),
        )
    )
    hash_componenti = (
        ("safety_rules", "hash_safety_rules"),
        ("synthetic_generator", "hash_synthetic_generator"),
        ("teacher_rules", "hash_teacher_rules"),
    )
    righe_hash = "".join(
        f"<tr><td>{nome}</td><td><code>"
        f"{html.escape(str(metadati.get(chiave) or '—'))}</code></td></tr>"
        for nome, chiave in hash_componenti
    )
    return (
        '<section class="tab-panel" data-tab="riproducibilita">'
        '<div class="card">'
        "<h2>Riproducibilità</h2>"
        '<p><span class="badge-deterministico">Report deterministico '
        "(stesso seed → stessi numeri)</span></p>"
        f"<h3>Seed base</h3><p>{html.escape(str(metadati.get('seed_base') or '—'))}</p>"
        "<h3>Split del dataset</h3>"
        '<table class="tabella-live"><thead><tr><th>Split</th>'
        "<th>Record</th><th>Seed</th></tr></thead>"
        f"<tbody>{righe_split}</tbody></table>"
        "<h3>Hash di provenienza</h3>"
        '<table class="tabella-live"><thead><tr><th>Componente</th>'
        "<th>Hash</th></tr></thead>"
        f"<tbody>{righe_hash}</tbody></table>"
        "</div></section>"
    )


# ---------------------------------------------------------------------------
# Viste 3-4: figure matplotlib (lazy, via tools/_figure.py) con degrado
# ---------------------------------------------------------------------------

def _figura_html(base64_png: Optional[str], didascalia: str, fallback: str) -> str:
    """Figura base64 come <img> incorporato, o segnaposto con messaggio."""
    if base64_png:
        return (
            '<figure class="figura">'
            f'<img class="figura-img" src="data:image/png;base64,{base64_png}" '
            f'alt="{html.escape(didascalia)}">'
            f"<figcaption>{html.escape(didascalia)}</figcaption>"
            "</figure>"
        )
    return f'<p class="segnaposto">{html.escape(fallback)}</p>'


def _carica_test_e_modello(
    test_dir: Optional[Path] = None,
) -> Tuple[Optional[Any], Optional[Any], Optional[Any], Optional[str]]:
    """(X_test, y_test, modello) con degrado grazioso; modello espone .scaler.

    ``test_dir`` (default: DATA_PROCESSED_DIR) è la directory con X_test.npy
    e y_test.npy — parametrizzata per i test con dataset finto.
    """
    try:
        import numpy as np
        from telemedicina_supervised.ml.mlp import MLP

        dir_dati = test_dir or DATA_PROCESSED_DIR
        X_test = np.load(dir_dati / "X_test.npy")
        y_test = np.load(dir_dati / "y_test.npy")
        modello = MLP.carica(ARTIFACT_DEFAULT)
        if modello.scaler is None:
            return None, None, None, (
                "Artifact senza scaler: rigenerare con "
                "python training/train_model.py --build-ml"
            )
        return X_test, y_test, modello, None
    except Exception as exc:
        print(f"Errore durante il caricamento di test/artifact: {exc}", file=sys.stderr)
        return None, None, None, (
            "Dati di test o artifact non disponibili: eseguire "
            "python training/train_model.py --build-ml"
        )


def _pannello_distillazione(
    report: Optional[Dict[str, Any]],
    avviso_report: Optional[str],
    test_dir: Optional[Path] = None,
) -> str:
    """Vista 3: teacher vs MLP (banner metriche, confusione, scatter, regioni).

    I numeri del banner sono LETTI da report.json (mai hardcoded); le figure
    ricalcolano solo ciò che serve alla visualizzazione (predizioni MLP,
    distanze dalle soglie, label del teacher sulla griglia).
    """
    if avviso_report or not isinstance(report, dict):
        return (
            '<section class="tab-panel" data-tab="teacher">'
            f'<div class="avviso">{html.escape(avviso_report or "Report non disponibile")}</div>'
            '<p class="segnaposto">Eseguire python training/train_model.py --build-ml '
            "per generare il report con le metriche.</p>"
            "</section>"
        )
    congelato = report.get("metriche_test_congelato") or {}
    no_buffer = report.get("metriche_test_no_buffer") or {}
    sicurezza = congelato.get("sicurezza") or {}
    acc_c = congelato.get("accuracy")
    acc_nb = no_buffer.get("accuracy")
    kappa = congelato.get("kappa_cohen")
    recall_alto = sicurezza.get("recall_alto")

    def _num(valore: Any, decimali: int = 4) -> str:
        return f"{valore:.{decimali}f}" if isinstance(valore, (int, float)) else "—"

    metriche = (
        '<div class="metriche-grid">'
        f'<div class="metriche-card"><span class="metriche-valore">{_num(acc_c)}</span>'
        '<span class="metriche-etichetta">Accuracy test congelato (bufferizzato)</span></div>'
        f'<div class="metriche-card"><span class="metriche-valore">{_num(acc_nb)}</span>'
        '<span class="metriche-etichetta">Accuracy test senza buffer</span></div>'
        f'<div class="metriche-card"><span class="metriche-valore">{_num(kappa)}</span>'
        '<span class="metriche-etichetta">Cohen kappa (congelato)</span></div>'
        f'<div class="metriche-card"><span class="metriche-valore">{_num(recall_alto)}</span>'
        '<span class="metriche-etichetta">Recall classe alto (congelato)</span></div>'
        "</div>"
    )
    nota = report.get("interpretazione")
    nota_html = (
        f'<p class="nota">{html.escape(nota)}</p>' if isinstance(nota, str) else ""
    )

    try:
        import _figure  # lazy: matplotlib non serve se il report manca
    except Exception as exc:
        print(f"Errore durante l'import di _figure: {exc}", file=sys.stderr)
        _figure = None

    matrice = congelato.get("matrice_confusione")
    fig_confusione = (
        _figure.figura_confusione(matrice, "Matrice di confusione — test congelato")
        if _figure is not None and matrice
        else None
    )
    X_test, y_test, modello, avviso_dati = _carica_test_e_modello(test_dir)
    fig_scatter = None
    fig_regioni = None
    if _figure is not None and modello is not None and X_test is not None and y_test is not None:
        fig_scatter = _figure.figura_scatter_errori(
            X_test, y_test, modello, modello.scaler
        )
        fig_regioni = _figure.figura_heatmap_regioni(
            X_test, modello, modello.scaler
        )
    avviso_dati_html = (
        f'<div class="avviso">{html.escape(avviso_dati)}</div>' if avviso_dati else ""
    )

    return (
        '<section class="tab-panel" data-tab="teacher">'
        '<div class="card">'
        "<h2>Knowledge distillation — teacher vs MLP</h2>"
        f"{metriche}"
        f"{nota_html}"
        "</div>"
        '<div class="card">'
        "<h2>Matrice di confusione (test congelato)</h2>"
        f"{_figura_html(fig_confusione, 'Matrice di confusione — test congelato', 'Matrice di confusione non disponibile nel report.')}"
        "</div>"
        '<div class="card">'
        "<h2>Errori dell'MLP sul test congelato</h2>"
        f"{avviso_dati_html}"
        f"{_figura_html(fig_scatter, 'Scatter sistolica × glicemia con errori evidenziati', 'Scatter non disponibile: servono dataset di test e artifact MLP.')}"
        "</div>"
        '<div class="card">'
        "<h2>Regioni di decisione (teacher vs MLP)</h2>"
        f"{_figura_html(fig_regioni, 'Regioni di decisione su sistolica × glicemia', 'Heatmap non disponibile: servono dataset di test e artifact MLP.')}"
        "</div>"
        "</section>"
    )


def _pannello_training(
    report: Optional[Dict[str, Any]],
    avviso_report: Optional[str],
) -> str:
    """Vista 4: curve loss, confronto grid, gradient check, architettura."""
    if avviso_report or not isinstance(report, dict):
        return (
            '<section class="tab-panel" data-tab="training">'
            f'<div class="avviso">{html.escape(avviso_report or "Report non disponibile")}</div>'
            '<p class="segnaposto">Eseguire python training/train_model.py --build-ml '
            "per generare il report con le metriche.</p>"
            "</section>"
        )
    try:
        import _figure  # lazy: matplotlib non serve se il report manca
    except Exception as exc:
        print(f"Errore durante l'import di _figure: {exc}", file=sys.stderr)
        _figure = None

    fig_loss = _figure.figura_loss_curve(report) if _figure is not None else None
    config = report.get("config_migliore") or {}
    fig_arch = (
        _figure.figura_architettura(config.get("n_hidden"))
        if _figure is not None
        else None
    )

    # Confronto grid: le 6 configurazioni ufficiali (config.GRID). Il loss_val
    # del punto di scelta non è registrato nel report corrente -> colonna "—"
    # e messaggio esplicito (la modifica di train_model.py è fuori scope V3).
    righe_grid = "".join(
        f"<tr><td>{c.get('n_hidden')}</td><td>{c.get('lr')}</td>"
        f"<td>{c.get('batch_size')}</td><td>{c.get('epoche')}</td><td>—</td></tr>"
        for c in GRID
    )
    grid_html = (
        '<div class="card">'
        "<h2>Confronto grid (6 configurazioni)</h2>"
        '<table class="tabella-live"><thead><tr><th>Hidden</th><th>LR</th>'
        "<th>Batch</th><th>Epoche</th><th>Loss val (punto di scelta)</th></tr></thead>"
        f"<tbody>{righe_grid}</tbody></table>"
        '<p class="nota">Il loss_val per configurazione non è registrato nel '
        "report corrente: la registrazione del confronto grid è fuori scope "
        "della Fase V3 (richiederebbe una modifica a training/train_model.py).</p>"
        "</div>"
    )

    return (
        '<section class="tab-panel" data-tab="training">'
        '<div class="card">'
        "<h2>Curve di loss — config migliore</h2>"
        f"{_figura_html(fig_loss, 'Loss train/val della config migliore', 'Curve di loss non disponibili nel report.')}"
        "</div>"
        f"{grid_html}"
        '<div class="card">'
        "<h2>Gradient check</h2>"
        '<p><span class="badge-deterministico">Gradient check ~1e-10</span> '
        "verificato dal test automatico <code>tests/test_mlp.py</code> "
        "(errore massimo relativo &lt; 2e-5).</p>"
        "</div>"
        '<div class="card">'
        "<h2>Architettura dell'MLP</h2>"
        f"{_figura_html(fig_arch, 'Architettura: input 6 → hidden ReLU → softmax 3', 'Diagramma non disponibile.')}"
        "</div>"
        "</section>"
    )


def _pannello_incertezza(
    report: Optional[Dict[str, Any]],
    avviso_report: Optional[str],
    notifiche: List[Dict[str, Any]],
    test_dir: Optional[Path] = None,
) -> str:
    """Vista 5: zona di incertezza e sicurezza.

    - Istogramma della confidenza softmax sul test congelato (figura).
    - Contatori di sistema: casi critici del test (classe ``alto``) e override
      di sicurezza LETTI da report.json; notifiche totali dal DB (read_only).
    - Tabella dei mancati dell'MLP RICALCOLATA sul test congelato (pred MLP
      vs label, casi ``alto`` non predetti ``alto``): il conteggio mostrato è
      quello reale, non un valore atteso.
    - Il richiamo di sistema sui critici è 1.0 per costruzione (safety gate
      a precedenza assoluta): testo esplicito, non figura.
    """
    if avviso_report or not isinstance(report, dict):
        return (
            '<section class="tab-panel" data-tab="incertezza">'
            f'<div class="avviso">{html.escape(avviso_report or "Report non disponibile")}</div>'
            '<p class="segnaposto">Eseguire python training/train_model.py --build-ml '
            "per generare il report con le metriche.</p>"
            "</section>"
        )

    congelato = report.get("metriche_test_congelato") or {}
    sicurezza = congelato.get("sicurezza") or {}
    distribuzione = report.get("distribuzione_classi") or {}
    test_distr = distribuzione.get("test") or {}
    # Chiavi della distribuzione per indice ("2") o per nome ("alto").
    critici = test_distr.get("2", test_distr.get("alto"))

    def _num(valore: Any, decimali: int = 4) -> str:
        return f"{valore:.{decimali}f}" if isinstance(valore, (int, float)) else "—"

    override = report.get("override_sicurezza")
    nota_override = ""
    if override is None:
        override = 0
        nota_override = (
            '<p class="nota">Override di sicurezza: non registrato nel report '
            "(il runtime lo traccia in SQLite, non nel report di training).</p>"
        )
    notifiche_totali = len(notifiche)

    metriche = (
        '<div class="metriche-grid">'
        f'<div class="metriche-card"><span class="metriche-valore">{_num(critici, 0)}</span>'
        '<span class="metriche-etichetta">Casi critici nel test (classe alto)</span></div>'
        f'<div class="metriche-card"><span class="metriche-valore">{_num(override, 0)}</span>'
        '<span class="metriche-etichetta">Override di sicurezza</span></div>'
        f'<div class="metriche-card"><span class="metriche-valore">{notifiche_totali}</span>'
        '<span class="metriche-etichetta">Notifiche totali (DB)</span></div>'
        "</div>"
        f"{nota_override}"
        '<p class="nota"><strong>Richiamo di sistema sui casi critici = 1.0 '
        "per costruzione</strong> — il safety gate non lascia mai passare un "
        "caso 'alto' dall'MLP.</p>"
    )

    try:
        import _figure  # lazy: matplotlib non serve se il report manca
    except Exception as exc:
        print(f"Errore durante l'import di _figure: {exc}", file=sys.stderr)
        _figure = None

    X_test, y_test, modello, avviso_dati = _carica_test_e_modello(test_dir)

    fig_istogramma = None
    if _figure is not None and modello is not None and X_test is not None:
        fig_istogramma = _figure.figura_istogramma_confidenze(
            X_test, modello, modello.scaler
        )

    # Tabella dei mancati: pred MLP vs label, casi 'alto' non predetti 'alto'.
    # Colonne: indice, sistolica/glicemia (ordine feature da safety_rules),
    # distanza dalle soglie (riuso di training.metrics).
    if modello is not None and X_test is not None and y_test is not None:
        import numpy as np
        from training.metrics import distanza_dalle_soglie

        y = np.asarray(y_test)
        pred = modello.predici_etichette(modello.scaler.transform(X_test))
        indici = np.flatnonzero((y == "alto") & (pred != "alto"))
        distanze = np.asarray(distanza_dalle_soglie(X_test), dtype=float)
        # Indici di sistolica/glicemia da FEATURE_ORDER (fonte unica in
        # tools/_figure.py), con fallback difensivo 0/5.
        try:
            i_sist = _figure.FEATURE_ORDER.index("pressione_sistolica")
            i_glic = _figure.FEATURE_ORDER.index("glicemia")
        except (AttributeError, ValueError):
            i_sist, i_glic = 0, 5
        righe = "".join(
            "<tr>"
            f"<td>{int(i)}</td>"
            f"<td>{float(X_test[i, i_sist]):.1f}</td>"
            f"<td>{float(X_test[i, i_glic]):.1f}</td>"
            f"<td>{float(distanze[i]):.3f}</td>"
            "</tr>"
            for i in indici
        )
        n_mancati = len(indici)
        tabella = (
            f'<table class="tabella-live" id="tabella-mancati">'
            "<thead><tr><th>Indice</th><th>Sistolica</th><th>Glicemia</th>"
            "<th>Distanza dalle soglie</th></tr></thead>"
            f"<tbody>{righe}</tbody></table>"
            '<p class="nota">Mancati ricalcolati sul test congelato '
            "(predizioni MLP vs label del teacher): il conteggio mostrato è "
            "quello reale del dataset corrente.</p>"
        )
    else:
        n_mancati = None
        tabella = (
            '<p class="segnaposto">'
            f"{html.escape(avviso_dati or 'Dati di test o artifact non disponibili.')}"
            "</p>"
        )

    card_mancati = (
        '<div class="metriche-grid">'
        f'<div class="metriche-card"><span class="metriche-valore">'
        f"{_num(n_mancati, 0) if n_mancati is not None else '—'}</span>"
        '<span class="metriche-etichetta">Mancati dell\'MLP sul test congelato '
        "(ricalcolati)</span></div></div>"
    )

    return (
        '<section class="tab-panel" data-tab="incertezza">'
        '<div class="card">'
        "<h2>Zona di incertezza e sicurezza</h2>"
        f"{metriche}"
        "</div>"
        '<div class="card">'
        "<h2>Confidenza del modello sul test congelato</h2>"
        f"{_figura_html(fig_istogramma, 'Istogramma della confidenza softmax — soglia di incertezza evidenziata', 'Istogramma non disponibile: servono dataset di test e artifact MLP (e matplotlib per la figura).')}"
        "</div>"
        '<div class="card">'
        "<h2>Mancati dell'MLP (casi critici non riconosciuti)</h2>"
        f"{card_mancati}"
        f"{tabella}"
        "</div>"
        "</section>"
    )


def _fmt_soglia(soglia: Dict[str, Any]) -> str:
    """Formatta una coppia di soglie {'alta': x|None, 'bassa': y|None}."""
    parti = []
    if soglia.get("alta") is not None:
        parti.append(f"≥ {soglia['alta']}")
    if soglia.get("bassa") is not None:
        parti.append(f"≤ {soglia['bassa']}")
    return " · ".join(parti) if parti else "—"


def _pannello_interattiva(gate_default: bool = True) -> str:
    """Vista 7: form campione + racconto passo-passo della valutazione."""
    campi = []
    for nome in FEATURE_ORDER:
        min_v, max_v = RANGE_FISIOLOGICI[nome]
        etichetta = ETICHETTE.get(nome, nome)
        unita = UNITA.get(nome, "")
        campi.append(
            f'<div class="campo">'
            f'<label for="in_{nome}">{html.escape(etichetta)}</label>'
            f'<input type="number" step="any" id="in_{nome}" name="{nome}" '
            f'placeholder="{min_v}-{max_v} {html.escape(unita)}">'
            f'<span class="hint">fisiologico {min_v}-{max_v} {html.escape(unita)}</span>'
            f'</div>'
        )
    form = "".join(campi)

    righe_soglie = ""
    for nome in FEATURE_ORDER:
        nor = RANGE_NORMALI[nome]
        righe_soglie += (
            f"<tr><td>{html.escape(ETICHETTE.get(nome, nome))}</td>"
            f"<td>{nor[0]}–{nor[1]}</td>"
            f"<td>{_fmt_soglia(SOGLIE_MODERATE[nome])}</td>"
            f"<td>{_fmt_soglia(SOGLIE_CRITICHE[nome])}</td></tr>"
        )

    return (
        '<section class="tab-panel" data-tab="interattiva">'
        '<div class="card">'
        "<h2>Analisi interattiva — inserisci un campione</h2>"
        '<p class="nota">Inserisci 6 parametri vitali: il sistema li valuta '
        "passo-passo, mostrando cosa succede dentro la rete fino alla risposta.</p>"
        f'<form id="form-campioni" class="form-campioni">{form}</form>'
        '<div class="toolbar-interattiva">'
        '<button type="submit" form="form-campioni">Valuta</button>'
        '<button type="button" id="btn-esempio" class="secondario" title="Clicca di nuovo per ciclare: basso → medio → alto rischio">Esempio</button>'
        f'<label class="check-gate"><input type="checkbox" id="gateAttivo"{" checked" if gate_default else ""}> '
        "Safety gate attivo (disattiva per modalità diagnostica: decide solo l'MLP)</label>"
        '<span id="stato-valuta"></span>'
        "</div>"
        '<div id="racconto"></div>'
        "<h3>Soglie dei gate (riferimento)</h3>"
        '<table class="tabella-live"><thead><tr><th>Parametro</th><th>Normale</th>'
        "<th>Moderata</th><th>Critica</th></tr></thead>"
        f"<tbody>{righe_soglie}</tbody></table>"
        '<p class="nota">La validazione controlla solo che i valori siano '
        "fisiologicamente <em>possibili</em>; il safety gate classifica con "
        "queste soglie: qualunque anomalia → medio, anomalia critica o "
        "pattern di correlazione → alto (MLP bypassato).</p>"
        "</div></section>"
    )


def genera_html(
    analisi: List[Dict[str, Any]],
    notifiche: List[Dict[str, Any]],
    avviso: Optional[str] = None,
    metadati: Optional[Dict[str, Any]] = None,
    avviso_metadati: Optional[str] = None,
    report: Optional[Dict[str, Any]] = None,
    avviso_report: Optional[str] = None,
    test_dir: Optional[Path] = None,
    gate_default: bool = True,
) -> str:
    """Costruisce il documento HTML completo (tabella Live server-side)."""
    timestamp_notifiche = {n.get("timestamp") for n in notifiche}
    righe = "".join(_riga_html(r, timestamp_notifiche) for r in analisi)
    corpo_tabella = f"<tbody>{righe}</tbody>" if analisi else ""
    avviso_html = (
        f'<div class="avviso">{html.escape(avviso)}</div>' if avviso else ""
    )

    # JSON incorporato per la Vista 2 (flusso) e per il refresh con --serve.
    record_json = [_record_json(r, timestamp_notifiche) for r in analisi]
    dati_json = json.dumps(
        {"soglia_incertezza": SOGLIA_INCERTEZZA, "record": record_json},
        ensure_ascii=False,
    ).replace("</", "<\\/")

    bottoni = "".join(
        f'<button type="button" class="tab{" active" if chiave == "live" else ""}"'
        f' data-tab="{chiave}">{html.escape(etichetta)}</button>'
        for chiave, etichetta in TAB
    )

    pannelli = []
    for chiave, etichetta in TAB:
        if chiave == "live":
            pannelli.append(
                '<section class="tab-panel active" data-tab="live">'
                f"{avviso_html}"
                '<div class="toolbar-live">'
                '<button type="button" id="btn-aggiorna">Aggiorna</button>'
                '<span id="stato-aggiorna"></span>'
                "</div>"
                '<table class="tabella-live">'
                "<thead><tr>"
                "<th>Timestamp</th><th>Classe</th><th>Allerta medico</th>"
                "<th>Errori</th><th>Probabilità</th><th>Messaggio</th>"
                "<th>Motivo fallback</th><th>Override sicurezza</th><th>Badge</th>"
                "</tr></thead>"
                f"{corpo_tabella}"
                "</table></section>"
            )
        elif chiave == "flusso":
            pannelli.append(
                '<section class="tab-panel" data-tab="flusso">'
                '<div class="card flusso">'
                "<h2>Flusso decisionale del caso</h2>"
                '<p class="segnaposto">Seleziona una riga nella tabella '
                '"1. Analisi Live" per vedere il percorso del caso.</p>'
                '<div id="flusso-decisionale"></div>'
                "</div></section>"
            )
        elif chiave == "teacher":
            pannelli.append(_pannello_distillazione(report, avviso_report, test_dir))
        elif chiave == "training":
            pannelli.append(_pannello_training(report, avviso_report))
        elif chiave == "riproducibilita":
            pannelli.append(_pannello_riproducibilita(metadati, avviso_metadati))
        elif chiave == "incertezza":
            pannelli.append(_pannello_incertezza(report, avviso_report, notifiche, test_dir))
        elif chiave == "interattiva":
            pannelli.append(_pannello_interattiva(gate_default))
        else:
            pannelli.append(
                f'<section class="tab-panel" data-tab="{chiave}">'
                '<p class="segnaposto">Pannello non disponibile.</p>'
                "</section>"
            )
    pannelli_html = "".join(pannelli)
    vincoli_html = "".join(
        f'<span class="vincolo">{html.escape(v)}</span>' for v in VINCOLI
    )

    return f"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Telemedicina — Sistema Supervised</title>
<style>
{_css()}
</style>
</head>
<body>
<header>
  <h1>Telemedicina — Sistema Supervised</h1>
  <p class="sottotitolo">{html.escape(SOTTOTITOLO)}</p>
  <div class="vincoli">{vincoli_html}</div>
  <div class="banner">{html.escape(BANNER)}</div>
</header>
<nav class="tab-bar">{bottoni}</nav>
<main>{pannelli_html}</main>
<footer>Dashboard generata localmente — nessuna richiesta di rete. <a href="/allenamento.html">Allenamento dal vivo →</a></footer>
<script type="application/json" id="dati-analisi">{dati_json}</script>
<script>
{_js()}
</script>
<script>
{_js_interattiva()}
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Genera la dashboard HTML statica del sistema supervised (Fase V4: dashboard completa)"
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        type=Path,
        default=OUTPUT_DEFAULT,
        help=f"percorso del file HTML di output (default: {OUTPUT_DEFAULT})",
    )
    parser.add_argument(
        "--db-path",
        dest="db_path",
        type=Path,
        default=DB_PATH,
        help=f"percorso del database SQLite (default: {DB_PATH})",
    )
    parser.add_argument(
        "--metadati-path",
        dest="metadati_path",
        type=Path,
        default=METADATI_DEFAULT,
        help=f"percorso di metadati.json (default: {METADATI_DEFAULT})",
    )
    parser.add_argument(
        "--report-path",
        dest="report_path",
        type=Path,
        default=REPORT_DEFAULT,
        help=f"percorso di report.json (default: {REPORT_DEFAULT})",
    )
    parser.add_argument(
        "--test-path",
        dest="test_path",
        type=Path,
        default=DATA_PROCESSED_DIR,
        help=f"directory con X_test.npy e y_test.npy (default: {DATA_PROCESSED_DIR})",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="dopo il build, serve il file su http://127.0.0.1:8000 (o variabile PORT)",
    )
    parser.add_argument(
        "--no-gate",
        dest="no_gate",
        action="store_true",
        help=(
            "disattiva il safety gate di default — modalità diagnostica; "
            "la checkbox nel pannello interattivo può override per richiesta"
        ),
    )
    return parser


def crea_handler(
    db_path: Path,
    directory: Path,
    agent=None,
    gate_default: bool = True,
):
    """Classe handler http: file statici da ``directory`` + API.

    - ``GET /api/analisi``: JSON dei record dal DB (sola lettura, tab Live).
    - ``POST /api/valuta``: riceve ``{"parametri": {...}}`` e ritorna la
      traccia completa della pipeline (validazione -> gate -> rete ->
      soglia -> risposta) per la pagina interattiva. Il corpo può contenere
      anche ``"gate"`` (bool, opzionale): assente vale ``gate_default``
      (modalità diagnostica quando il tool è avviato con ``--no-gate``).

    ``agent`` (SupervisedAgent) è opzionale: se omesso ne viene creato uno
    lazy (il modello è caricato/cacheato alla prima richiesta di valutazione).
    """
    import http.server

    agente = agent if agent is not None else SupervisedAgent()

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

        def do_GET(self):
            if self.path == "/api/analisi":
                self._api_analisi()
                return
            if self.path == "/api/allena/stato":
                try:
                    import allena_live

                    self._json(200, allena_live.stato_sessione())
                except Exception as exc:  # mai traceback verso il client
                    self._json(500, {"errore": str(exc)})
                return
            super().do_GET()

        def do_POST(self):
            try:
                lunghezza = int(self.headers.get("Content-Length", 0) or 0)
                corpo = self.rfile.read(lunghezza) if lunghezza else b"{}"
                dati = json.loads(corpo.decode("utf-8") or "{}")
            except Exception as exc:  # corpo mancante o JSON non valido
                self._json(400, {"errore": f"corpo JSON non valido: {exc}"})
                return

            if self.path == "/api/valuta":
                try:
                    parametri = dati.get("parametri") or {}
                    gate = dati.get("gate", gate_default)
                    if not isinstance(gate, bool):
                        gate = bool(gate)
                    traccia = trace_analysis(parametri, agente, gate=gate)
                    self._json(200, traccia)
                except Exception as exc:  # mai traceback verso il client
                    self._json(500, {"errore": str(exc)})
                return

            if self.path in ("/api/allena/avvia", "/api/allena/step", "/api/allena/reset"):
                try:
                    import allena_live

                    if self.path == "/api/allena/avvia":
                        esito = allena_live.avvia_sessione(
                            n_hidden=dati.get("n_hidden", 16),
                            lr=dati.get("lr", 0.05),
                            batch_size=dati.get("batch_size", 32),
                            n_campioni=dati.get("n_campioni", 2000),
                            max_epoche=dati.get("max_epoche", 20),
                        )
                    elif self.path == "/api/allena/step":
                        esito = allena_live.step_sessione(int(dati.get("n_passi", 1)))
                    else:
                        esito = allena_live.reset_sessione()
                    self._json(200, esito)
                except FileNotFoundError:
                    self._json(500, {
                        "errore": "dataset non trovato: eseguire prima "
                                  "python training/generate_dataset.py"
                    })
                except Exception as exc:  # mai traceback verso il client
                    self._json(500, {"errore": str(exc)})
                return

            self._json(404, {"errore": "risorsa non trovata"})

        def _json(self, codice, oggetto):
            corpo = json.dumps(oggetto, ensure_ascii=False).encode("utf-8")
            self.send_response(codice)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(corpo)))
            self.end_headers()
            self.wfile.write(corpo)

        def _api_analisi(self):
            try:
                analisi, notifiche, _ = _leggi_dati(db_path)
                timestamp_notifiche = {n.get("timestamp") for n in notifiche}
                record = [_record_json(r, timestamp_notifiche) for r in analisi]
                corpo = json.dumps(
                    {"soglia_incertezza": SOGLIA_INCERTEZZA, "record": record},
                    ensure_ascii=False,
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(corpo)))
                self.end_headers()
                self.wfile.write(corpo)
            except Exception as exc:
                corpo = json.dumps(
                    {"errore": str(exc)}, ensure_ascii=False
                ).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(corpo)))
                self.end_headers()
                self.wfile.write(corpo)

    return Handler


def _serve(
    out_path: Path,
    db_path: Path,
    ferma: Optional[threading.Event] = None,
    gate_default: bool = True,
) -> int:
    """Serve il file generato + endpoint /api/analisi (sola lettura).

    Gestione avvio/chiusura:
      - SIGINT (Ctrl+C) e SIGTERM arrestano il server con un messaggio
        pulito (nessun traceback): il gestore imposta l'evento ``ferma`` e il
        ciclo di ``serve_forever`` (thread daemon) viene chiuso via
        ``server.shutdown()``;
      - porta già occupata -> errore amichevole e codice di uscita 1;
      - ``ferma`` è iniettabile nei test (Event impostato da un altro
        thread) per verificare la chiusura pulita senza segnali.
    """
    import http.server
    import signal

    if ferma is None:
        ferma = threading.Event()
    if threading.current_thread() is threading.main_thread():

        def _arresta(_segno, _frame):
            ferma.set()

        signal.signal(signal.SIGINT, _arresta)
        signal.signal(signal.SIGTERM, _arresta)
        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, _arresta)

    porta = int(os.environ.get("PORT", "8000"))
    handler = crea_handler(
        db_path, out_path.parent, agent=SupervisedAgent(), gate_default=gate_default
    )
    try:
        server = http.server.ThreadingHTTPServer(("127.0.0.1", porta), handler)
    except OSError as exc:
        print(
            f"ERRORE: impossibile avviare il server sulla porta {porta}: {exc}",
            file=sys.stderr,
        )
        print(
            "Porta già in uso? Chiudi l'istanza precedente oppure usa "
            f"PORT=<altra> python tools/genera_dashboard.py --serve",
            file=sys.stderr,
        )
        return 1
    with server:
        porta_reale = server.server_address[1]
        print(
            f"Dashboard disponibile su http://127.0.0.1:{porta_reale}/{out_path.name}"
        )
        print(f"API: http://127.0.0.1:{porta_reale}/api/analisi")
        print(f"API: http://127.0.0.1:{porta_reale}/api/valuta (POST)")
        print("Premere Ctrl+C per fermare il server.")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            ferma.wait()
        finally:
            server.shutdown()
            thread.join(timeout=2)
        print("Server arrestato: dashboard chiusa.")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    db_path = Path(args.db_path)
    out_path = Path(args.out_path)
    metadati_path = Path(args.metadati_path)
    report_path = Path(args.report_path)
    test_path = Path(args.test_path)

    try:
        analisi, notifiche, avviso = _leggi_dati(db_path)
        metadati, avviso_metadati = _leggi_metadati(metadati_path)
        report, avviso_report = _leggi_report(report_path)
        documento = genera_html(
            analisi, notifiche, avviso, metadati, avviso_metadati,
            report, avviso_report, test_path,
            gate_default=not args.no_gate,
        )
    except Exception as exc:  # ultima rete di sicurezza: mai traceback finale
        print(f"Errore durante la generazione della dashboard: {exc}", file=sys.stderr)
        documento = genera_html(
            [], [], f"Errore durante la generazione della dashboard: {exc}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(documento, encoding="utf-8")
    print(f"Dashboard generata: {out_path}")

    # Pagina "Allenamento dal vivo" (stessa directory: servita da --serve).
    try:
        import allena_live

        pagina_allena = out_path.parent / "allenamento.html"
        pagina_allena.write_text(
            allena_live.genera_html_allenamento(), encoding="utf-8"
        )
        print(f"Pagine generate: {pagina_allena}")
    except Exception as exc:  # la dashboard resta utilizzabile anche senza
        print(f"Allenamento live non generato: {exc}", file=sys.stderr)

    if args.serve:
        return _serve(out_path, db_path, gate_default=not args.no_gate)
    return 0


if __name__ == "__main__":
    sys.exit(main())