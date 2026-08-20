#!/usr/bin/env python3
"""tools/genera_dashboard.py — Dashboard HTML statica del sistema supervised (Fase V3).

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
  - vista "6. Riproducibilità": seed, split e hash da ``metadati.json``;
  - banner didattico obbligatorio e 6 tab (la vista 5 è segnaposto).

matplotlib è importato LAZY (via ``tools/_figure.py``): senza la dipendenza
dev le figure diventano segnaposto con messaggio, mai un crash.

Uso:
    python tools/genera_dashboard.py                  # build statico
    python tools/genera_dashboard.py --out OUT        # output custom
    python tools/genera_dashboard.py --db-path DB     # database custom
    python tools/genera_dashboard.py --metadati-path M  # metadati custom
    python tools/genera_dashboard.py --report-path R  # report custom
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


def _carica_test_e_modello() -> Tuple[Optional[Any], Optional[Any], Optional[Any], Optional[str]]:
    """(X_test, y_test, modello) con degrado grazioso; modello espone .scaler."""
    try:
        import numpy as np
        from telemedicina_supervised.ml.mlp import MLP

        X_test = np.load(DATA_PROCESSED_DIR / "X_test.npy")
        y_test = np.load(DATA_PROCESSED_DIR / "y_test.npy")
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
    X_test, y_test, modello, avviso_dati = _carica_test_e_modello()
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


def genera_html(
    analisi: List[Dict[str, Any]],
    notifiche: List[Dict[str, Any]],
    avviso: Optional[str] = None,
    metadati: Optional[Dict[str, Any]] = None,
    avviso_metadati: Optional[str] = None,
    report: Optional[Dict[str, Any]] = None,
    avviso_report: Optional[str] = None,
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
            pannelli.append(_pannello_distillazione(report, avviso_report))
        elif chiave == "training":
            pannelli.append(_pannello_training(report, avviso_report))
        elif chiave == "riproducibilita":
            pannelli.append(_pannello_riproducibilita(metadati, avviso_metadati))
        else:
            pannelli.append(
                f'<section class="tab-panel" data-tab="{chiave}">'
                '<p class="segnaposto">In arrivo nelle fasi successive.</p>'
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
<footer>Dashboard generata localmente — nessuna richiesta di rete.</footer>
<script type="application/json" id="dati-analisi">{dati_json}</script>
<script>
{_js()}
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Genera la dashboard HTML statica del sistema supervised (Fase V3)"
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
        "--serve",
        action="store_true",
        help="dopo il build, serve il file su http://127.0.0.1:8000 (o variabile PORT)",
    )
    return parser


def crea_handler(db_path: Path, directory: Path):
    """Classe handler http: file statici da ``directory`` + GET /api/analisi.

    L'endpoint /api/analisi restituisce il JSON dei record (stessa struttura
    del JSON incorporato nell'HTML, inclusi ``percorso`` e ``riga_html``),
    letto dal DB in sola lettura. Esposta per i test (porta efimera).
    """
    import http.server

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

        def do_GET(self):
            if self.path == "/api/analisi":
                self._api_analisi()
                return
            super().do_GET()

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


def _serve(out_path: Path, db_path: Path) -> None:
    """Serve il file generato + endpoint /api/analisi (sola lettura)."""
    import http.server

    porta = int(os.environ.get("PORT", "8000"))
    handler = crea_handler(db_path, out_path.parent)
    with http.server.ThreadingHTTPServer(("127.0.0.1", porta), handler) as server:
        porta_reale = server.server_address[1]
        print(
            f"Dashboard disponibile su http://127.0.0.1:{porta_reale}/{out_path.name}"
        )
        print(f"API: http://127.0.0.1:{porta_reale}/api/analisi")
        print("Premere Ctrl+C per fermare il server.")
        server.serve_forever()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    db_path = Path(args.db_path)
    out_path = Path(args.out_path)
    metadati_path = Path(args.metadati_path)
    report_path = Path(args.report_path)

    try:
        analisi, notifiche, avviso = _leggi_dati(db_path)
        metadati, avviso_metadati = _leggi_metadati(metadati_path)
        report, avviso_report = _leggi_report(report_path)
        documento = genera_html(
            analisi, notifiche, avviso, metadati, avviso_metadati,
            report, avviso_report,
        )
    except Exception as exc:  # ultima rete di sicurezza: mai traceback finale
        print(f"Errore durante la generazione della dashboard: {exc}", file=sys.stderr)
        documento = genera_html(
            [], [], f"Errore durante la generazione della dashboard: {exc}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(documento, encoding="utf-8")
    print(f"Dashboard generata: {out_path}")

    if args.serve:
        _serve(out_path, db_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())