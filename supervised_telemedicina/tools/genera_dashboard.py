#!/usr/bin/env python3
"""tools/genera_dashboard.py — Dashboard HTML statica del sistema supervised (Fase V1).

Genera un file HTML autonomo (CSS e JS inline, nessun CDN, nessuna richiesta
di rete) con la vista "1. Analisi Live": tabella delle analisi da
``analisi.db`` con badge distintivi (MLP / GATE / FALLBACK / NOTIFICA),
banner didattico obbligatorio e 6 tab (le viste 2-6 sono segnaposto).

Solo stdlib in questa fase: matplotlib NON viene importato (le figure
arriveranno nelle fasi successive, vedi ``requirements_dashboard.txt``).

Uso:
    python tools/genera_dashboard.py                  # build statico
    python tools/genera_dashboard.py --out OUT        # output custom
    python tools/genera_dashboard.py --db-path DB     # database custom
    python tools/genera_dashboard.py --serve          # build + http.server

Il tool è eseguibile da qualsiasi CWD: i percorsi di default (DB e output)
sono derivati da ``__file__``, non dalla directory corrente.
"""

import argparse
import html
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Percorso assoluto a src/ (il tool vive in tools/, un livello sotto la root).
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from telemedicina_supervised.config import DB_PATH  # noqa: E402
from telemedicina_supervised.database.analisi_db import AnalisiDatabase  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DEFAULT = PROJECT_ROOT / "data" / "dashboard.html"

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
        if isinstance(motivo, str) and motivo.startswith("safety gate"):
            badge.append("gate")
        else:
            badge.append("fallback")
    return badge


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
        "<tr>"
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
footer {
  padding: 16px 32px;
  font-size: 12px;
  color: #6c757d;
}
"""


def _js() -> str:
    return """\
(function () {
  var barra = document.querySelector('.tab-bar');
  if (!barra) { return; }
  barra.addEventListener('click', function (evento) {
    var bottone = evento.target.closest('.tab');
    if (!bottone) { return; }
    var bersaglio = bottone.getAttribute('data-tab');
    document.querySelectorAll('.tab').forEach(function (b) {
      b.classList.toggle('active', b === bottone);
    });
    document.querySelectorAll('.tab-panel').forEach(function (p) {
      p.classList.toggle('active', p.getAttribute('data-tab') === bersaglio);
    });
  });
})();
"""


def genera_html(
    analisi: List[Dict[str, Any]],
    notifiche: List[Dict[str, Any]],
    avviso: Optional[str] = None,
) -> str:
    """Costruisce il documento HTML completo (tabella Live server-side)."""
    timestamp_notifiche = {n.get("timestamp") for n in notifiche}
    righe = "".join(_riga_html(r, timestamp_notifiche) for r in analisi)
    corpo_tabella = f"<tbody>{righe}</tbody>" if analisi else ""
    avviso_html = (
        f'<div class="avviso">{html.escape(avviso)}</div>' if avviso else ""
    )

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
                '<table class="tabella-live">'
                "<thead><tr>"
                "<th>Timestamp</th><th>Classe</th><th>Allerta medico</th>"
                "<th>Errori</th><th>Probabilità</th><th>Messaggio</th>"
                "<th>Motivo fallback</th><th>Override sicurezza</th><th>Badge</th>"
                "</tr></thead>"
                f"{corpo_tabella}"
                "</table></section>"
            )
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
        description="Genera la dashboard HTML statica del sistema supervised (Fase V1)"
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
        "--serve",
        action="store_true",
        help="dopo il build, serve il file su http://127.0.0.1:8000 (o variabile PORT)",
    )
    return parser


def _serve(out_path: Path) -> None:
    """Serve il file statico generato (solo file, nessuna logica applicativa)."""
    import functools
    import http.server

    porta = int(os.environ.get("PORT", "8000"))
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(out_path.parent)
    )
    with http.server.ThreadingHTTPServer(("127.0.0.1", porta), handler) as server:
        print(f"Dashboard disponibile su http://127.0.0.1:{porta}/{out_path.name}")
        print("Premere Ctrl+C per fermare il server.")
        server.serve_forever()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    db_path = Path(args.db_path)
    out_path = Path(args.out_path)

    try:
        analisi, notifiche, avviso = _leggi_dati(db_path)
        documento = genera_html(analisi, notifiche, avviso)
    except Exception as exc:  # ultima rete di sicurezza: mai traceback finale
        print(f"Errore durante la generazione della dashboard: {exc}", file=sys.stderr)
        documento = genera_html(
            [], [], f"Errore durante la generazione della dashboard: {exc}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(documento, encoding="utf-8")
    print(f"Dashboard generata: {out_path}")

    if args.serve:
        _serve(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())