"""tests/test_dashboard.py — Guard e test della dashboard (Fase V1).

Verifica che:
- il runtime (src/, training/, main.py) non importi matplotlib né il tool;
- il tool generi l'HTML con badge corretti, banner, tab e senza link esterni;
- il degrado grazioso funzioni per DB inesistente e DB vuoto.

Nessuna dipendenza da matplotlib: tutto gira con stdlib + numpy.
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = PROJECT_ROOT / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import genera_dashboard  # noqa: E402
from telemedicina_supervised.database.analisi_db import AnalisiDatabase  # noqa: E402


def _crea_db_con_casi(db_path: Path) -> None:
    """Popola il DB con i 3 casi di prova (A gate+notifica, B mlp, C fallback)."""
    db = AnalisiDatabase(db_path)
    db.inserisci_analisi({
        "timestamp": "2026-08-19T10:00:00.000000+00:00",
        "classe": "alto",
        "allerta_medico": True,
        "errori": [],
        "probabilita": None,
        "metadati": {
            "modello_usato": False,
            "fallback": True,
            "motivo_fallback": "safety gate: classe critica, MLP bypassato",
            "override_sicurezza": False,
        },
        "messaggio": "Rischio critico",
    })
    db.inserisci_notifica({
        "timestamp": "2026-08-19T10:00:00.000000+00:00",
        "classe": "alto",
        "messaggio": "Allerta",
    })
    db.inserisci_analisi({
        "timestamp": "2026-08-19T10:01:00.000000+00:00",
        "classe": "basso",
        "allerta_medico": False,
        "errori": [],
        "probabilita": {"basso": 0.99, "medio": 0.007, "alto": 0.003},
        "metadati": {
            "modello_usato": True,
            "fallback": False,
            "motivo_fallback": None,
            "override_sicurezza": False,
        },
        "messaggio": "OK",
    })
    db.inserisci_analisi({
        "timestamp": "2026-08-19T10:02:00.000000+00:00",
        "classe": "medio",
        "allerta_medico": False,
        "errori": [],
        "probabilita": None,
        "metadati": {
            "modello_usato": False,
            "fallback": True,
            "motivo_fallback": "confidenza sotto la soglia di incertezza",
            "override_sicurezza": False,
        },
        "messaggio": "Incertezza",
    })
    db.chiudi()


def _riga_per_timestamp(html_doc: str, timestamp: str) -> str:
    """Estrae il <tr> che contiene il timestamp dato (fallisce se assente)."""
    inizio = html_doc.find(timestamp)
    if inizio == -1:
        raise AssertionError(f"timestamp non trovato nel documento: {timestamp}")
    inizio_tr = html_doc.rfind("<tr>", 0, inizio)
    fine_tr = html_doc.find("</tr>", inizio)
    if inizio_tr == -1 or fine_tr == -1:
        raise AssertionError(f"riga non delimitata per il timestamp: {timestamp}")
    return html_doc[inizio_tr:fine_tr + len("</tr>")]


class TestGuardRuntime(unittest.TestCase):
    def test_runtime_senza_matplotlib_e_senza_tool(self):
        file_py = [PROJECT_ROOT / "main.py"]
        for directory in (PROJECT_ROOT / "src", PROJECT_ROOT / "training"):
            file_py.extend(p for p in directory.rglob("*.py") if p.is_file())
        self.assertTrue(file_py, "nessun file python da controllare")
        for percorso in file_py:
            contenuto = percorso.read_text(encoding="utf-8")
            self.assertNotIn(
                "matplotlib", contenuto, f"'matplotlib' presente in {percorso}"
            )
            # Solo pattern di import (non la substring nuda: una docstring
            # futura che menzioni "tools" non deve far scattare il guard).
            self.assertNotIn(
                "import tools", contenuto, f"'import tools' presente in {percorso}"
            )
            self.assertNotIn(
                "from tools", contenuto, f"'from tools' presente in {percorso}"
            )
            self.assertNotIn(
                "genera_dashboard",
                contenuto,
                f"'genera_dashboard' presente in {percorso}",
            )


class TestBuildBadge(unittest.TestCase):
    def test_build_con_badge_corretti(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = tmp_path / "analisi.db"
            out_path = tmp_path / "dash.html"
            # Stato di data/dashboard.html prima del build: può esistere da un
            # build reale precedente; il build di test non deve cambiarlo.
            dashboard_reale = PROJECT_ROOT / "data" / "dashboard.html"
            esisteva_prima = dashboard_reale.exists()
            _crea_db_con_casi(db_path)
            self.assertEqual(
                genera_dashboard.main(
                    ["--db-path", str(db_path), "--out", str(out_path)]
                ),
                0,
            )
            html_doc = out_path.read_text(encoding="utf-8")

            # Banner didattico obbligatorio.
            self.assertIn("nessuna pretesa diagnostica", html_doc)

            # 6 tab e testo segnaposto.
            for etichetta in (
                "1. Analisi Live",
                "2. Flusso decisionale",
                "3. Teacher vs MLP",
                "4. Training",
                "5. Incertezza",
                "6. Riproducibilità",
            ):
                self.assertIn(etichetta, html_doc)
            self.assertIn("In arrivo nelle fasi successive", html_doc)

            # Badge per riga (estratti dalla riga del timestamp, così i nomi
            # classe CSS non interferiscono con l'assert).
            riga_a = _riga_per_timestamp(
                html_doc, "2026-08-19T10:00:00.000000+00:00"
            )
            self.assertIn("badge-gate", riga_a)
            self.assertIn("badge-notifica", riga_a)
            riga_b = _riga_per_timestamp(
                html_doc, "2026-08-19T10:01:00.000000+00:00"
            )
            self.assertIn("badge-mlp", riga_b)
            riga_c = _riga_per_timestamp(
                html_doc, "2026-08-19T10:02:00.000000+00:00"
            )
            self.assertIn("badge-fallback", riga_c)

            # Nessun link esterno nel documento.
            self.assertNotIn("http://", html_doc)
            self.assertNotIn("https://", html_doc)
            self.assertNotIn("//cdn", html_doc)

            # Ordine visivo: leggi_analisi ordina id DESC -> C (più recente)
            # prima di B, B prima di A.
            pos_c = html_doc.find("2026-08-19T10:02:00.000000+00:00")
            pos_b = html_doc.find("2026-08-19T10:01:00.000000+00:00")
            pos_a = html_doc.find("2026-08-19T10:00:00.000000+00:00")
            self.assertLess(pos_c, pos_b)
            self.assertLess(pos_b, pos_a)

            # Nessun file scritto fuori dal tmpdir: lo stato di
            # data/dashboard.html è invariato rispetto a prima del build.
            self.assertEqual(dashboard_reale.exists(), esisteva_prima)


class TestDegrado(unittest.TestCase):
    def test_db_inesistente_via_subprocess(self):
        # Invocazione via subprocess con cwd arbitraria: il tool usa path
        # assoluti derivati da __file__, quindi funziona da qualsiasi CWD.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = tmp_path / "mancante.db"
            out_path = tmp_path / "dash.html"
            risultato = subprocess.run(
                [
                    sys.executable,
                    str(TOOLS_DIR / "genera_dashboard.py"),
                    "--db-path",
                    str(db_path),
                    "--out",
                    str(out_path),
                ],
                cwd=str(tmp_path),
                capture_output=True,
                text=True,
            )
            self.assertEqual(risultato.returncode, 0)
            html_doc = out_path.read_text(encoding="utf-8")
            self.assertIn("Database non trovato", html_doc)
            # Vista Live vuota: nessun tbody con righe dati.
            self.assertNotIn("<tbody>", html_doc)

    def test_db_vuoto(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = tmp_path / "vuoto.db"
            out_path = tmp_path / "dash.html"
            db = AnalisiDatabase(db_path)
            db.leggi_analisi()  # crea file e schema, nessun record
            db.chiudi()
            self.assertEqual(
                genera_dashboard.main(
                    ["--db-path", str(db_path), "--out", str(out_path)]
                ),
                0,
            )
            html_doc = out_path.read_text(encoding="utf-8")
            self.assertIn("Nessuna analisi registrata nel database", html_doc)
            self.assertNotIn("<tbody>", html_doc)


if __name__ == "__main__":
    unittest.main()