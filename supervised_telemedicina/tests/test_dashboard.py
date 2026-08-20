"""tests/test_dashboard.py — Guard e test della dashboard (Fase V1 + V2 + V3 + V4).

Verifica che:
- il runtime (src/, training/, main.py) non importi matplotlib né il tool;
- il tool generi l'HTML con badge corretti, banner, tab e senza link esterni;
- il degrado grazioso funzioni per DB inesistente, DB vuoto e DB corrotto;
- la Vista 6 (Riproducibilità) mostri seed/split/hash da metadati.json;
- la Vista 2 (Flusso decisionale) incorpori il JSON dei record con i
  percorsi corretti e la struttura JS del flusso;
- l'endpoint /api/analisi risponda con il JSON dei record (200 e 500);
- le viste 3-4 (figure matplotlib) degradino a segnaposto senza matplotlib
  e mostrino figure base64 + numeri da report.json con matplotlib;
- la Vista 5 (Incertezza) mostri contatori da report.json, il testo di
  sicurezza "1.0 per costruzione", la tabella dei mancati ricalcolata sul
  test congelato (anche con dataset finto via --test-path) e l'istogramma
  della confidenza con matplotlib;
- nessun segnaposto residuo: i 6 tab sono tutti implementati.

Nessuna dipendenza da matplotlib: tutto gira con stdlib + numpy.
"""

import io
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stderr
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = PROJECT_ROOT / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import genera_dashboard  # noqa: E402
from telemedicina_supervised.database.analisi_db import AnalisiDatabase  # noqa: E402


def _matplotlib_disponibile() -> bool:
    try:
        import matplotlib  # noqa: F401

        return True
    except Exception:
        return False


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


def _estrai_json(html_doc: str) -> dict:
    """Estrae e decodifica il JSON incorporato (id="dati-analisi")."""
    marker = 'id="dati-analisi">'
    inizio = html_doc.index(marker) + len(marker)
    fine = html_doc.index("</script>", inizio)
    return json.loads(html_doc[inizio:fine])


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

            # 6 tab, tutti implementati (nessun segnaposto residuo).
            for etichetta in (
                "1. Analisi Live",
                "2. Flusso decisionale",
                "3. Teacher vs MLP",
                "4. Training",
                "5. Incertezza",
                "6. Riproducibilità",
            ):
                self.assertIn(etichetta, html_doc)
            self.assertNotIn("In arrivo nelle fasi successive", html_doc)

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
            # Vista Live vuota: nessuna riga dati cliccabile (il pannello
            # Riproducibilità può avere un tbody dai metadati reali).
            self.assertNotIn("data-id=", html_doc)

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
            self.assertNotIn("data-id=", html_doc)


class TestVista6Riproducibilita(unittest.TestCase):
    def test_metadati_finti_mostrati_nel_tab(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = tmp_path / "analisi.db"
            out_path = tmp_path / "dash.html"
            metadati_path = tmp_path / "metadati.json"
            _crea_db_con_casi(db_path)
            metadati_path.write_text(
                json.dumps({
                    "seed_base": 7,
                    "blocchi": {
                        "train": {"seed": 7, "n": 111},
                        "val": {"seed": 8, "n": 222},
                        "test": {"seed": 9, "n": 333},
                    },
                    "hash_safety_rules": "hashA",
                    "hash_synthetic_generator": "hashB",
                    "hash_teacher_rules": "hashC",
                }),
                encoding="utf-8",
            )
            self.assertEqual(
                genera_dashboard.main([
                    "--db-path", str(db_path),
                    "--out", str(out_path),
                    "--metadati-path", str(metadati_path),
                ]),
                0,
            )
            html_doc = out_path.read_text(encoding="utf-8")
            self.assertIn("Report deterministico", html_doc)
            # Etichette e valori espliciti della Vista 6 (nota oracle V2).
            self.assertIn("Seed base", html_doc)
            self.assertIn(">7<", html_doc)  # seed_base del metadati finto
            self.assertIn("Split del dataset", html_doc)
            self.assertIn("Hash di provenienza", html_doc)
            for etichetta in ("Train", "Validazione", "Test"):
                self.assertIn(etichetta, html_doc)
            self.assertIn("111", html_doc)
            self.assertIn("222", html_doc)
            self.assertIn("333", html_doc)
            self.assertIn("hashA", html_doc)
            self.assertIn("hashB", html_doc)
            self.assertIn("hashC", html_doc)

    def test_metadati_mancanti_degrado(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = tmp_path / "analisi.db"
            out_path = tmp_path / "dash.html"
            _crea_db_con_casi(db_path)
            self.assertEqual(
                genera_dashboard.main([
                    "--db-path", str(db_path),
                    "--out", str(out_path),
                    "--metadati-path", str(tmp_path / "mancante.json"),
                ]),
                0,
            )
            html_doc = out_path.read_text(encoding="utf-8")
            self.assertIn("Metadati non trovati", html_doc)


class TestVista2Flusso(unittest.TestCase):
    def test_json_incorporato_e_struttura_flusso(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = tmp_path / "analisi.db"
            out_path = tmp_path / "dash.html"
            _crea_db_con_casi(db_path)
            self.assertEqual(
                genera_dashboard.main(
                    ["--db-path", str(db_path), "--out", str(out_path)]
                ),
                0,
            )
            html_doc = out_path.read_text(encoding="utf-8")

            # JSON incorporato valido, con soglia da config e percorsi corretti.
            dati = _estrai_json(html_doc)
            self.assertEqual(dati["soglia_incertezza"], 0.6)
            record = {r["id"]: r for r in dati["record"]}
            self.assertEqual(record[1]["percorso"], "gate")
            self.assertEqual(record[2]["percorso"], "mlp")
            self.assertEqual(record[3]["percorso"], "fallback")
            self.assertIn(
                "safety gate: classe critica",
                record[1]["metadati"]["motivo_fallback"],
            )
            self.assertIn("riga_html", record[1])

            # Struttura del flusso presente (contenitore + nodi nel JS).
            self.assertIn('id="flusso-decisionale"', html_doc)
            self.assertIn("renderFlusso", html_doc)
            for nodo in (
                "Safety gate",
                "MLP softmax",
                "Soglia di incertezza",
                "Regola max(rule, MLP)",
                "Persistenza DB + notifica",
            ):
                self.assertIn(nodo, html_doc)

            # Righe cliccabili (data-id) e pulsante Aggiorna.
            self.assertIn('data-id="1"', html_doc)
            self.assertIn('id="btn-aggiorna"', html_doc)


class TestCasiAggiuntivi(unittest.TestCase):
    def test_mlp_fallback_insieme(self):
        # modello_usato=True E confidenza sotto soglia: badge MLP + FALLBACK,
        # percorso "fallback" (il flusso mostra MLP raggiunto e soglia fallita).
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = tmp_path / "analisi.db"
            out_path = tmp_path / "dash.html"
            timestamp = "2026-08-19T11:00:00.000000+00:00"
            db = AnalisiDatabase(db_path)
            db.inserisci_analisi({
                "timestamp": timestamp,
                "classe": "medio",
                "allerta_medico": False,
                "errori": [],
                "probabilita": {"basso": 0.30, "medio": 0.42, "alto": 0.28},
                "metadati": {
                    "modello_usato": True,
                    "classe_mlp": "medio",
                    "classe_regola": "medio",
                    "confidenza": 0.42,
                    "fallback": True,
                    "motivo_fallback": "confidenza sotto la soglia di incertezza",
                    "override_sicurezza": False,
                },
                "messaggio": "Incertezza",
            })
            db.chiudi()
            self.assertEqual(
                genera_dashboard.main(
                    ["--db-path", str(db_path), "--out", str(out_path)]
                ),
                0,
            )
            html_doc = out_path.read_text(encoding="utf-8")
            riga = _riga_per_timestamp(html_doc, timestamp)
            self.assertIn("badge-mlp", riga)
            self.assertIn("badge-fallback", riga)
            dati = _estrai_json(html_doc)
            self.assertEqual(dati["record"][0]["percorso"], "fallback")

    def test_gate_case_insensitive(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = tmp_path / "analisi.db"
            out_path = tmp_path / "dash.html"
            timestamp = "2026-08-19T11:30:00.000000+00:00"
            db = AnalisiDatabase(db_path)
            db.inserisci_analisi({
                "timestamp": timestamp,
                "classe": "alto",
                "allerta_medico": True,
                "errori": [],
                "probabilita": None,
                "metadati": {
                    "modello_usato": False,
                    "fallback": True,
                    "motivo_fallback": "SAFETY GATE: classe critica, MLP bypassato",
                    "override_sicurezza": False,
                },
                "messaggio": "Critico",
            })
            db.chiudi()
            self.assertEqual(
                genera_dashboard.main(
                    ["--db-path", str(db_path), "--out", str(out_path)]
                ),
                0,
            )
            html_doc = out_path.read_text(encoding="utf-8")
            riga = _riga_per_timestamp(html_doc, timestamp)
            self.assertIn("badge-gate", riga)
            self.assertNotIn("badge-fallback", riga)
            dati = _estrai_json(html_doc)
            self.assertEqual(dati["record"][0]["percorso"], "gate")

    def test_db_corrotto_degrado(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = tmp_path / "corrotto.db"
            db_path.write_bytes(b"questo non e' un database sqlite")
            out_path = tmp_path / "dash.html"
            self.assertEqual(
                genera_dashboard.main(
                    ["--db-path", str(db_path), "--out", str(out_path)]
                ),
                0,
            )
            html_doc = out_path.read_text(encoding="utf-8")
            self.assertIn("Errore durante la lettura", html_doc)


class TestApiAnalisi(unittest.TestCase):
    def test_api_analisi_risponde_json(self):
        import http.server
        import threading
        import urllib.request

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = tmp_path / "analisi.db"
            _crea_db_con_casi(db_path)
            handler = genera_dashboard.crea_handler(db_path, tmp_path)
            server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                porta = server.server_address[1]
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{porta}/api/analisi", timeout=5
                ) as risposta:
                    self.assertEqual(risposta.status, 200)
                    dati = json.loads(risposta.read().decode("utf-8"))
                self.assertEqual(dati["soglia_incertezza"], 0.6)
                self.assertEqual(len(dati["record"]), 3)
                self.assertEqual(
                    {r["percorso"] for r in dati["record"]},
                    {"gate", "mlp", "fallback"},
                )
                self.assertIn("riga_html", dati["record"][0])
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_api_analisi_errore_500(self):
        # Ramo 500: la lettura del DB fallisce -> risposta JSON con "errore".
        # _leggi_dati degrada internamente quasi tutti gli errori, quindi il
        # ramo 500 è difensivo: lo si forza con un mock per verificare il
        # contratto HTTP (500 + corpo JSON).
        import http.server
        import threading
        import urllib.error
        import urllib.request
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = tmp_path / "analisi.db"
            _crea_db_con_casi(db_path)
            handler = genera_dashboard.crea_handler(db_path, tmp_path)
            server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                porta = server.server_address[1]
                with mock.patch.object(
                    genera_dashboard,
                    "_leggi_dati",
                    side_effect=RuntimeError("lettura fallita"),
                ):
                    with self.assertRaises(urllib.error.HTTPError) as ctx:
                        urllib.request.urlopen(
                            f"http://127.0.0.1:{porta}/api/analisi", timeout=5
                        )
                self.assertEqual(ctx.exception.code, 500)
                corpo = json.loads(ctx.exception.read().decode("utf-8"))
                self.assertIn("errore", corpo)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)


class TestFigure(unittest.TestCase):
    def test_placeholder_senza_matplotlib(self):
        # Subprocess con matplotlib bloccato (modulo finto che alza
        # ImportError): le viste 3-4 degradano a segnaposto, mai un crash.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "matplotlib.py").write_text(
                "raise ImportError('matplotlib bloccato per il test')\n",
                encoding="utf-8",
            )
            db_path = tmp_path / "analisi.db"
            out_path = tmp_path / "dash.html"
            _crea_db_con_casi(db_path)
            env = dict(os.environ)
            env["PYTHONPATH"] = str(tmp_path) + os.pathsep + env.get("PYTHONPATH", "")
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
                env=env,
            )
            self.assertEqual(risultato.returncode, 0, risultato.stderr)
            html_doc = out_path.read_text(encoding="utf-8")
            # Nessuna figura base64, ma i pannelli esistono con messaggio.
            self.assertNotIn("data:image/png;base64,", html_doc)
            self.assertIn("Knowledge distillation", html_doc)
            self.assertIn("Curve di loss", html_doc)
            self.assertIn("non disponibile", html_doc)

    @unittest.skipUnless(
        _matplotlib_disponibile(), "matplotlib non installato"
    )
    def test_figure_con_matplotlib(self):
        # Con matplotlib: figure base64 presenti e numeri del banner letti
        # dal report finto (--report-path), mai hardcoded.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = tmp_path / "analisi.db"
            out_path = tmp_path / "dash.html"
            report_path = tmp_path / "report.json"
            _crea_db_con_casi(db_path)
            report_path.write_text(
                json.dumps({
                    "config_migliore": {
                        "n_hidden": 32, "lr": 0.05, "batch_size": 32, "epoche": 25,
                    },
                    "loss_train_finale": [0.5 - i * 0.01 for i in range(25)],
                    "loss_val_finale": [0.55 - i * 0.01 for i in range(25)],
                    "metriche_test_congelato": {
                        "accuracy": 0.1234,
                        "kappa_cohen": 0.5678,
                        "per_classe": {"2": {"recall": 0.9999}},
                        "matrice_confusione": [[10, 0, 0], [0, 10, 0], [0, 0, 10]],
                        "sicurezza": {"recall_alto": 0.9999},
                    },
                    "metriche_test_no_buffer": {"accuracy": 0.4321},
                    "analisi_errori_distanza": {"n_errori": 5},
                }),
                encoding="utf-8",
            )
            self.assertEqual(
                genera_dashboard.main([
                    "--db-path", str(db_path),
                    "--out", str(out_path),
                    "--report-path", str(report_path),
                ]),
                0,
            )
            html_doc = out_path.read_text(encoding="utf-8")
            self.assertIn("data:image/png;base64,", html_doc)
            # Numeri del banner dal report finto (formattati a 4 decimali).
            self.assertIn("0.1234", html_doc)
            self.assertIn("0.4321", html_doc)
            self.assertIn("0.5678", html_doc)
            self.assertIn("0.9999", html_doc)
            self.assertIn("Gradient check ~1e-10", html_doc)


def _valore_metriche(html_doc: str, etichetta: str) -> str:
    """Valore della metriche-card con l'etichetta data (regex)."""
    import re

    m = re.search(
        r'<span class="metriche-valore">([^<]+)</span>'
        r'<span class="metriche-etichetta">' + re.escape(etichetta) + r"</span>",
        html_doc,
    )
    if not m:
        raise AssertionError(f"card metriche non trovata: {etichetta}")
    return m.group(1)


class TestVista5Incertezza(unittest.TestCase):
    """Vista 5: contatori da report.json, testo di sicurezza, tabella mancati.

    La tabella dei mancati è RICALCOLATA sul test congelato (pred MLP vs
    label): il conteggio atteso è quello reale del dataset, mai un valore
    fisso. Con --test-path si può puntare a un dataset finto in tmpdir.
    """

    def _build(
        self,
        tmp_path: Path,
        report_path: Path = None,
        test_path: Path = None,
    ) -> str:
        db_path = tmp_path / "analisi.db"
        out_path = tmp_path / "dash.html"
        _crea_db_con_casi(db_path)
        args = ["--db-path", str(db_path), "--out", str(out_path)]
        if report_path is not None:
            args += ["--report-path", str(report_path)]
        if test_path is not None:
            args += ["--test-path", str(test_path)]
        self.assertEqual(genera_dashboard.main(args), 0)
        return out_path.read_text(encoding="utf-8")

    def test_contenuti_vista5_con_dati_reali(self):
        # Dati reali (deterministici): contatori da report.json, testo di
        # sicurezza, tabella mancati con il conteggio REALE ricalcolato.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            html_doc = self._build(tmp_path)

            self.assertIn("1.0 per costruzione", html_doc)
            self.assertIn("non registrato nel report", html_doc)
            self.assertEqual(
                _valore_metriche(html_doc, "Casi critici nel test (classe alto)"),
                "7100",
            )
            self.assertEqual(
                _valore_metriche(html_doc, "Override di sicurezza"), "0"
            )
            self.assertEqual(
                _valore_metriche(html_doc, "Notifiche totali (DB)"), "1"
            )

            # Conteggio reale dei mancati: ricalcolo indipendente nel test.
            import numpy as np
            from telemedicina_supervised.ml.mlp import MLP

            X_test = np.load(PROJECT_ROOT / "data" / "processed" / "X_test.npy")
            y_test = np.load(PROJECT_ROOT / "data" / "processed" / "y_test.npy")
            modello = MLP.carica(
                PROJECT_ROOT / "data" / "models" / "mlp_telemedicina.npz"
            )
            pred = modello.predici_etichette(modello.scaler.transform(X_test))
            attesi = int(np.count_nonzero((y_test == "alto") & (pred != "alto")))
            corpo = html_doc.split('id="tabella-mancati"')[1]
            righe = corpo.split("<tbody>")[1].split("</tbody>")[0]
            self.assertEqual(righe.count("<tr>"), attesi)
            self.assertEqual(
                _valore_metriche(
                    html_doc, "Mancati dell'MLP sul test congelato (ricalcolati)"
                ),
                str(attesi),
            )

            # Nessun segnaposto residuo in nessun tab.
            self.assertNotIn("In arrivo nelle fasi successive", html_doc)

    @unittest.skipUnless(
        _matplotlib_disponibile(), "matplotlib non installato"
    )
    def test_istogramma_confidenze_con_matplotlib(self):
        # Con matplotlib: 6 figure totali (5 della V3 + istogramma V5) e
        # didascalia dell'istogramma presente.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            html_doc = self._build(tmp_path)
            self.assertEqual(html_doc.count("data:image/png;base64,"), 6)
            self.assertIn("Istogramma della confidenza", html_doc)

    def test_degrado_vista5_senza_dataset(self):
        # --test-path vuoto: la tabella mancati degrada, i contatori restano.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            vuoto = tmp_path / "vuoto"
            vuoto.mkdir()
            html_doc = self._build(tmp_path, test_path=vuoto)
            self.assertIn("Dati di test o artifact non disponibili", html_doc)
            self.assertNotIn("tabella-mancati", html_doc)
            self.assertIn("1.0 per costruzione", html_doc)
            self.assertEqual(
                _valore_metriche(html_doc, "Casi critici nel test (classe alto)"),
                "7100",
            )

    def test_tabella_mancati_con_dataset_finto(self):
        # Dataset finto in tmpdir (--test-path): 8 casi tutti 'alto' con
        # valori sani -> l'MLP li predice tutti 'basso' -> 8 mancati reali.
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            test_path = tmp_path / "test_finto"
            test_path.mkdir()
            np.save(
                test_path / "X_test.npy",
                np.full((8, 6), [120.0, 80.0, 75.0, 36.8, 98.0, 95.0]),
            )
            np.save(
                test_path / "y_test.npy",
                np.array(["alto"] * 8),
            )
            report_path = tmp_path / "report.json"
            report_path.write_text(
                json.dumps({
                    "metriche_test_congelato": {},
                    "distribuzione_classi": {},
                }),
                encoding="utf-8",
            )
            html_doc = self._build(
                tmp_path, report_path=report_path, test_path=test_path
            )
            corpo = html_doc.split('id="tabella-mancati"')[1]
            righe = corpo.split("<tbody>")[1].split("</tbody>")[0]
            self.assertEqual(righe.count("<tr>"), 8)
            self.assertEqual(
                _valore_metriche(
                    html_doc, "Mancati dell'MLP sul test congelato (ricalcolati)"
                ),
                "8",
            )
            self.assertIn("1.0 per costruzione", html_doc)
            # Contatori con report finto senza distribuzione: "—".
            self.assertEqual(
                _valore_metriche(html_doc, "Casi critici nel test (classe alto)"),
                "—",
            )


class TestServeAvvioChiusura(unittest.TestCase):
    def test_serve_avvia_e_chiude_pulito(self):
        # _serve con Event iniettabile: parte, risponde e si chiude senza
        # eccezioni ne' traceback (stesso codice usato da Ctrl+C/SIGTERM).
        # Porta libera via PORT (la 8000 di default può essere occupata).
        import socket
        import threading
        import urllib.request
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = tmp_path / "analisi.db"
            _crea_db_con_casi(db_path)
            out_path = tmp_path / "dashboard.html"
            out_path.write_text("<!doctype html><html></html>", encoding="utf-8")

            libera = socket.socket()
            libera.bind(("127.0.0.1", 0))
            porta = libera.getsockname()[1]
            libera.close()

            ferma = threading.Event()
            esito = []

            def _avvia():
                esito.append(
                    genera_dashboard._serve(out_path, db_path, ferma)
                )

            with mock.patch.dict(
                os.environ, {"PORT": str(porta)}, clear=False
            ):
                thread = threading.Thread(target=_avvia, daemon=True)
                thread.start()
                try:
                    # Attende che il server risponda (max ~5s).
                    risposta = None
                    for _i in range(50):
                        try:
                            with urllib.request.urlopen(
                                f"http://127.0.0.1:{porta}/dashboard.html",
                                timeout=1,
                            ) as r:
                                risposta = r.status
                            break
                        except Exception:
                            time.sleep(0.1)
                    self.assertEqual(risposta, 200)
                    # Chiusura pulita: nessuna eccezione nel thread, uscita 0.
                    ferma.set()
                    thread.join(timeout=5)
                    self.assertFalse(thread.is_alive())
                    self.assertEqual(esito, [0])
                finally:
                    ferma.set()
                    thread.join(timeout=5)

    def test_serve_porta_occupata_errore_amichevole(self):
        # Porta già occupata -> codice 1 e messaggio chiaro, non un traceback.
        import socket
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = tmp_path / "analisi.db"
            _crea_db_con_casi(db_path)
            out_path = tmp_path / "dashboard.html"
            out_path.write_text("<!doctype html>", encoding="utf-8")

            occupato = socket.socket()
            occupato.bind(("127.0.0.1", 0))
            occupato.listen(1)
            try:
                porta = occupato.getsockname()[1]
                with mock.patch.dict(
                    os.environ, {"PORT": str(porta)}, clear=False
                ):
                    with redirect_stderr(io.StringIO()) as err:
                        esito = genera_dashboard._serve(out_path, db_path)
                self.assertEqual(esito, 1)
                self.assertIn("Porta già in uso", err.getvalue())
            finally:
                occupato.close()


if __name__ == "__main__":
    unittest.main()