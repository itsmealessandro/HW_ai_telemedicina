"""tests/test_dashboard.py — Guard e test della dashboard (Fase V1 + V2 + V3).

Verifica che:
- il runtime (src/, training/, main.py) non importi matplotlib né il tool;
- il tool generi l'HTML con badge corretti, banner, tab e senza link esterni;
- il degrado grazioso funzioni per DB inesistente, DB vuoto e DB corrotto;
- la Vista 6 (Riproducibilità) mostri seed/split/hash da metadati.json;
- la Vista 2 (Flusso decisionale) incorpori il JSON dei record con i
  percorsi corretti e la struttura JS del flusso;
- l'endpoint /api/analisi risponda con il JSON dei record (200 e 500);
- le viste 3-4 (figure matplotlib) degradino a segnaposto senza matplotlib
  e mostrino figure base64 + numeri da report.json con matplotlib.

Nessuna dipendenza da matplotlib: tutto gira con stdlib + numpy.
"""

import json
import os
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


if __name__ == "__main__":
    unittest.main()