"""
test_end_to_end.py — Test end-to-end della Fase 7.

Percorso completo: CLI -> AnalysisService -> SupervisedAgent -> SQLite.

Coprono:
  - caso critico ('alto') da CLI: record in tabella `analisi` con
    allerta_medico=1 e notifica registrata (DB + stderr);
  - caso normale ('basso') da CLI: record scritto, nessuna notifica;
  - fallback senza artifact: record comunque scritto con probabilita NULL;
  - persistenza della probabilita quando l'MLP è usato;
  - database non scrivibile: fallimento RUMOROSO (ValueError), mai
    perdita silenziosa di record clinici;
  - contratto di ottieni_storico() invariato.

Nessun file viene creato fuori dal tmpdir dei test.
"""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import main as cli
from telemedicina_supervised.database.analisi_db import AnalisiDatabase
from telemedicina_supervised.ml.mlp import MLP
from telemedicina_supervised.ml.scaler import StandardScaler
from telemedicina_supervised.services.analysis_service import AnalysisService

BASSO = {
    "pressione_sistolica": 120,
    "pressione_diastolica": 80,
    "frequenza_cardiaca": 75,
    "temperatura": 36.8,
    "saturazione_ossigeno": 98,
    "glicemia": 95,
}

ALTO = {
    "pressione_sistolica": 185,
    "pressione_diastolica": 110,
    "frequenza_cardiaca": 80,
    "temperatura": 36.8,
    "saturazione_ossigeno": 98,
    "glicemia": 95,
}


def _artifact(path: Path) -> MLP:
    """Artifact valido che predice 'basso' con confidenza alta."""
    modello = MLP(6, 2, ("basso", "medio", "alto"), seed=41)
    modello.W1.fill(0.0)
    modello.W2.fill(0.0)
    modello.b2[:] = 0.0
    modello.b2[0] = 3.0
    modello.scaler = StandardScaler().fit(np.ones((3, 6)))
    modello.salva(path)
    return modello


class TestEndToEnd(unittest.TestCase):
    def test_cli_caso_alto_registra_record_e_notifica(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "analisi.db"
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = cli.main(
                    [
                        "--run-ml",
                        "--parametri", json.dumps(ALTO),
                        "--modello-path", str(root / "mancante.npz"),
                        "--db-path", str(db),
                    ]
                )
            self.assertEqual(code, 0)

            repo = AnalisiDatabase(db)
            analisi = repo.leggi_analisi()
            self.assertEqual(len(analisi), 1)
            record = analisi[0]
            self.assertEqual(record["classe"], "alto")
            self.assertTrue(record["allerta_medico"])
            self.assertEqual(record["errori"], [])
            self.assertIsNone(record["probabilita"])  # gate: MLP bypassato
            self.assertIn("modello_usato", record["metadati"])
            self.assertFalse(record["metadati"]["modello_usato"])
            self.assertTrue(record["messaggio"])
            self.assertTrue(record["timestamp"])

            notifiche = repo.leggi_notifiche()
            self.assertEqual(len(notifiche), 1)
            self.assertEqual(notifiche[0]["classe"], "alto")
            self.assertIn("[NOTIFICA]", err.getvalue())

    def test_cli_caso_basso_senza_notifica_e_fallback_null(self):
        # Fallback senza artifact: il record viene comunque scritto con
        # probabilita NULL e senza notifica (nessuna allerta).
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "analisi.db"
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = cli.main(
                    [
                        "--run-ml",
                        "--parametri", json.dumps(BASSO),
                        "--modello-path", str(root / "mancante.npz"),
                        "--db-path", str(db),
                    ]
                )
            self.assertEqual(code, 0)

            repo = AnalisiDatabase(db)
            analisi = repo.leggi_analisi()
            self.assertEqual(len(analisi), 1)
            record = analisi[0]
            self.assertEqual(record["classe"], "basso")
            self.assertFalse(record["allerta_medico"])
            self.assertIsNone(record["probabilita"])  # fallback: nessun MLP
            self.assertEqual(repo.leggi_notifiche(), [])
            self.assertNotIn("[NOTIFICA]", err.getvalue())

    def test_cli_caso_basso_con_modello_persiste_probabilita(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "analisi.db"
            modello = root / "modello.npz"
            _artifact(modello)
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = cli.main(
                    [
                        "--run-ml",
                        "--parametri", json.dumps(BASSO),
                        "--modello-path", str(modello),
                        "--db-path", str(db),
                    ]
                )
            self.assertEqual(code, 0)

            repo = AnalisiDatabase(db)
            record = repo.leggi_analisi()[0]
            self.assertEqual(record["classe"], "basso")
            self.assertIsNotNone(record["probabilita"])
            self.assertAlmostEqual(sum(record["probabilita"].values()), 1.0)
            self.assertTrue(record["metadati"]["modello_usato"])

    def test_db_non_scrivibile_fallisce_rumorosamente(self):
        # Parent directory inesistente: nessun record può essere perso in
        # silenzio -> ValueError con messaggio chiaro.
        with tempfile.TemporaryDirectory() as tmp:
            servizio = AnalysisService(db_path=Path(tmp) / "sotto" / "analisi.db")
            with self.assertRaises(ValueError) as ctx:
                servizio.analizza(BASSO)
            self.assertIn("database non scrivibile", str(ctx.exception))

    def test_servizio_contratto_storico_invariato(self):
        # Il contratto pubblico resta: analizza() e ottieni_storico()
        # funzionano come prima, con la persistenza in più.
        with tempfile.TemporaryDirectory() as tmp:
            servizio = AnalysisService(db_path=Path(tmp) / "analisi.db")
            servizio.analizza(BASSO)
            servizio.analizza(ALTO)
            self.assertEqual(len(servizio.ottieni_storico()), 2)
            self.assertEqual(servizio.ultima["classe"], "alto")
            repo = AnalisiDatabase(Path(tmp) / "analisi.db")
            self.assertEqual(len(repo.leggi_analisi()), 2)
            self.assertEqual(len(repo.leggi_notifiche()), 1)


if __name__ == "__main__":
    unittest.main()