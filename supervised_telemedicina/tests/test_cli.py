import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import main as cli
from telemedicina_supervised.config import (
    DATA_PROCESSED_DIR,
    MODEL_ARTIFACT_PATH,
    REPORT_PATH,
    SEED_DEFAULT,
    SOGLIA_INCERTEZZA,
    carica_modello,
)
from telemedicina_supervised.ml.mlp import MLP
from telemedicina_supervised.ml.scaler import StandardScaler
from telemedicina_supervised.safety.safety_rules import FEATURE_ORDER


PARAMETRI = {
    "pressione_sistolica": 120,
    "pressione_diastolica": 80,
    "frequenza_cardiaca": 70,
    "temperatura": 36.7,
    "saturazione_ossigeno": 98,
    "glicemia": 100,
}


def _artifact(path: Path) -> MLP:
    """Artifact valido che predice 'basso' con confidenza alta."""
    modello = MLP(6, 2, ("basso", "medio", "alto"), seed=SEED_DEFAULT)
    modello.W1.fill(0.0)
    modello.W2.fill(0.0)
    modello.b2[:] = 0.0
    modello.b2[0] = 3.0
    modello.scaler = StandardScaler().fit(np.ones((3, 6)))
    modello.salva(path)
    return modello


class TestCLI(unittest.TestCase):
    def test_parser_mutua_esclusione_e_no_args(self):
        with self.assertRaises(SystemExit):
            cli.build_parser().parse_args(["--build-ml", "--run-ml"])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(cli.main([]), 0)
        self.assertIn("usage:", output.getvalue())

    def test_build_in_directory_temporanea(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, out = root / "processed", root / "models"
            data.mkdir()
            rng = np.random.default_rng(SEED_DEFAULT)
            for split, n in (("train", 12), ("val", 6), ("test", 6)):
                np.save(data / f"X_{split}.npy", rng.normal(size=(n, 6)))
                np.save(data / f"y_{split}.npy", np.array(["basso"] * n))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(["--build-ml", "--data-dir", str(data), "--out-dir", str(out)]), 0)
            self.assertTrue((out / "mlp_telemedicina.npz").exists())
            self.assertTrue((out / "report.json").exists())

    def test_runtime_valido_e_fallback(self):
        raw = json.dumps(PARAMETRI)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(cli.main(["--run-ml", "--parametri", raw, "--modello-path", "/missing/model.npz"]), 0)
        text = output.getvalue()
        self.assertIn("Classe:", text)
        self.assertIn("Raccomandazione:", text)
        self.assertIn("modello non trovato in /missing/model.npz; eseguire python main.py --build-ml", text)
        # La CLI passa dal AnalysisService (percorso applicativo unico).
        self.assertIn("[AnalysisService]", text)

    def test_runtime_alto_con_artifact(self):
        # Caso critico da CLI: il safety gate bypassa l'MLP e la risposta
        # resta identica a prima (classe, messaggio, metadati).
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.npz"
            _artifact(path)
            raw = json.dumps(dict(PARAMETRI, pressione_sistolica=185, pressione_diastolica=110))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(cli.main(["--run-ml", "--parametri", raw, "--modello-path", str(path)]), 0)
            text = output.getvalue()
            self.assertIn("Classe: alto", text)
            self.assertIn("Fallback rule-based: safety gate: classe critica, MLP bypassato", text)
            self.assertIn("Metadati:", text)
            self.assertIn('"override_sicurezza": false', text)

    def test_runtime_modello_usato(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.npz"
            _artifact(path)
            raw = json.dumps(PARAMETRI)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(cli.main(["--run-ml", "--parametri", raw, "--modello-path", str(path)]), 0)
            text = output.getvalue()
            self.assertIn("Modello MLP usato: si", text)
            self.assertIn("Probabilità:", text)
            self.assertIn("Classe: basso", text)

    def test_config_defaults_and_loader(self):
        self.assertEqual(SEED_DEFAULT, 41)
        self.assertEqual(SOGLIA_INCERTEZZA, 0.6)
        self.assertTrue(DATA_PROCESSED_DIR.is_absolute())
        self.assertTrue(MODEL_ARTIFACT_PATH.is_absolute())
        self.assertTrue(REPORT_PATH.is_absolute())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.npz"
            MLP(6, 4, ("basso", "medio", "alto")).salva(path)
            self.assertIsNotNone(carica_modello(path))
            path.write_bytes(b"not an npz")
            self.assertIsNone(carica_modello(path))

    def test_json_invalido(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(cli.main(["--run-ml", "--parametri", "{" ]), 2)
        self.assertIn("JSON non valido", err.getvalue())


if __name__ == "__main__":
    unittest.main()
