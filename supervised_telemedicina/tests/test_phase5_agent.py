import tempfile
import unittest
from pathlib import Path

import numpy as np

from telemedicina_supervised.agents.supervised_agent import SupervisedAgent
from telemedicina_supervised.ml.mlp import MLP
from telemedicina_supervised.ml.scaler import StandardScaler
from telemedicina_supervised.models.vital_parameters import VitalParameters


BASE = {
    "pressione_sistolica": 120.0,
    "pressione_diastolica": 80.0,
    "frequenza_cardiaca": 70.0,
    "temperatura": 36.8,
    "saturazione_ossigeno": 98.0,
    "glicemia": 95.0,
}


def artifact(path: Path, *, scaler=True, classi=("basso", "medio", "alto"), n_input=6):
    model = MLP(n_input, 2, classi, seed=41)
    model.W1.fill(0.0)
    model.W2.fill(0.0)
    model.b2[:] = 0.0
    model.b2[0] = 3.0
    if scaler:
        model.scaler = StandardScaler().fit(np.ones((3, n_input)))
    model.salva(path)
    return model


class TestPhase5Agent(unittest.TestCase):
    def test_valid_artifact_probabilita_e_scaler(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.npz"
            artifact(path)
            outcome = SupervisedAgent(path).predict(BASE)
            self.assertTrue(outcome.modello_usato)
            self.assertAlmostEqual(sum((outcome.probabilita or {}).values()), 1.0)
            self.assertEqual(outcome.classe_mlp, "basso")

    def test_vital_parameters_object_inference_and_dict_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.npz"
            artifact(path)
            vitali = VitalParameters(**BASE)
            outcome = SupervisedAgent(path).predict(vitali)
            self.assertTrue(outcome.modello_usato)
            compat = SupervisedAgent(path).analizza_parametri(vitali)
            self.assertIn("probabilita", compat)
            self.assertIn("metadati", compat)

    def test_invalid_artifacts_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = SupervisedAgent(root / "missing.npz").predict(BASE)
            self.assertTrue(missing.fallback)
            self.assertIn("non trovato", missing.motivo_fallback or "")
            bad = root / "bad.npz"
            bad.write_bytes(b"corrupt")
            self.assertTrue(SupervisedAgent(bad).predict(BASE).fallback)
            no_scaler = root / "no_scaler.npz"
            artifact(no_scaler, scaler=False)
            self.assertIn("scaler", SupervisedAgent(no_scaler).predict(BASE).motivo_fallback or "")
            wrong_class = root / "classes.npz"
            artifact(wrong_class, classi=("x", "y", "z"))
            self.assertTrue(SupervisedAgent(wrong_class).predict(BASE).fallback)
            wrong_version = root / "version.npz"
            artifact(wrong_version)
            with np.load(wrong_version) as saved:
                payload = {name: saved[name] for name in saved.files}
            payload["versione_formato"] = np.array(999)
            np.savez(wrong_version, **payload)
            self.assertTrue(SupervisedAgent(wrong_version).predict(BASE).fallback)
            wrong_bias = root / "bias.npz"
            model = artifact(wrong_bias)
            model.b2 = np.zeros(2)
            model.salva(wrong_bias)
            self.assertIn("bias", SupervisedAgent(wrong_bias).predict(BASE).motivo_fallback or "")

    def test_invalid_weights_dimension_and_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.npz"
            model = artifact(path)
            model.W1[0, 0] = np.nan
            model.salva(path)
            self.assertTrue(SupervisedAgent(path).predict(BASE).fallback)
            wrong_dim = Path(tmp) / "dimension.npz"
            artifact(wrong_dim, n_input=5)
            self.assertTrue(SupervisedAgent(wrong_dim).predict(BASE).fallback)
            invalid = dict(BASE, glicemia="bad")
            outcome = SupervisedAgent(path).predict(invalid)
            self.assertTrue(outcome.errore)
            self.assertIsNone(outcome.probabilita)

    def test_gate_and_no_declassification(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.npz"
            artifact(path)
            critical = dict(BASE, pressione_sistolica=185.0)
            outcome = SupervisedAgent(path).predict(critical)
            self.assertEqual(outcome.classe, "alto")
            self.assertIsNone(outcome.probabilita)
            moderate = dict(BASE, pressione_sistolica=165.0)
            outcome = SupervisedAgent(path).predict(moderate)
            self.assertEqual(outcome.classe, "medio")

    def test_final_class_table_and_safety_parity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.npz"
            model = artifact(path)
            for indice, classe_attesa in enumerate(("basso", "medio", "alto")):
                model.b2[:] = 0.0
                model.b2[indice] = 5.0
                outcome = SupervisedAgent(modello=model).predict(BASE)
                self.assertEqual(outcome.classe, classe_attesa)
                self.assertEqual(outcome.classe_mlp, classe_attesa)

            model.b2[:] = 0.0
            model.b2[0] = 5.0
            caso_medio = dict(BASE, pressione_sistolica=165.0)
            outcome = SupervisedAgent(modello=model).predict(caso_medio)
            self.assertEqual(outcome.classe, "medio")
            self.assertTrue(outcome.override_sicurezza)

            caso_critico = dict(BASE, pressione_sistolica=185.0)
            gated = SupervisedAgent(modello=model).predict(caso_critico)
            fallback = SupervisedAgent(modello_path=Path(tmp) / "missing.npz").predict(
                caso_critico
            )
            self.assertEqual(gated.classe, fallback.classe)
            self.assertEqual(gated.probabilita, fallback.probabilita)
            self.assertEqual(gated.errore, fallback.errore)
            self.assertEqual(gated.messaggio, fallback.messaggio)

    def test_uncertainty_delegates_to_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.npz"
            model = artifact(path)
            model.b2[:] = 0.0
            model.salva(path)
            outcome = SupervisedAgent(path).predict(BASE)
            self.assertTrue(outcome.fallback)
            self.assertEqual(outcome.classe, "basso")

    def test_normal_boundaries_remain_rule_based(self):
        agent = SupervisedAgent()
        self.assertEqual(agent.predict(dict(BASE, pressione_sistolica=139.9)).classe, "basso")
        self.assertEqual(agent.predict(dict(BASE, pressione_sistolica=140.1)).classe, "medio")
        self.assertEqual(agent.predict(dict(BASE, pressione_diastolica=89.9)).classe, "basso")
        self.assertEqual(agent.predict(dict(BASE, pressione_diastolica=90.1)).classe, "medio")


if __name__ == "__main__":
    unittest.main()
