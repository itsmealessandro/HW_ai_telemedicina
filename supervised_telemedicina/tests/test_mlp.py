"""Test toy del core MLP, indipendenti dal dataset telemedico reale."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from telemedicina_supervised.ml.gradient_check import errore_massimo_relativo
from telemedicina_supervised.ml.labels import mappa_indici, to_etichette, to_indici
from telemedicina_supervised.ml.mlp import MLP, VERSIONE_FORMATO
from telemedicina_supervised.ml.scaler import StandardScaler


class TestMLPCore(unittest.TestCase):
    """Gate Fase 3: invarianti matematiche e persistenza."""

    CLASSI = ("classe_a", "classe_b", "classe_c")

    def test_forme_softmax_finita_e_somma_uno(self):
        modello = MLP(n_input=3, n_hidden=4, classi=self.CLASSI, seed=41)
        X = np.array(
            [
                [1.0, -0.5, 0.2],
                [-0.4, 1.3, 0.7],
                [0.8, 0.1, -1.1],
                [2.0, -2.0, 1.0],
                [-1.0, 0.3, 0.4],
            ]
        )

        avanti = modello.forward(X)
        probs = avanti["probs"]
        self.assertEqual(avanti["z1"].shape, (5, 4))
        self.assertEqual(avanti["a1"].shape, (5, 4))
        self.assertEqual(avanti["z2"].shape, (5, 3))
        self.assertEqual(probs.shape, (5, 3))
        self.assertTrue(np.isfinite(probs).all())
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-12)

    def test_gradient_checking(self):
        modello = MLP(n_input=3, n_hidden=4, classi=self.CLASSI, seed=17)
        X = np.array(
            [
                [0.2, -0.7, 1.1],
                [-1.2, 0.4, 0.6],
                [0.8, 0.9, -0.3],
                [1.4, -0.2, -0.5],
                [-0.6, -0.8, 0.2],
            ]
        )
        y = np.array([0, 1, 2, 1, 0], dtype=np.int64)

        errore = errore_massimo_relativo(modello, X, y, eps=1e-6)
        self.assertLess(errore, 2e-5, msg=f"errore gradient check: {errore}")

    def test_loss_diminuisce_su_toy_separabile(self):
        rng = np.random.default_rng(23)
        centri = np.array([[-2.0, -1.0], [0.0, 2.0], [2.0, -1.0]])
        X = np.vstack(
            [centro + 0.15 * rng.normal(size=(24, 2)) for centro in centri]
        )
        y = np.repeat(np.arange(3, dtype=np.int64), 24)
        modello = MLP(n_input=2, n_hidden=8, classi=self.CLASSI, seed=41)

        loss_iniziale = modello.loss(X, y)
        storia = modello.train(
            X,
            y,
            epoche=80,
            batch_size=12,
            lr=0.05,
            pazienza=15,
        )

        self.assertGreater(len(storia["loss_train"]), 1)
        self.assertLess(storia["loss_train"][-1], loss_iniziale)
        self.assertLess(modello.loss(X, y), loss_iniziale)

    def test_training_riproducibile_con_seed(self):
        rng = np.random.default_rng(31)
        X = rng.normal(size=(27, 3))
        y = np.arange(27, dtype=np.int64) % 3
        primo = MLP(n_input=3, n_hidden=5, classi=self.CLASSI, seed=41)
        secondo = MLP(n_input=3, n_hidden=5, classi=self.CLASSI, seed=41)

        primo.train(X, y, epoche=12, batch_size=9, lr=0.03, pazienza=7)
        secondo.train(X, y, epoche=12, batch_size=9, lr=0.03, pazienza=7)

        np.testing.assert_allclose(primo.W1, secondo.W1)
        np.testing.assert_allclose(primo.W2, secondo.W2)
        np.testing.assert_array_equal(
            primo.predici_indici(X), secondo.predici_indici(X)
        )

    def test_scaler_fit_inverso_e_artifact_roundtrip(self):
        X_train = np.array(
            [[10.0, 1.0, -2.0], [12.0, 3.0, 0.0], [14.0, 5.0, 2.0]]
        )
        X_probe = np.array([[11.0, 2.0, -1.0], [15.0, 7.0, 3.0]])
        scaler = StandardScaler().fit(X_train)
        X_norm = scaler.transform(X_train)
        np.testing.assert_allclose(X_norm.mean(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(scaler.inverso(X_norm), X_train)

        modello = MLP(n_input=3, n_hidden=4, classi=self.CLASSI, seed=41)
        with tempfile.TemporaryDirectory() as directory:
            percorso = Path(directory) / "modello.npz"
            modello.salva(percorso, scaler=scaler)
            caricato = MLP.carica(percorso)

            np.testing.assert_allclose(
                modello.probabilità(scaler.transform(X_probe)),
                caricato.probabilità(caricato.scaler.transform(X_probe)),
            )
            self.assertEqual(caricato.classi, self.CLASSI)
            self.assertEqual(caricato.n_input, 3)
            self.assertEqual(caricato.n_hidden, 4)
            self.assertIsNotNone(caricato.scaler)

            with np.load(percorso) as dati:
                self.assertEqual(int(dati["versione_formato"]), VERSIONE_FORMATO)
                self.assertIn("scaler_media", dati)
                self.assertIn("scaler_dev", dati)

    def test_label_encoding_stabile(self):
        classi = ("basso", "medio", "alto")
        etichette = np.array(["alto", "basso", "medio", "basso"])
        indici = to_indici(etichette, classi)
        np.testing.assert_array_equal(indici, [2, 0, 1, 0])
        np.testing.assert_array_equal(to_etichette(indici, classi), etichette)
        self.assertEqual(mappa_indici(classi), {"basso": 0, "medio": 1, "alto": 2})


if __name__ == "__main__":
    unittest.main()
