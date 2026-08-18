"""Test della Fase 4: metriche di valutazione e workflow di training."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from telemedicina_supervised.ml.labels import to_indici
from telemedicina_supervised.ml.mlp import MLP
from telemedicina_supervised.ml.scaler import StandardScaler
from telemedicina_supervised.safety.safety_rules import CLASSI

from training.metrics import (
    accuratezza,
    distanza_dalle_soglie,
    kappa_cohen,
    macro_f1,
    matrice_confusione,
    metriche_per_classe,
    metriche_sicurezza,
    riepilogo_metriche,
)
from training.synthetic_generator import genera_batch
from training.train_model import (
    GRID,
    addestra_configurazione,
    analisi_errori_per_distanza,
    baseline_rule_based,
    carica_dataset,
    prepara,
    seleziona_migliore,
)


class TestMetriche(unittest.TestCase):
    """Metriche pure: casi noti e invarianti."""

    def test_accuratezza_e_confusione(self):
        y_true = np.array([0, 1, 2, 1, 0, 2])
        y_pred = np.array([0, 1, 2, 0, 0, 2])
        # Corretti: pos 0,1,2,4,5 (5 su 6); errato: pos 3 (1 predetto 0).
        self.assertAlmostEqual(accuratezza(y_true, y_pred), 5 / 6)
        matrice = matrice_confusione(y_true, y_pred, 3)
        np.testing.assert_array_equal(
            matrice,
            [[2, 0, 0], [1, 1, 0], [0, 0, 2]],
        )

    def test_metriche_per_classe_e_macro_f1(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 2, 2, 2])
        per_classe = metriche_per_classe(y_true, y_pred, 3)
        self.assertAlmostEqual(per_classe["0"]["precision"], 1.0)
        self.assertAlmostEqual(per_classe["0"]["recall"], 1.0)
        self.assertAlmostEqual(per_classe["1"]["recall"], 0.5)
        self.assertAlmostEqual(per_classe["2"]["precision"], 2 / 3)
        self.assertAlmostEqual(macro_f1(y_true, y_pred, 3), (1.0 + 2 / 3 + 0.8) / 3)

    def test_kappa_perfetto_e_casuale(self):
        y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        self.assertAlmostEqual(kappa_cohen(y, y, 3), 1.0)
        # Predizioni tutte uguali: kappa = 0 (accordo solo per caso).
        y_pred_costante = np.zeros_like(y)
        self.assertAlmostEqual(kappa_cohen(y, y_pred_costante, 3), 0.0)

    def test_distanza_dalle_soglie(self):
        # Campione esattamente su una soglia critica: distanza 0.
        X_soglia = np.array([[180.0, 90.0, 70.0, 36.5, 98.0, 100.0]])
        distanze = distanza_dalle_soglie(X_soglia)
        self.assertAlmostEqual(distanze[0], 0.0)
        # Campione normale: distanza > 0 da ogni confine.
        X_normale = np.array([[120.0, 80.0, 75.0, 36.6, 98.0, 100.0]])
        self.assertGreater(distanza_dalle_soglie(X_normale)[0], 0.0)

    def test_distanza_normalizzata_per_ampiezza_range(self):
        # La distanza è normalizzata per l'ampiezza del range normale:
        # glicemia 300 (confine 140, ampiezza 70) -> 160/70 ~= 2.29,
        # NON 160 grezzi. Senza normalizzazione la temperatura (range
        # stretto) dominerebbe la distanza minima di ogni campione.
        X_glicemia = np.array([[120.0, 80.0, 75.0, 36.6, 98.0, 300.0]])
        distanza = distanza_dalle_soglie(X_glicemia)[0]
        self.assertAlmostEqual(distanza, 160.0 / 70.0, places=6)
        # Temperatura appena fuori range (37.6, confine 37.5, ampiezza 1.5)
        # deve dare una distanza piccola ma comparabile in unità di ampiezza.
        X_temperatura = np.array([[120.0, 80.0, 75.0, 37.6, 98.0, 100.0]])
        distanza_t = distanza_dalle_soglie(X_temperatura)[0]
        self.assertAlmostEqual(distanza_t, 0.1 / 1.5, places=6)
        self.assertGreater(distanza, distanza_t)

    def test_metriche_sicurezza(self):
        y_true = np.array([2, 2, 2, 1, 0])
        y_pred = np.array([2, 1, 0, 1, 0])
        sicurezza = metriche_sicurezza(y_true, y_pred, 3, indice_alto=2)
        self.assertAlmostEqual(sicurezza["recall_alto"], 1 / 3)
        self.assertEqual(sicurezza["mancati_alto"], 2)
        # Declassati a NON-alto (qualsiasi classe inferiore, anche 'medio'):
        # pos 1 (2->1) e pos 2 (2->0).
        self.assertEqual(sicurezza["declassati_critici"], 2)

    def test_riepilogo_completo(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        riepilogo = riepilogo_metriche(y_true, y_pred, 3, indice_alto=2)
        self.assertAlmostEqual(riepilogo["accuracy"], 1.0)
        self.assertAlmostEqual(riepilogo["kappa_cohen"], 1.0)
        self.assertEqual(len(riepilogo["matrice_confusione"]), 3)
        self.assertAlmostEqual(riepilogo["sicurezza"]["recall_alto"], 1.0)


class TestWorkflowTraining(unittest.TestCase):
    """Workflow Fase 4 su scala ridotta (dataset sintetico piccolo)."""

    @classmethod
    def setUpClass(cls):
        cls.X_train, cls.y_train, _ = genera_batch(101, 600, buffer_zone=True)
        cls.X_val, cls.y_val, _ = genera_batch(102, 200, buffer_zone=True)
        cls.X_test, cls.y_test, _ = genera_batch(103, 300, buffer_zone=True)

    def test_prepara_scaler_solo_train(self):
        dati = {
            "X_train": self.X_train,
            "y_train": self.y_train,
            "X_val": self.X_val,
            "y_val": self.y_val,
            "X_test": self.X_test,
            "y_test": self.y_test,
        }
        preparati, scaler = prepara(dati)
        self.assertIsNotNone(scaler.media)
        self.assertEqual(preparati["X_train"].shape, (600, 6))
        self.assertEqual(preparati["y_train"].dtype, np.int64)
        # Lo scaler è stato stimato sul train: media ~0 sul train.
        np.testing.assert_allclose(
            preparati["X_train"].mean(axis=0), 0.0, atol=1e-10
        )

    def test_addestra_configurazione_loss_decresce(self):
        scaler = StandardScaler().fit(self.X_train)
        X_tr = scaler.transform(self.X_train)
        X_va = scaler.transform(self.X_val)
        y_tr = to_indici(self.y_train, CLASSI)
        y_va = to_indici(self.y_val, CLASSI)
        modello, storia = addestra_configurazione(
            X_tr, y_tr, X_va, y_va, GRID[0], seed=41
        )
        self.assertLess(storia["loss_val"][-1], storia["loss_val"][0])
        self.assertEqual(len(storia["loss_train"]), len(storia["loss_val"]))

    def test_seleziona_migliore_su_validation(self):
        scaler = StandardScaler().fit(self.X_train)
        X_tr = scaler.transform(self.X_train)
        X_va = scaler.transform(self.X_val)
        y_tr = to_indici(self.y_train, CLASSI)
        y_va = to_indici(self.y_val, CLASSI)
        migliore, loss, _ = seleziona_migliore(
            X_tr, y_tr, X_va, y_va, seed=41
        )
        self.assertIn("n_hidden", migliore)
        self.assertGreater(loss, 0.0)

    def test_analisi_errori_per_distanza(self):
        y_true = np.array([0, 1, 2, 0])
        y_pred = np.array([0, 1, 2, 1])  # un errore
        X = np.array(
            [
                [120.0, 80.0, 75.0, 36.6, 98.0, 100.0],
                [130.0, 85.0, 90.0, 36.8, 97.0, 110.0],
                [200.0, 120.0, 130.0, 39.0, 85.0, 250.0],
                [95.0, 65.0, 70.0, 36.4, 96.0, 90.0],
            ]
        )
        risultato = analisi_errori_per_distanza(X, y_true, y_pred)
        self.assertEqual(risultato["n_errori"], 1)
        self.assertGreater(risultato["distanza_media_errori"], 0.0)

    def test_baseline_rule_based_concorda_col_teacher(self):
        # Le label del dataset sono generate dal teacher: la baseline deve
        # concordare quasi totalmente (regressione: ndarray non accettato
        # da classifica_rischio -> accordo 0.0).
        risultato = baseline_rule_based(self.X_test, self.y_test)
        self.assertGreater(risultato["accordo_teacher_label"], 0.99)
        self.assertEqual(risultato["n_campioni"], 300)

    def test_carica_dataset_e_roundtrip_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            for nome in ("train", "val", "test"):
                np.save(directory_path / f"X_{nome}.npy", self.X_train[:10])
                np.save(directory_path / f"y_{nome}.npy", self.y_train[:10])
            dati = carica_dataset(directory_path)
            self.assertEqual(dati["X_train"].shape, (10, 6))
            self.assertEqual(dati["y_test"].shape, (10,))

            # Roundtrip completo: modello + scaler.
            scaler = StandardScaler().fit(self.X_train)
            modello = MLP(n_input=6, n_hidden=8, classi=CLASSI, seed=41)
            percorso = directory_path / "modello.npz"
            modello.salva(percorso, scaler=scaler)
            caricato = MLP.carica(percorso)
            self.assertIsNotNone(caricato.scaler)
            np.testing.assert_allclose(
                modello.W1, caricato.W1
            )


if __name__ == "__main__":
    unittest.main()