"""
tests/test_dataset.py — Test del generatore sintetico (Fase 2).

Copre: riproducibilità con lo stesso seed, vincolo sistolica > diastolica,
assenza di label 'errore', presenza di tutte e 3 le classi, buffer zone
(attiva e disattivata), diversità tra seed diversi.

Nessun import dal legacy (archive/).
"""

import unittest

import numpy as np

from telemedicina_supervised.safety.safety_rules import (
    FEATURE_ORDER,
    RANGE_FISIOLOGICI,
    valida_parametri,
)
from training.synthetic_generator import (
    DIFF_MIN_SIST_DIAST,
    MARGINI_BUFFER,
    _in_buffer_zone,
    genera_batch,
)

IDX_SIST = FEATURE_ORDER.index("pressione_sistolica")
IDX_DIAST = FEATURE_ORDER.index("pressione_diastolica")


def _campione_da_riga(riga: np.ndarray) -> dict:
    """Converte una riga di X in un dict con i nomi ufficiali."""
    return {nome: float(riga[i]) for i, nome in enumerate(FEATURE_ORDER)}


class TestGeneratoreSintetico(unittest.TestCase):
    """Proprietà fondamentali del generatore."""

    def test_riproducibilita_stesso_seed(self):
        X1, y1, _ = genera_batch(seed=7, n=200)
        X2, y2, _ = genera_batch(seed=7, n=200)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_seed_diversi_producono_array_diversi(self):
        X1, _, _ = genera_batch(seed=7, n=200)
        X2, _, _ = genera_batch(seed=8, n=200)
        self.assertFalse(np.array_equal(X1, X2))

    def test_vincolo_sistolica_maggiore_diastolica(self):
        X, _, _ = genera_batch(seed=7, n=500)
        differenze = X[:, IDX_SIST] - X[:, IDX_DIAST]
        self.assertTrue(np.all(differenze >= DIFF_MIN_SIST_DIAST))

    def test_tutti_i_campioni_validi(self):
        X, y, _ = genera_batch(seed=7, n=500)
        self.assertNotIn("errore", set(y))
        for riga in X:
            valido, errori = valida_parametri(_campione_da_riga(riga))
            self.assertTrue(valido, msg=f"campione invalido: {errori}")

    def test_valori_dentro_range_fisiologici(self):
        X, _, _ = genera_batch(seed=7, n=500)
        for i, nome in enumerate(FEATURE_ORDER):
            min_v, max_v = RANGE_FISIOLOGICI[nome]
            self.assertGreaterEqual(X[:, i].min(), min_v)
            self.assertLessEqual(X[:, i].max(), max_v)

    def test_tutte_le_classi_presenti(self):
        X, y, _ = genera_batch(seed=7, n=5000)
        self.assertEqual(set(y), {"basso", "medio", "alto"})

    def test_buffer_zone_attiva_nessun_campione_vicino_alle_soglie(self):
        X, _, _ = genera_batch(seed=7, n=1000, buffer_zone=True)
        for riga in X:
            self.assertFalse(
                _in_buffer_zone(_campione_da_riga(riga)),
                msg=f"campione in buffer zone: {riga}",
            )

    def test_buffer_zone_disattivata_consente_campioni_vicini_alle_soglie(self):
        # Con buffer_zone=False il generatore NON deve scartare i campioni
        # vicini alle soglie: su un batch grande deve comparirne almeno uno.
        X, _, _ = genera_batch(seed=7, n=5000, buffer_zone=False)
        trovato = any(_in_buffer_zone(_campione_da_riga(riga)) for riga in X)
        self.assertTrue(trovato)

    def test_margini_buffer_definiti_per_tutte_le_feature(self):
        self.assertEqual(set(MARGINI_BUFFER.keys()), set(FEATURE_ORDER))
        for margine in MARGINI_BUFFER.values():
            self.assertGreater(margine, 0.0)


if __name__ == "__main__":
    unittest.main()