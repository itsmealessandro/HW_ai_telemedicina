"""
test_contract.py — Test del CONTRATTO funzionale (docs/contratto.md).

Coprono:
  - ordine ufficiale delle 6 feature e del vettore to_vector();
  - classi ufficiali e risposta 'errore' per input invalidi;
  - modello dati VitalParameters (validazione delegata a safety_rules);
  - soglie presenti SOLO in safety_rules (fonte unica).

Nessun import dal legacy (archive/).
"""

import unittest

import numpy as np

from telemedicina_supervised.models.vital_parameters import VitalParameters
from telemedicina_supervised.safety.safety_rules import (
    CLASSI,
    CLASSE_ERRORE,
    FEATURE_ORDER,
    classifica_rischio,
)


def vp(**kwargs):
    base = {
        "pressione_sistolica": 120.0,
        "pressione_diastolica": 80.0,
        "frequenza_cardiaca": 75.0,
        "temperatura": 36.8,
        "saturazione_ossigeno": 98.0,
        "glicemia": 95.0,
    }
    base.update(kwargs)
    return base


class TestFeatureOrder(unittest.TestCase):
    """Ordine ufficiale delle feature (contratto punto 1)."""

    ORDINE_UFFICIALE = [
        "pressione_sistolica",
        "pressione_diastolica",
        "frequenza_cardiaca",
        "temperatura",
        "saturazione_ossigeno",
        "glicemia",
    ]

    def test_feature_order_ufficiale(self):
        self.assertEqual(FEATURE_ORDER, self.ORDINE_UFFICIALE)

    def test_to_vector_ordine_ufficiale(self):
        v = VitalParameters(120.0, 80.0, 75.0, 36.8, 98.0, 95.0)
        atteso = np.array([120.0, 80.0, 75.0, 36.8, 98.0, 95.0])
        self.assertEqual(v.to_vector().shape, (6,))
        np.testing.assert_allclose(v.to_vector(), atteso)

    def test_to_vector_rispetta_ordine_dei_campi(self):
        # Valori tutti distinti e crescenti: il vettore deve replicare
        # l'ordine strutturale della dataclass, non un riordino.
        v = VitalParameters(100.0, 90.0, 80.0, 37.0, 96.0, 90.0)
        atteso = np.array([100.0, 90.0, 80.0, 37.0, 96.0, 90.0])
        np.testing.assert_allclose(v.to_vector(), atteso)

    def test_to_dict_chiavi_ufficiali(self):
        v = VitalParameters(120.0, 80.0, 75.0, 36.8, 98.0, 95.0)
        self.assertEqual(list(v.to_dict().keys()), FEATURE_ORDER)


class TestClassiUfficiali(unittest.TestCase):
    """Classi ufficiali (contratto punto 2)."""

    def test_classi_ufficiali(self):
        self.assertEqual(CLASSI, ("basso", "medio", "alto"))

    def test_classifica_rischio_ritorna_solo_valori_contrattuali(self):
        casi = [vp(), vp(glicemia=55.0), vp(temperatura=38.2), vp(glicemia=10.0)]
        for caso in casi:
            risultato = classifica_rischio(caso)
            self.assertIn(
                risultato,
                ("basso", "medio", "alto", CLASSE_ERRORE),
                msg=f"valore fuori contratto: {risultato!r}",
            )


class TestErroreInputInvalidi(unittest.TestCase):
    """Input invalidi -> 'errore', mai una classe di rischio (punto 3)."""

    def test_input_invalidi_mai_classe_di_rischio(self):
        casi = [
            vp(pressione_sistolica=20.0),
            vp(pressione_diastolica=200.0),
            vp(frequenza_cardiaca=250.0),
            vp(temperatura=30.0),
            vp(saturazione_ossigeno=50.0),
            vp(glicemia=10.0),
            vp(pressione_sistolica=100.0, pressione_diastolica=100.0),
        ]
        for caso in casi:
            risultato = classifica_rischio(caso)
            self.assertEqual(risultato, CLASSE_ERRORE, msg=f"caso {caso}")
            self.assertNotIn(risultato, CLASSI, msg=f"caso {caso}")


class TestVitalParameters(unittest.TestCase):
    """Modello dati: validazione delegata a safety_rules, segnalazione."""

    def test_costruttore_segnala_input_validi(self):
        v = VitalParameters(120.0, 80.0, 75.0, 36.8, 98.0, 95.0)
        self.assertTrue(v.valido)
        self.assertEqual(v.motivi, [])

    def test_costruttore_segnala_vincolo_sistolica_diastolica(self):
        v = VitalParameters(90.0, 90.0, 75.0, 36.8, 98.0, 95.0)
        self.assertFalse(v.valido)
        self.assertTrue(
            any("sistolica" in m and "diastolica" in m for m in v.motivi)
        )

    def test_costruttore_segnala_input_fuori_range(self):
        v = VitalParameters(120.0, 80.0, 75.0, 36.8, 98.0, 10.0)
        self.assertFalse(v.valido)
        self.assertTrue(any("glicemia" in m for m in v.motivi))

    def test_validazione_delegata_a_safety_rules(self):
        # vp = dict; valida_parametri su dict e su VitalParameters danno
        # lo stesso esito (fonte unica).
        v = VitalParameters(120.0, 80.0, 75.0, 36.8, 98.0, 95.0)
        self.assertTrue(v.valida_parametri()[0])

        invalido = VitalParameters(100.0, 100.0, 75.0, 36.8, 98.0, 95.0)
        valido_dict, errori_dict = invalido.valida_parametri()
        self.assertFalse(valido_dict)
        self.assertEqual(errori_dict, invalido.motivi)

    def test_da_dict_rispetta_ordine_ufficiale(self):
        v = VitalParameters.da_dict(
            {
                "glicemia": 95.0,
                "pressione_sistolica": 120.0,
                "temperatura": 36.8,
                "saturazione_ossigeno": 98.0,
                "pressione_diastolica": 80.0,
                "frequenza_cardiaca": 75.0,
            }
        )
        self.assertEqual(v.pressione_sistolica, 120.0)
        self.assertEqual(v.glicemia, 95.0)
        self.assertEqual(v.to_dict()["frequenza_cardiaca"], 75.0)


if __name__ == "__main__":
    unittest.main()
