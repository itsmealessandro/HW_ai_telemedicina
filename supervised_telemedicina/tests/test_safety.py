"""
test_safety.py — Test di validazione e classificazione rule-based.

Coprono:
  - validazione dei range fisiologici e vincolo sistolica > diastolica;
  - classificazione basso/medio/alto su casi noti;
  - risposta 'errore' per input invalidi.

Nessun import dal legacy (archive/).
"""

import unittest
from typing import Any, Dict

from telemedicina_supervised.safety.safety_rules import (
    CLASSI,
    CLASSE_ERRORE,
    RANGE_FISIOLOGICI,
    RANGE_NORMALI,
    SOGLIE_CRITICHE,
    SOGLIE_MODERATE,
    classifica_rischio,
    valida_parametri,
)


def vp(**kwargs: Any) -> Dict[str, Any]:
    """Combinazione di 6 parametri validi, sovrascrivibile per singolo test."""
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


class TestValidaParametri(unittest.TestCase):
    """Validazione fisiologica (range + vincolo sistolica > diastolica)."""

    def test_input_normale_valido(self):
        valido, errori = valida_parametri(vp())
        self.assertTrue(valido)
        self.assertEqual(errori, [])

    def test_range_fisiologici_estremi_inclusi(self):
        # I limiti dei range sono INCLUSI (fedeli al legacy).
        estremi = {
            "pressione_sistolica": 50.0,
            "pressione_diastolica": 30.0,
            "frequenza_cardiaca": 30.0,
            "temperatura": 34.0,
            "saturazione_ossigeno": 70.0,
            "glicemia": 20.0,
        }
        valido, errori = valida_parametri(vp(**estremi))
        self.assertTrue(valido, errori)

        estremi_sup = {
            "pressione_sistolica": 250.0,
            "pressione_diastolica": 150.0,
            "frequenza_cardiaca": 200.0,
            "temperatura": 42.0,
            "saturazione_ossigeno": 100.0,
            "glicemia": 500.0,
        }
        valido, errori = valida_parametri(vp(**estremi_sup))
        self.assertTrue(valido, errori)

    def test_sistolica_fuori_range_basso(self):
        valido, errori = valida_parametri(vp(pressione_sistolica=20.0))
        self.assertFalse(valido)
        self.assertTrue(any("pressione_sistolica" in e for e in errori))

    def test_sistolica_fuori_range_alto(self):
        valido, errori = valida_parametri(vp(pressione_sistolica=300.0))
        self.assertFalse(valido)
        self.assertTrue(any("pressione_sistolica" in e for e in errori))

    def test_diastolica_fuori_range(self):
        valido, errori = valida_parametri(vp(pressione_diastolica=200.0))
        self.assertFalse(valido)
        self.assertTrue(any("pressione_diastolica" in e for e in errori))

    def test_frequenza_cardiaca_fuori_range(self):
        valido, errori = valida_parametri(vp(frequenza_cardiaca=250.0))
        self.assertFalse(valido)
        self.assertTrue(any("frequenza_cardiaca" in e for e in errori))

    def test_temperatura_fuori_range(self):
        valido, errori = valida_parametri(vp(temperatura=30.0))
        self.assertFalse(valido)
        self.assertTrue(any("temperatura" in e for e in errori))

    def test_saturazione_fuori_range(self):
        valido, errori = valida_parametri(vp(saturazione_ossigeno=50.0))
        self.assertFalse(valido)
        self.assertTrue(any("saturazione_ossigeno" in e for e in errori))

    def test_glicemia_fuori_range(self):
        valido, errori = valida_parametri(vp(glicemia=10.0))
        self.assertFalse(valido)
        self.assertTrue(any("glicemia" in e for e in errori))

    def test_vincolo_sistolica_maggiore_diastolica(self):
        valido, errori = valida_parametri(
            vp(pressione_sistolica=100.0, pressione_diastolica=100.0)
        )
        self.assertFalse(valido)
        self.assertTrue(any("sistolica" in e and "diastolica" in e for e in errori))

    def test_vincolo_sistolica_uguale_diastolica(self):
        valido, _ = valida_parametri(
            vp(pressione_sistolica=80.0, pressione_diastolica=80.0)
        )
        self.assertFalse(valido)

    def test_input_non_numerico(self):
        valido, errori = valida_parametri(vp(glicemia="abc"))
        self.assertFalse(valido)
        self.assertTrue(any("glicemia" in e for e in errori))

    def test_range_normali_e_soglie_presenti_per_tutte_le_feature(self):
        # Le tabelle del contratto coprono le 6 feature ufficiali.
        for nome in RANGE_FISIOLOGICI:
            self.assertIn(nome, RANGE_NORMALI)
            self.assertIn(nome, SOGLIE_CRITICHE)
            self.assertIn(nome, SOGLIE_MODERATE)


class TestClassificaRischio(unittest.TestCase):
    """Classificazione basso/medio/alto su casi noti."""

    def test_normale_basso(self):
        self.assertEqual(classifica_rischio(vp()), "basso")

    def test_ipertensione_critica_alto(self):
        self.assertEqual(
            classifica_rischio(
                vp(pressione_sistolica=185.0, pressione_diastolica=110.0)
            ),
            "alto",
        )

    def test_shock_alto(self):
        # Ipotensione + tachicardia => pattern shock => alto.
        self.assertEqual(
            classifica_rischio(
                vp(
                    pressione_sistolica=85.0,
                    pressione_diastolica=55.0,
                    frequenza_cardiaca=115.0,
                )
            ),
            "alto",
        )

    def test_ipossia_alto(self):
        self.assertEqual(
            classifica_rischio(
                vp(saturazione_ossigeno=88.0, frequenza_cardiaca=110.0)
            ),
            "alto",
        )

    def test_boundary_spo2_90_critica_alto(self):
        # Boundary P1: soglia critica bassa SpO2 = 90 (codice: <= 90 =>
        # critica). SpO2=90.0 con resto normale => 'alto'.
        self.assertEqual(
            classifica_rischio(vp(saturazione_ossigeno=90.0)), "alto"
        )

    def test_boundary_spo2_90_1_non_critico(self):
        # Boundary P1: SpO2=90.1 è SOPRA la soglia critica (non critica
        # per questo solo parametro): resta 'medio' (anomalia moderata,
        # <= 93), mai 'alto'.
        risultato = classifica_rischio(vp(saturazione_ossigeno=90.1))
        self.assertNotEqual(risultato, "alto")
        self.assertEqual(risultato, "medio")

    def test_bradicardia_critica_alto(self):
        self.assertEqual(classifica_rischio(vp(frequenza_cardiaca=45.0)), "alto")

    def test_febbre_critica_alto(self):
        self.assertEqual(classifica_rischio(vp(temperatura=39.0)), "alto")

    def test_ipoglicemia_critica_alto(self):
        self.assertEqual(classifica_rischio(vp(glicemia=55.0)), "alto")

    def test_ipoglicemia_al_valore_di_soglia_unificata_alto(self):
        # Soglia critica bassa unificata a 60: glicemia=60 => critica => alto.
        self.assertEqual(classifica_rischio(vp(glicemia=60.0)), "alto")

    def test_iperglicemia_critica_alto(self):
        self.assertEqual(classifica_rischio(vp(glicemia=210.0)), "alto")

    def test_ipotensione_medio(self):
        # Sistolica lieve (<90 ma >=80), senza tachicardia: nessun pattern,
        # nessuna anomalia critica => medio.
        self.assertEqual(
            classifica_rischio(
                vp(pressione_sistolica=85.0, pressione_diastolica=60.0)
            ),
            "medio",
        )

    def test_febbre_moderata_medio(self):
        self.assertEqual(classifica_rischio(vp(temperatura=38.2)), "medio")

    def test_bradicardia_moderata_medio(self):
        self.assertEqual(classifica_rischio(vp(frequenza_cardiaca=54.0)), "medio")

    def test_glicemia_limite_basso_medio(self):
        # Glicemia 62: fuori range normale ma non critica (<=65 => moderata) => medio.
        self.assertEqual(classifica_rischio(vp(glicemia=62.0)), "medio")

    def test_input_invalido_errore_mai_classe(self):
        for caso in (
            vp(pressione_sistolica=20.0),  # fuori range
            vp(glicemia=10.0),             # fuori range
            vp(pressione_sistolica=90.0, pressione_diastolica=90.0),  # vincolo
            vp(saturazione_ossigeno="n/d"),  # non numerico
        ):
            risultato = classifica_rischio(caso)
            self.assertEqual(risultato, CLASSE_ERRORE)
            self.assertNotIn(risultato, CLASSI)

    def test_classi_ufficiali(self):
        self.assertEqual(CLASSI, ("basso", "medio", "alto"))


if __name__ == "__main__":
    unittest.main()
