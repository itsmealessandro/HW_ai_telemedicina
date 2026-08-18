"""
test_teacher.py — Test del teacher offline (training/teacher_rules.py).

Coprono:
  - la coerenza teacher <-> safety_rules (stessa fonte di verità);
  - le label sui casi noti (normale, critici, invalidi);
  - la risposta 'errore' per input invalidi.

Nessun import dal legacy (archive/).
"""

import unittest

from telemedicina_supervised.safety.safety_rules import classifica_rischio
from training.teacher_rules import TeacherRules, label as label_modulo


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


class TestTeacherRules(unittest.TestCase):
    """Teacher offline: riusa classifica_rischio, nessuna soglia propria."""

    def setUp(self):
        self.teacher = TeacherRules()

    def test_coerenza_teacher_safety_stessa_fonte(self):
        # Su un campione di casi (normali, anomalie, critici, invalidi)
        # la label del teacher è IDENTICA a classifica_rischio.
        casi = [
            vp(),
            vp(pressione_sistolica=185.0, pressione_diastolica=110.0),
            vp(pressione_sistolica=85.0, frequenza_cardiaca=115.0),
            vp(saturazione_ossigeno=88.0),
            vp(frequenza_cardiaca=45.0),
            vp(temperatura=39.0),
            vp(glicemia=55.0),
            vp(glicemia=210.0),
            vp(pressione_sistolica=85.0, pressione_diastolica=60.0),
            vp(temperatura=38.2),
            vp(glicemia=62.0),
            vp(glicemia=10.0),
            vp(pressione_sistolica=90.0, pressione_diastolica=90.0),
        ]
        for caso in casi:
            self.assertEqual(
                self.teacher.label(caso),
                classifica_rischio(caso),
                msg=f"label teacher != classifica_rischio per {caso}",
            )

    def test_label_casi_noti(self):
        self.assertEqual(self.teacher.label(vp()), "basso")
        self.assertEqual(
            self.teacher.label(vp(pressione_sistolica=185.0)), "alto"
        )
        self.assertEqual(self.teacher.label(vp(frequenza_cardiaca=45.0)), "alto")
        self.assertEqual(self.teacher.label(vp(temperatura=38.2)), "medio")
        self.assertEqual(self.teacher.label(vp(glicemia=55.0)), "alto")

    def test_label_input_invalido_errore(self):
        self.assertEqual(self.teacher.label(vp(glicemia=10.0)), "errore")
        self.assertEqual(
            self.teacher.label(
                vp(pressione_sistolica=90.0, pressione_diastolica=90.0)
            ),
            "errore",
        )

    def test_funzione_di_modulo_equivalente(self):
        for caso in (vp(), vp(glicemia=55.0), vp(pressione_sistolica=20.0)):
            self.assertEqual(label_modulo(caso), classifica_rischio(caso))

    def test_etichetta_alias(self):
        self.assertEqual(
            self.teacher.etichetta(vp(temperatura=39.0)),
            self.teacher.label(vp(temperatura=39.0)),
        )

    def test_teacher_accetta_oggetti_vital_parameters(self):
        from telemedicina_supervised.models.vital_parameters import VitalParameters

        v = VitalParameters(120.0, 80.0, 75.0, 36.8, 98.0, 95.0)
        self.assertEqual(self.teacher.label(v), "basso")


if __name__ == "__main__":
    unittest.main()
