"""
test_agent_service.py — Test del contratto SupervisedAgent e
AnalysisService (baseline Fase 1).

Coprono:
  - contratto predict(): classe/probabilita/errore/messaggio;
  - PRECEDENZA DELLE REGOLE DI SICUREZZA: un input critico non viene mai
    declassato (la baseline rule-based non può restituire 'basso');
  - input invalidi -> 'errore' con errore=True e probabilita=None;
  - contratto del servizio: ServiceOutcome, allerta medico, storico.

Nessun import dal legacy (archive/).
"""

import unittest
from pathlib import Path
from typing import Dict

from telemedicina_supervised.agents.supervised_agent import AgentOutcome, SupervisedAgent
from telemedicina_supervised.models.vital_parameters import VitalParameters
from telemedicina_supervised.safety.safety_rules import CLASSE_ERRORE, classifica_rischio
from telemedicina_supervised.services.analysis_service import (
    AnalysisService,
    ServiceOutcome,
)


def vp(**kwargs) -> Dict:
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


# Casi critici che le regole giudicano 'alto' e che non possono MAI essere
# declassati (precedenza assoluta del safety gate).
CASI_CRITICI = [
    vp(pressione_sistolica=185.0, pressione_diastolica=110.0),  # crisi ipertensiva
    vp(pressione_sistolica=85.0, frequenza_cardiaca=115.0),     # shock
    vp(saturazione_ossigeno=88.0, frequenza_cardiaca=110.0),    # ipossia
    vp(frequenza_cardiaca=45.0),                                # bradicardia critica
    vp(temperatura=39.0),                                       # febbre critica
    vp(glicemia=55.0),                                          # ipoglicemia critica
    vp(glicemia=210.0),                                         # iperglicemia critica
]

CASI_INVALIDI = [
    vp(pressione_sistolica=20.0),
    vp(glicemia=10.0),
    vp(pressione_sistolica=100.0, pressione_diastolica=100.0),
    vp(saturazione_ossigeno=50.0),
]


class TestSupervisedAgent(unittest.TestCase):
    """Contratto del SupervisedAgent (baseline rule-based)."""

    def setUp(self):
        # Explicit absent artifact keeps this legacy fixture rule-based.
        self.agent = SupervisedAgent(modello_path=Path("/missing/legacy-model.npz"))

    def test_output_è_un_agent_outcome(self):
        esito = self.agent.predict(vp())
        self.assertIsInstance(esito, AgentOutcome)

    def test_input_normale_basso(self):
        esito = self.agent.predict(vp())
        self.assertEqual(esito.classe, "basso")
        self.assertFalse(esito.errore)
        self.assertIsNone(esito.probabilita)  # baseline rule-based: nessuna softmax
        self.assertTrue(esito.messaggio)

    def test_input_critico_mai_declassato(self):
        # Precedenza delle regole di sicurezza: per OGNI caso critico la
        # baseline restituisce 'alto', mai 'basso'/'medio'.
        for caso in CASI_CRITICI:
            esito = self.agent.predict(caso)
            self.assertEqual(
                esito.classe,
                "alto",
                msg=f"caso critico declassato: {caso}",
            )
            self.assertFalse(esito.errore)
            self.assertNotEqual(esito.classe, "basso")
            self.assertNotEqual(esito.classe, "medio")

    def test_input_critico_coerente_con_safety_rules(self):
        # La decisione dell'agente coincide ESATTAMENTE con la
        # classificazione rule-based (stessa fonte).
        for caso in CASI_CRITICI:
            self.assertEqual(
                self.agent.predict(caso).classe, classifica_rischio(caso)
            )

    def test_input_invalido_errore(self):
        for caso in CASI_INVALIDI:
            esito = self.agent.predict(caso)
            self.assertEqual(esito.classe, CLASSE_ERRORE, msg=f"caso {caso}")
            self.assertTrue(esito.errore, msg=f"caso {caso}")
            self.assertIsNone(esito.probabilita)
            self.assertTrue(esito.messaggio)

    def test_accetta_vital_parameters(self):
        v = VitalParameters(120.0, 80.0, 75.0, 36.8, 98.0, 95.0)
        self.assertEqual(self.agent.predict(v).classe, "basso")

        critico = VitalParameters(185.0, 110.0, 80.0, 36.8, 98.0, 95.0)
        self.assertEqual(self.agent.predict(critico).classe, "alto")

        invalido = VitalParameters(90.0, 90.0, 75.0, 36.8, 98.0, 95.0)
        esito = self.agent.predict(invalido)
        self.assertEqual(esito.classe, CLASSE_ERRORE)
        self.assertTrue(esito.errore)


class TestAnalysisService(unittest.TestCase):
    """Contratto del servizio (ServiceOutcome)."""

    def setUp(self):
        self.servizio = AnalysisService()

    def test_output_è_un_service_outcome(self):
        esito = self.servizio.analizza(vp())
        self.assertIsInstance(esito, ServiceOutcome)

    def test_caso_normale_basso_senza_allerta(self):
        esito = self.servizio.analizza(vp())
        self.assertEqual(esito.classe, "basso")
        self.assertFalse(esito.allerta_medico)
        self.assertEqual(esito.errori, [])
        self.assertTrue(esito.messaggio_paziente)

    def test_caso_critico_allerta_medico(self):
        esito = self.servizio.analizza(vp(glicemia=55.0))
        self.assertEqual(esito.classe, "alto")
        self.assertTrue(esito.allerta_medico)
        self.assertEqual(esito.errori, [])

    def test_input_invalido_allerta_medico_ed_errori(self):
        esito = self.servizio.analizza(vp(glicemia=10.0))
        self.assertEqual(esito.classe, CLASSE_ERRORE)
        self.assertTrue(esito.allerta_medico)
        self.assertGreater(len(esito.errori), 0)

    def test_storico_in_memoria(self):
        self.servizio.analizza(vp())
        self.servizio.analizza(vp(glicemia=55.0))
        self.servizio.analizza(vp(glicemia=10.0))
        self.assertEqual(len(self.servizio.ottieni_storico()), 3)
        self.assertEqual(self.servizio.ultima["classe"], CLASSE_ERRORE)

    def test_servizio_non_declassa_mai_casi_critici(self):
        for caso in CASI_CRITICI:
            esito = self.servizio.analizza(caso)
            self.assertEqual(
                esito.classe,
                "alto",
                msg=f"servizio ha declassato il caso {caso}",
            )
            self.assertTrue(esito.allerta_medico)


if __name__ == "__main__":
    unittest.main()
