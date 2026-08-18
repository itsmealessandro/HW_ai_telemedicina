"""
rl_environment.py - Ambiente simulato per l'addestramento RL.

Simula un paziente con uno stato nascosto (severita) e fornisce
reward in base alla bonta dell'azione scelta dall'agente.
"""

import random
from telemedicina.models.vital_parameters import VitalParameters

SEVERITA = ["NORMALE", "ATTENZIONE", "CRITICO", "EMERGENZA"]

# reward[azione][severita]
# azioni: 0=monitoring, 1=contatta_medico, 2=pronto_soccorso, 3=emergenza
MATRICE_REWARD = [
    [10, -5, -15, -20],   # monitoring
    [-1, 10,   5, -10],   # contatta_medico
    [-5,  2,  10,  -5],   # pronto_soccorso
    [-10, -5,  -2,  10],  # emergenza
]


def _genera_parametri(severita: int) -> VitalParameters:
    """Genera parametri vitali coerenti con la severita."""
    if severita == 0:  # NORMALE
        return VitalParameters(
            pressione_sistolica=random.uniform(100, 130),
            pressione_diastolica=random.uniform(65, 85),
            frequenza_cardiaca=random.uniform(65, 90),
            temperatura=random.uniform(36.2, 37.2),
            saturazione_ossigeno=random.uniform(96, 100),
            glicemia=random.uniform(75, 110),
        )
    elif severita == 1:  # ATTENZIONE
        return VitalParameters(
            pressione_sistolica=random.uniform(140, 160),
            pressione_diastolica=random.uniform(90, 100),
            frequenza_cardiaca=random.uniform(90, 105),
            temperatura=random.uniform(37.3, 38.2),
            saturazione_ossigeno=random.uniform(93, 95),
            glicemia=random.uniform(140, 170),
        )
    elif severita == 2:  # CRITICO
        return VitalParameters(
            pressione_sistolica=random.uniform(165, 179),
            pressione_diastolica=random.uniform(105, 109),
            frequenza_cardiaca=random.uniform(106, 119),
            temperatura=random.uniform(38.3, 39.2),
            saturazione_ossigeno=random.uniform(90, 92),
            glicemia=random.uniform(60, 69),
        )
    else:  # EMERGENZA
        return VitalParameters(
            pressione_sistolica=random.uniform(180, 200),
            pressione_diastolica=random.uniform(110, 130),
            frequenza_cardiaca=random.uniform(120, 140),
            temperatura=random.uniform(39.5, 41.0),
            saturazione_ossigeno=random.uniform(80, 89),
            glicemia=random.uniform(200, 250),
        )


class SimPatientEnv:
    """Ambiente simulato che modella un paziente."""

    def __init__(self):
        self.severita = 0
        self.parametri = None

    def reset(self):
        """Resetta l'ambiente con una nuova severita casuale."""
        self.severita = random.randint(0, 3)
        self.parametri = _genera_parametri(self.severita)
        return self.severita

    def step(self, azione: int):
        """
        Applica un'azione e restituisce reward e nuovo stato.

        Returns:
            tuple: (reward, nuova_severita, done)
        """
        reward = MATRICE_REWARD[azione][self.severita]

        if reward >= 10:
            self.severita = max(0, self.severita - random.randint(1, 2))
        elif reward >= 5:
            self.severita = max(0, self.severita - 1)
        elif reward < -10:
            self.severita = min(3, self.severita + random.randint(1, 2))
        elif reward < 0:
            self.severita = min(3, self.severita + 1)

        self.parametri = _genera_parametri(self.severita)
        done = random.random() < 0.4

        return reward, self.severita, done
