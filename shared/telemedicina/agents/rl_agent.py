"""
rl_agent.py - Agente Q-learning per telemedicina.

Discretizza i parametri vitali in uno stato discreto e apprende
la policy ottima tramite Q-learning con epsilon-greedy.
"""

import random
import pickle
import numpy as np
from telemedicina.models.vital_parameters import VitalParameters
from telemedicina import config

AZIONI = ["monitoring", "contatta_medico", "pronto_soccorso", "emergenza"]

N_BINS = [4, 4, 3, 3, 3, 3]
N_STATI = 4 * 4 * 3 * 3 * 3 * 3  # 1296

SOGLIE = [
    [90, 140, 180],
    [60, 90, 110],
    [60, 100, 999],
    [36.0, 37.5, 999],
    [90, 95, 999],
    [70, 140, 999],
]


def _bin_index(valore: float, soglie: list) -> int:
    for i, s in enumerate(soglie):
        if valore < s:
            return i
    return len(soglie)


class QLearningAgent:
    """Agente che apprende tramite Q-learning."""

    def __init__(self, alpha=None, gamma=None, epsilon=None):
        self.alpha = alpha or config.RL_ALPHA
        self.gamma = gamma or config.RL_GAMMA
        self.epsilon = epsilon or config.RL_EPSILON_INIT
        self.epsilon_decay = config.RL_EPSILON_DECAY
        self.epsilon_min = config.RL_EPSILON_MIN
        self.q_table = np.zeros((N_STATI, len(AZIONI)))

    def discretizza(self, parametri: VitalParameters) -> int:
        """Converte parametri vitali in un indice di stato."""
        bins = [
            _bin_index(parametri.pressione_sistolica, SOGLIE[0]),
            _bin_index(parametri.pressione_diastolica, SOGLIE[1]),
            _bin_index(parametri.frequenza_cardiaca, SOGLIE[2]),
            _bin_index(parametri.temperatura, SOGLIE[3]),
            _bin_index(parametri.saturazione_ossigeno, SOGLIE[4]),
            _bin_index(parametri.glicemia, SOGLIE[5]),
        ]
        idx = 0
        stride = 1
        for i in range(5, -1, -1):
            idx += bins[i] * stride
            stride *= N_BINS[i]
        return idx

    def scegli_azione(self, stato: int, training: bool = True) -> int:
        """Epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randint(0, len(AZIONI) - 1)
        return int(np.argmax(self.q_table[stato]))

    def impara(self, stato, azione, reward, stato_next, done):
        """Q-learning update."""
        target = reward
        if not done:
            target += self.gamma * float(np.max(self.q_table[stato_next]))
        self.q_table[stato, azione] += self.alpha * (
            target - self.q_table[stato, azione]
        )

    def decadi_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def salva_q_table(self, path: str = None):
        path = path or config.RL_QTABLE_PATH
        with open(path, "wb") as f:
            pickle.dump(self.q_table, f)

    def carica_q_table(self, path: str = None):
        path = path or config.RL_QTABLE_PATH
        try:
            with open(path, "rb") as f:
                self.q_table = pickle.load(f)
            return True
        except (FileNotFoundError, pickle.UnpicklingError):
            return False

    def _safety_override(self, parametri: VitalParameters) -> dict:
        crit = parametri

        if crit.pressione_sistolica >= 180 or crit.pressione_diastolica >= 110:
            return {
                "livello_rischio": "alto",
                "anomalie": ["Crisi ipertensiva: pressione pericolosamente elevata."],
                "raccomandazioni": "ATTENZIONE: Crisi ipertensiva. Recarsi al pronto soccorso per valutazione medica urgente.",
                "allerta_medico": True,
                "dettagli_analisi": {"agente": "rl", "safety_override": "crisi_ipertensiva"},
            }

        if crit.pressione_sistolica < 90 and crit.frequenza_cardiaca > 100:
            return {
                "livello_rischio": "alto",
                "anomalie": ["SHOCK POSSIBILE: Ipotensione con tachicardia compensatoria."],
                "raccomandazioni": "ATTENZIONE: Possibile shock. Chiamare il 118 IMMEDIATAMENTE.",
                "allerta_medico": True,
                "dettagli_analisi": {"agente": "rl", "safety_override": "shock"},
            }

        if crit.frequenza_cardiaca >= 130 or crit.frequenza_cardiaca <= 45:
            return {
                "livello_rischio": "alto",
                "anomalie": [f"Frequenza cardiaca critica: {crit.frequenza_cardiaca} bpm."],
                "raccomandazioni": "ATTENZIONE: Frequenza cardiaca critica. Recarsi al pronto soccorso.",
                "allerta_medico": True,
                "dettagli_analisi": {"agente": "rl", "safety_override": "frequenza_critica"},
            }

        if crit.saturazione_ossigeno < 88:
            return {
                "livello_rischio": "alto",
                "anomalie": [f"Ipossia severa: SpO2={crit.saturazione_ossigeno}%."],
                "raccomandazioni": "ATTENZIONE: Ipossia severa. Chiamare il 118 IMMEDIATAMENTE.",
                "allerta_medico": True,
                "dettagli_analisi": {"agente": "rl", "safety_override": "ipossia_severa"},
            }

        if crit.temperatura >= 39.5:
            return {
                "livello_rischio": "alto",
                "anomalie": [f"Ipertermia critica: {crit.temperatura} C."],
                "raccomandazioni": "ATTENZIONE: Ipertermia critica. Recarsi al pronto soccorso.",
                "allerta_medico": True,
                "dettagli_analisi": {"agente": "rl", "safety_override": "ipertermia_critica"},
            }

        if crit.glicemia <= 55 or crit.glicemia >= 250:
            return {
                "livello_rischio": "alto",
                "anomalie": [f"Glicemia critica: {crit.glicemia} mg/dL."],
                "raccomandazioni": "ATTENZIONE: Glicemia critica. Recarsi al pronto soccorso.",
                "allerta_medico": True,
                "dettagli_analisi": {"agente": "rl", "safety_override": "glicemia_critica"},
            }

        return None

    def predici_azione(self, parametri: VitalParameters) -> dict:
        override = self._safety_override(parametri)
        if override:
            return override

        stato = self.discretizza(parametri)
        azione_idx = self.scegli_azione(stato, training=False)
        azione = AZIONI[azione_idx]

        if azione == "monitoring":
            return {
                "livello_rischio": "basso",
                "anomalie": [],
                "raccomandazioni": "Parametri nella norma. Continuare il monitoraggio regolare.",
                "allerta_medico": False,
                "dettagli_analisi": {"agente": "rl", "azione": azione},
            }
        elif azione == "contatta_medico":
            return {
                "livello_rischio": "medio",
                "anomalie": [],
                "raccomandazioni": "Contattare il proprio medico entro 24 ore e ripetere le misurazioni.",
                "allerta_medico": False,
                "dettagli_analisi": {"agente": "rl", "azione": azione},
            }
        elif azione == "pronto_soccorso":
            return {
                "livello_rischio": "alto",
                "anomalie": [],
                "raccomandazioni": "Recarsi al pronto soccorso per valutazione medica urgente.",
                "allerta_medico": True,
                "dettagli_analisi": {"agente": "rl", "azione": azione},
            }
        else:  # emergenza
            return {
                "livello_rischio": "alto",
                "anomalie": [],
                "raccomandazioni": "Chiamare il 118 IMMEDIATAMENTE. Emergenza medica in corso.",
                "allerta_medico": True,
                "dettagli_analisi": {"agente": "rl", "azione": azione},
            }

    def analizza_parametri(self, parametri: VitalParameters) -> dict:
        return self.predici_azione(parametri)
