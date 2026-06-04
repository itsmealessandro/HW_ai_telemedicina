import random
import threading
import time
from collections import deque
from datetime import datetime
from typing import Optional

from telemedicina.agents.rl_agent import QLearningAgent
from telemedicina.models.vital_parameters import VitalParameters

SEVERITA = ["NORMALE", "ATTENZIONE", "CRITICO", "EMERGENZA"]

NOMI_PAZIENTI = [
    "Mario Rossi", "Laura Bianchi", "Giuseppe Verdi", "Anna Neri",
    "Antonio Esposito", "Maria Romano", "Francesco Colombo", "Paola Ricci",
    "Roberto Galli", "Elena Conti", "Marco Costa", "Sofia Moretti",
    "Luca Marino", "Giulia Barbieri", "Davide Fontana",
]

PA_TEMPLATES = {
    0: ((100, 130), (65, 85), (65, 90), (36.2, 37.2), (96, 100), (75, 110)),
    1: ((140, 160), (90, 100), (90, 105), (37.3, 38.2), (93, 95), (140, 170)),
    2: ((165, 179), (105, 109), (106, 119), (38.3, 39.2), (90, 92), (60, 69)),
    3: ((180, 200), (110, 130), (120, 140), (39.5, 41.0), (80, 89), (200, 250)),
}

DELTA = [
    (2.0, 1.5, 1.5, 0.1, 0.8, 4.0),
]

PROB_CAMBIO_SEVERITA = 0.08
PROB_PEGGIORA = 0.5
MAX_STORICO = 30


def _genera_parametri(severita: int) -> VitalParameters:
    (pa_sys_lo, pa_sys_hi), (pa_dia_lo, pa_dia_hi), (fc_lo, fc_hi), \
        (temp_lo, temp_hi), (spo2_lo, spo2_hi), (glu_lo, glu_hi) = PA_TEMPLATES[severita]
    return VitalParameters(
        pressione_sistolica=round(random.uniform(pa_sys_lo, pa_sys_hi), 1),
        pressione_diastolica=round(random.uniform(pa_dia_lo, pa_dia_hi), 1),
        frequenza_cardiaca=round(random.uniform(fc_lo, fc_hi), 1),
        temperatura=round(random.uniform(temp_lo, temp_hi), 1),
        saturazione_ossigeno=round(random.uniform(spo2_lo, spo2_hi), 1),
        glicemia=round(random.uniform(glu_lo, glu_hi), 1),
    )


def _applica_delta(vp: VitalParameters, severita: int) -> VitalParameters:
    d = DELTA[0]
    (pa_sys_lo, pa_sys_hi), (pa_dia_lo, pa_dia_hi), (fc_lo, fc_hi), \
        (temp_lo, temp_hi), (spo2_lo, spo2_hi), (glu_lo, glu_hi) = PA_TEMPLATES[severita]
    return VitalParameters(
        pressione_sistolica=round(max(pa_sys_lo, min(pa_sys_hi, vp.pressione_sistolica + random.gauss(0, d[0]))), 1),
        pressione_diastolica=round(max(pa_dia_lo, min(pa_dia_hi, vp.pressione_diastolica + random.gauss(0, d[1]))), 1),
        frequenza_cardiaca=round(max(fc_lo, min(fc_hi, vp.frequenza_cardiaca + random.gauss(0, d[2]))), 1),
        temperatura=round(max(temp_lo, min(temp_hi, vp.temperatura + random.gauss(0, d[3]))), 1),
        saturazione_ossigeno=round(max(spo2_lo, min(spo2_hi, vp.saturazione_ossigeno + random.gauss(0, d[4]))), 1),
        glicemia=round(max(glu_lo, min(glu_hi, vp.glicemia + random.gauss(0, d[5]))), 1),
    )


class PazienteSimulato:
    def __init__(self, paziente_id: str, nome: str, severita: int):
        self.paziente_id = paziente_id
        self.nome = nome
        self.severita = severita
        self.parametri = _genera_parametri(severita)
        self.analisi: dict = {}
        self.storico: deque = deque(maxlen=MAX_STORICO)

    def aggiorna(self, agente: QLearningAgent):
        if random.random() < PROB_CAMBIO_SEVERITA:
            if random.random() < PROB_PEGGIORA:
                self.severita = min(3, self.severita + 1)
            else:
                self.severita = max(0, self.severita - 1)

        self.parametri = _applica_delta(self.parametri, self.severita)

        self.analisi = agente.predici_azione(self.parametri)

        self.storico.append({
            "timestamp": datetime.now().isoformat(sep=" ", timespec="seconds"),
            "parametri": self.parametri,
            "azione": self.analisi.get("dettagli_analisi", {}).get("azione", "monitoring"),
            "livello_rischio": self.analisi.get("livello_rischio", "basso"),
            "raccomandazioni": self.analisi.get("raccomandazioni", ""),
            "allerta_medico": self.analisi.get("allerta_medico", False),
        })

    def ultime_azioni(self, n: int = 10):
        return list(self.storico)[-n:]


class PatientMonitor:
    _instance: Optional["PatientMonitor"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, num_pazienti: int = 15):
        if self._initialized:
            return
        self._initialized = True
        self.num_pazienti = num_pazienti
        self.pazienti: dict[str, PazienteSimulato] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._agente = QLearningAgent()
        self._agente_ok = self._agente.carica_q_table()
        self._cicli = 0

    @property
    def agente_pronto(self) -> bool:
        return self._agente_ok

    def avvia(self):
        if self._running:
            return
        self._running = True

        for i in range(self.num_pazienti):
            pid = f"P{i+1:03d}"
            nome = NOMI_PAZIENTI[i] if i < len(NOMI_PAZIENTI) else f"Paziente {pid}"
            severita = random.choices([0, 1, 2, 3], weights=[40, 30, 20, 10])[0]
            self.pazienti[pid] = PazienteSimulato(pid, nome, severita)

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def ferma(self):
        self._running = False

    def _run(self):
        while self._running:
            with self._lock:
                for p in self.pazienti.values():
                    p.aggiorna(self._agente)
                self._cicli += 1
            time.sleep(3)

    def get_stato(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "paziente_id": p.paziente_id,
                    "nome": p.nome,
                    "severita": p.severita,
                    "severita_label": SEVERITA[p.severita],
                    "pressione_sistolica": p.parametri.pressione_sistolica,
                    "pressione_diastolica": p.parametri.pressione_diastolica,
                    "frequenza_cardiaca": p.parametri.frequenza_cardiaca,
                    "temperatura": p.parametri.temperatura,
                    "saturazione_ossigeno": p.parametri.saturazione_ossigeno,
                    "glicemia": p.parametri.glicemia,
                    "livello_rischio": p.analisi.get("livello_rischio", "basso"),
                    "azione": p.analisi.get("dettagli_analisi", {}).get("azione", "monitoring"),
                    "raccomandazioni": p.analisi.get("raccomandazioni", ""),
                    "allerta_medico": p.analisi.get("allerta_medico", False),
                }
                for p in self.pazienti.values()
            ]

    def get_paziente(self, paziente_id: str) -> Optional[dict]:
        for p in self.pazienti.values():
            if p.paziente_id == paziente_id:
                storico = p.ultime_azioni(20)
                return {
                    "paziente_id": p.paziente_id,
                    "nome": p.nome,
                    "severita": p.severita,
                    "severita_label": SEVERITA[p.severita],
                    "pressione_sistolica": p.parametri.pressione_sistolica,
                    "pressione_diastolica": p.parametri.pressione_diastolica,
                    "frequenza_cardiaca": p.parametri.frequenza_cardiaca,
                    "temperatura": p.parametri.temperatura,
                    "saturazione_ossigeno": p.parametri.saturazione_ossigeno,
                    "glicemia": p.parametri.glicemia,
                    "livello_rischio": p.analisi.get("livello_rischio", "basso"),
                    "azione": p.analisi.get("dettagli_analisi", {}).get("azione", "monitoring"),
                    "raccomandazioni": p.analisi.get("raccomandazioni", ""),
                    "allerta_medico": p.analisi.get("allerta_medico", False),
                    "storico": storico,
                }
        return None

    def get_statistiche(self) -> dict:
        stato = self.get_stato()
        totale = len(stato)
        alert = sum(1 for p in stato if p["allerta_medico"])
        rischio_basso = sum(1 for p in stato if p["livello_rischio"] == "basso")
        rischio_medio = sum(1 for p in stato if p["livello_rischio"] == "medio")
        rischio_alto = sum(1 for p in stato if p["livello_rischio"] == "alto")
        in_emergenza = sum(1 for p in stato if p["azione"] == "emergenza")
        return {
            "totale": totale,
            "alert": alert,
            "rischio_basso": rischio_basso,
            "rischio_medio": rischio_medio,
            "rischio_alto": rischio_alto,
            "in_emergenza": in_emergenza,
            "cicli": self._cicli,
        }
