import json
import os
import random
import threading
import time

import redis

from telemedicina import config
from telemedicina.models.vital_parameters import VitalParameters

SEVERITA_LABEL = ["NORMALE", "ATTENZIONE", "CRITICO", "EMERGENZA"]

PA_TEMPLATES = {
    0: ((100, 130), (65, 85), (65, 90), (36.2, 37.2), (96, 100), (75, 110)),
    1: ((140, 160), (90, 100), (90, 105), (37.3, 38.2), (93, 95), (140, 170)),
    2: ((165, 179), (105, 109), (106, 119), (38.3, 39.2), (90, 92), (60, 69)),
    3: ((180, 200), (110, 130), (120, 140), (39.5, 41.0), (80, 89), (200, 250)),
}

DELTA = (2.0, 1.5, 1.5, 0.1, 0.8, 4.0)
PROB_CAMBIO_SEVERITA = 0.08
PROB_PEGGIORA = 0.5

NOMI_PAZIENTI = [
    "Mario Rossi", "Laura Bianchi", "Giuseppe Verdi", "Anna Neri",
    "Antonio Esposito", "Maria Romano", "Francesco Colombo", "Paola Ricci",
    "Roberto Galli", "Elena Conti", "Marco Costa", "Sofia Moretti",
    "Luca Marino", "Giulia Barbieri", "Davide Fontana",
]


def _genera_parametri(severita):
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


def _applica_delta(vp, severita):
    (pa_sys_lo, pa_sys_hi), (pa_dia_lo, pa_dia_hi), (fc_lo, fc_hi), \
        (temp_lo, temp_hi), (spo2_lo, spo2_hi), (glu_lo, glu_hi) = PA_TEMPLATES[severita]
    d = DELTA
    return VitalParameters(
        pressione_sistolica=round(max(pa_sys_lo, min(pa_sys_hi, vp.pressione_sistolica + random.gauss(0, d[0]))), 1),
        pressione_diastolica=round(max(pa_dia_lo, min(pa_dia_hi, vp.pressione_diastolica + random.gauss(0, d[1]))), 1),
        frequenza_cardiaca=round(max(fc_lo, min(fc_hi, vp.frequenza_cardiaca + random.gauss(0, d[2]))), 1),
        temperatura=round(max(temp_lo, min(temp_hi, vp.temperatura + random.gauss(0, d[3]))), 1),
        saturazione_ossigeno=round(max(spo2_lo, min(spo2_hi, vp.saturazione_ossigeno + random.gauss(0, d[4]))), 1),
        glicemia=round(max(glu_lo, min(glu_hi, vp.glicemia + random.gauss(0, d[5]))), 1),
    )


class Paziente:
    def __init__(self, pid, nome, severita):
        self.paziente_id = pid
        self.nome = nome
        self.severita = severita
        self.parametri = _genera_parametri(severita)

    def aggiorna(self):
        if random.random() < PROB_CAMBIO_SEVERITA:
            if random.random() < PROB_PEGGIORA:
                self.severita = min(3, self.severita + 1)
            else:
                self.severita = max(0, self.severita - 1)
        self.parametri = _applica_delta(self.parametri, self.severita)

    def to_dict(self):
        return {
            "paziente_id": self.paziente_id,
            "nome": self.nome,
            "severita": self.severita,
            "severita_label": SEVERITA_LABEL[self.severita],
            "pressione_sistolica": self.parametri.pressione_sistolica,
            "pressione_diastolica": self.parametri.pressione_diastolica,
            "frequenza_cardiaca": self.parametri.frequenza_cardiaca,
            "temperatura": self.parametri.temperatura,
            "saturazione_ossigeno": self.parametri.saturazione_ossigeno,
            "glicemia": self.parametri.glicemia,
        }


def _build_state(patients):
    return {p.paziente_id: p.to_dict() for p in patients}


def _run_simulator(rconn, patients, stop_event):
    while not stop_event.is_set():
        for p in patients:
            p.aggiorna()
        rconn.set("patients:state", json.dumps(_build_state(patients)))
        time.sleep(3)


def main():
    import uvicorn
    from fastapi import FastAPI

    port = int(os.environ.get("SERVICE_PORT", "8001"))
    app = FastAPI(title="Monitor (M) — MAPE-K")

    rconn = redis.Redis(host=config.REDIS_HOST, port=config.REDIS_PORT, decode_responses=True)
    stop_event = threading.Event()

    patients = []
    for i in range(15):
        pid = f"P{i+1:03d}"
        nome = NOMI_PAZIENTI[i]
        severita = random.choices([0, 1, 2, 3], weights=[40, 30, 20, 10])[0]
        patients.append(Paziente(pid, nome, severita))

    thread = threading.Thread(target=_run_simulator, args=(rconn, patients, stop_event), daemon=True)
    thread.start()

    @app.get("/health")
    def health():
        return {"status": "ok", "pazienti": len(patients)}

    @app.on_event("shutdown")
    def shutdown():
        stop_event.set()

    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
