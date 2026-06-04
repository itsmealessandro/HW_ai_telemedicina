import json
import os
import threading
import time

import numpy as np
import redis

from telemedicina import config
from telemedicina.agents.rl_agent import QLearningAgent, AZIONI
from telemedicina.models.vital_parameters import VitalParameters


def _run_analysis(rconn, agent, stop_event):
    while not stop_event.is_set():
        raw = rconn.get("patients:state")
        if not raw:
            time.sleep(3)
            continue

        try:
            pazienti = json.loads(raw)
        except json.JSONDecodeError:
            time.sleep(3)
            continue

        qvalues_out = {}
        actions_out = {}

        for pid, pdata in pazienti.items():
            vp = VitalParameters(
                pressione_sistolica=pdata["pressione_sistolica"],
                pressione_diastolica=pdata["pressione_diastolica"],
                frequenza_cardiaca=pdata["frequenza_cardiaca"],
                temperatura=pdata["temperatura"],
                saturazione_ossigeno=pdata["saturazione_ossigeno"],
                glicemia=pdata["glicemia"],
            )

            stato = agent.discretizza(vp)
            qvals = agent.q_table[stato]
            qvalues_out[pid] = {
                act: round(float(qvals[i]), 2) for i, act in enumerate(AZIONI)
            }

            analisi = agent.predici_azione(vp)
            dettagli = analisi.get("dettagli_analisi", {})
            safety = dettagli.get("safety_override")

            if safety:
                MAP_OVERRIDE_AZIONE = {
                    "crisi_ipertensiva": "pronto_soccorso",
                    "shock": "emergenza",
                    "frequenza_critica": "pronto_soccorso",
                    "ipossia_severa": "emergenza",
                    "ipertermia_critica": "pronto_soccorso",
                    "glicemia_critica": "pronto_soccorso",
                }
                azione = MAP_OVERRIDE_AZIONE.get(safety, "pronto_soccorso")
            else:
                azione = dettagli.get("azione", AZIONI[0])

            actions_out[pid] = {
                "azione": azione,
                "livello_rischio": analisi.get("livello_rischio", "basso"),
                "raccomandazioni": analisi.get("raccomandazioni", ""),
                "allerta_medico": analisi.get("allerta_medico", False),
                "safety_override": safety or False,
            }

            if safety:
                azione_idx = AZIONI.index(azione)
                qvalues_out[pid][AZIONI[azione_idx]] = max(
                    qvals[azione_idx] + 10, float(qvals[azione_idx])
                )
                qvalues_out[pid] = dict(
                    sorted(qvalues_out[pid].items(), key=lambda x: x[1], reverse=True)
                )

        rconn.set("patients:qvalues", json.dumps(qvalues_out))
        rconn.set("patients:actions", json.dumps(actions_out))
        time.sleep(3)


def main():
    import uvicorn
    from fastapi import FastAPI

    port = int(os.environ.get("SERVICE_PORT", "8010"))
    app = FastAPI(title="Analyzer+Plan (A+P) — MAPE-K")

    rconn = redis.Redis(host=config.REDIS_HOST, port=config.REDIS_PORT, decode_responses=True)
    stop_event = threading.Event()

    agent = QLearningAgent()
    if not agent.carica_q_table():
        print("ERRORE: Q-table non trovata. Creo tabella vuota (solo safety override).")
        agent.q_table = np.zeros((1296, len(AZIONI)))

    thread = threading.Thread(target=_run_analysis, args=(rconn, agent, stop_event), daemon=True)
    thread.start()

    @app.get("/health")
    def health():
        return {"status": "ok", "qtable_caricata": agent.carica_q_table()}

    @app.on_event("shutdown")
    def shutdown():
        stop_event.set()

    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
