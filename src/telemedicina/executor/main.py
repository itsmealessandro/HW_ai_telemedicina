import json
import os
import threading
import time
from collections import deque

import redis

from telemedicina import config
from telemedicina.executor.ws_manager import WSManager

MAX_STORICO = 30


def _compute_stats(patients):
    tot = len(patients)
    basso = sum(1 for p in patients if p["livello_rischio"] == "basso")
    medio = sum(1 for p in patients if p["livello_rischio"] == "medio")
    alto = sum(1 for p in patients if p["livello_rischio"] == "alto")
    alert = sum(1 for p in patients if p.get("allerta_medico", False))
    return {
        "totale": tot,
        "rischio_basso": basso,
        "rischio_medio": medio,
        "rischio_alto": alto,
        "alert": alert,
    }


def _enrich(pid, state_data, qv_data, act_data):
    p = dict(state_data.get(pid, {}))
    qv = qv_data.get(pid, {})
    act = act_data.get(pid, {})
    p["q_values"] = qv
    p["azione"] = act.get("azione", "monitoring")
    p["livello_rischio"] = act.get("livello_rischio", "basso")
    p["raccomandazioni"] = act.get("raccomandazioni", "")
    p["allerta_medico"] = act.get("allerta_medico", False)
    p["safety_override"] = act.get("safety_override", False)
    return p


def _run_orchestrator(rconn, ws_manager, cache, lock, stop_event):
    while not stop_event.is_set():
        raw_state = rconn.get("patients:state")
        raw_qv = rconn.get("patients:qvalues")
        raw_act = rconn.get("patients:actions")

        if not (raw_state and raw_qv and raw_act):
            time.sleep(3)
            continue

        try:
            state_data = json.loads(raw_state)
            qv_data = json.loads(raw_qv)
            act_data = json.loads(raw_act)
        except (json.JSONDecodeError, TypeError):
            time.sleep(3)
            continue

        patients = []
        for pid in state_data:
            enriched = _enrich(pid, state_data, qv_data, act_data)
            patients.append(enriched)

        stats = _compute_stats(patients)

        with lock:
            cache["patients"] = patients
            cache["stats"] = stats
            for p in patients:
                pid = p["paziente_id"]
                if pid not in cache["storico"]:
                    cache["storico"][pid] = deque(maxlen=MAX_STORICO)
                cache["storico"][pid].append({
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "pressione_sistolica": p.get("pressione_sistolica"),
                    "pressione_diastolica": p.get("pressione_diastolica"),
                    "frequenza_cardiaca": p.get("frequenza_cardiaca"),
                    "temperatura": p.get("temperatura"),
                    "saturazione_ossigeno": p.get("saturazione_ossigeno"),
                    "glicemia": p.get("glicemia"),
                    "q_values": p.get("q_values", {}),
                    "azione": p.get("azione"),
                    "livello_rischio": p.get("livello_rischio"),
                })

            patients_with_history = []
            for p in patients:
                p_copy = dict(p)
                p_copy["storico"] = list(cache["storico"].get(p["paziente_id"], []))[-10:]
                patients_with_history.append(p_copy)

        try:
            import asyncio
            loop = asyncio.new_event_loop()
            loop.run_until_complete(ws_manager.broadcast({
                "type": "update",
                "patients": patients_with_history,
                "stats": stats,
            }))
            loop.close()
        except Exception:
            pass

        time.sleep(3)


def main():
    import uvicorn
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware

    port = int(os.environ.get("SERVICE_PORT", "8020"))
    app = FastAPI(title="Executor (E) — MAPE-K")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    rconn = redis.Redis(host=config.REDIS_HOST, port=config.REDIS_PORT, decode_responses=True)
    stop_event = threading.Event()
    ws_manager = WSManager()

    cache = {"patients": [], "stats": {}, "storico": {}}
    lock = threading.Lock()

    thread = threading.Thread(
        target=_run_orchestrator, args=(rconn, ws_manager, cache, lock, stop_event), daemon=True
    )
    thread.start()

    # ── REST endpoints ──

    @app.get("/api/stats")
    def get_stats():
        with lock:
            return cache.get("stats", {})

    @app.get("/api/patients")
    def get_patients():
        with lock:
            return cache.get("patients", [])

    @app.get("/api/patients/{pid}")
    def get_patient(pid: str):
        with lock:
            patients = cache.get("patients", [])
            storico = cache.get("storico", {}).get(pid, [])
            for p in patients:
                if p["paziente_id"] == pid:
                    return {"patient": p, "storico": list(storico)}
            return {"error": "Patient not found"}, 404

    @app.get("/health")
    def health():
        return {"status": "ok", "ws_connections": ws_manager.count}

    # ── WebSocket ──

    @app.websocket("/api/ws")
    async def websocket_endpoint(ws: WebSocket):
        await ws.accept()
        await ws_manager.connect(ws)
        try:
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            await ws_manager.disconnect(ws)
        except Exception:
            await ws_manager.disconnect(ws)

    @app.on_event("shutdown")
    def shutdown():
        stop_event.set()

    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
