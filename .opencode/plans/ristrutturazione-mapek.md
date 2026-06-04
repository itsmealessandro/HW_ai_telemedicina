# Ristrutturazione Microservizi MAPE-K

## Obiettivo

Separare il monolite Streamlit in 5 servizi Docker seguendo il modello **MAPE-K** (Monitor-Analyze-Plan-Execute-Knowledge), con comunicazione via Redis e frontend Alpine.js.

---

## Architettura

```
compose.yaml
┌────────────────────────────────────────────────────────────────────────────┐
│                         MAPE-K LOOP                                         │
│                                                                            │
│  ┌────────────┐     ┌──────────────┐     ┌──────────────┐                 │
│  │  MONITOR   │     │  ANALYZE     │     │   EXECUTE    │                 │
│  │  :8001     │────▶│  + PLAN      │────▶│  :8020       │                 │
│  │            │     │  :8010       │     │              │                 │
│  │Simulatore  │     │Q-table       │     │WebSocket     │                 │
│  │15 pazienti │     │predici_azione│     │REST API      │                 │
│  │scrittura   │     │safety        │     │storico       │                 │
│  │parametri   │     │override      │     │notifiche     │                 │
│  └──────┬─────┘     └──────┬───────┘     └──────┬───────┘                 │
│         │                  │                    │                          │
│         └──────────────────┴────────────────────┘                          │
│                                    │                                       │
│                            ┌───────▼────────┐                             │
│                            │     REDIS      │                             │
│                            │  patients:*     │                             │
│                            └────────────────┘                             │
│                                                                            │
│  ┌─────────────────────────────────────────────┐                          │
│  │              FRONTEND                        │                          │
│  │  Nginx :80 + Alpine.js SPA                  │                          │
│  │  Mostra M→A→P→E in card paziente            │                          │
│  └─────────────────────────────────────────────┘                          │
│                                                                            │
│  Servizio KNOWLEDGE (K): Q-table .pkl + DB SQLite (volumi condivisi)      │
└────────────────────────────────────────────────────────────────────────────┘
```

## Servizi

| Servizio | Container | Ruolo MAPE-K | Porta |
|----------|-----------|-------------|-------|
| **monitor** | `docker/monitor.Dockerfile` | **M** — Genera parametri 15 pazienti, varia ogni 3s, scrive su Redis `patients:state` | `8001` |
| **analyzer** | `docker/analyzer.Dockerfile` | **A+P** — Legge `patients:state`, discretizza, Q-table → Q-values, argmax + safety override, scrive `patients:qvalues` + `patients:actions` | `8010` |
| **executor** | `docker/executor.Dockerfile` | **E** — Legge tutti i `patients:*`, assembla stato arricchito, storico in memoria, push WebSocket, REST API | `8020` |
| **frontend** | `docker/nginx.Dockerfile` | **UI** — Nginx serve `static/`, proxy `/api/` → executor | `80` |
| **redis** | `redis:alpine` | **Bus dati** — Redis pub/sub-like (chiavi) tra servizi | `6379` |
| **knowledge** | Volumi condivisi | **K** — `data/models/q_table.pkl` montato in analyzer, `data/telemedicina.db` storico | - |

## Flusso dati (un ciclo MAPE-K ogni 3s)

```
  MONITOR                    ANALYZE + PLAN                  EXECUTE
┌──────────────┐          ┌──────────────────┐          ┌──────────────┐
│ 15 pazienti  │          │ Q-table lookup    │          │ Assemblea     │
│ thread 3s    │  scrive  │ discretizza stato │  scrive  │ stato + push  │
│ delta        │────────▶│ argmax + safety   │────────▶│ WS → frontend │
│ gaussiano    │  Redis   │ override          │  Redis   │ storico DB    │
└──────────────┘          └──────────────────┘          └──────────────┘
       │                         │                            │
  patients:state           patients:qvalues             patients:enriched
                           patients:actions             (letto da executor)
```

## Redis keys

| Key | Scritto da | Formato |
|-----|-----------|---------|
| `patients:state` | monitor | `{P001: {nome, severita, parametri{...}}, ...}` |
| `patients:qvalues` | analyzer | `{P001: {monitoring: -2.3, contatta_medico: 1.5, ...}, ...}` |
| `patients:actions` | analyzer | `{P001: {azione, livello_rischio, raccomandazioni, allerta_medico}, ...}` |

## Endpoint executor (esposti al frontend)

| Endpoint | Metodo | Cosa restituisce |
|----------|--------|-----------------|
| `/api/stats` | GET | Statistiche: total, rischio_basso/medio/alto, alert |
| `/api/patients` | GET | Lista tutti pazienti con stato arricchito (parametri + qvalues + azione) |
| `/api/patients/{id}` | GET | Singolo paziente + storico ultime 30 rilevazioni |
| `/api/ws` | WebSocket | Push automatico ogni 3s dello stato completo |

## Componenti da creare

```
telemedicina/
├── compose.yaml
├── requirements.txt                       (fastapi, uvicorn, redis, numpy)
├── nginx.conf
│
├── docker/
│   ├── monitor.Dockerfile
│   ├── analyzer.Dockerfile
│   ├── executor.Dockerfile
│   └── nginx.Dockerfile
│
├── src/telemedicina/
│   ├── config.py                          (aggiunto REDIS_HOST/PORT)
│   ├── models/vital_parameters.py         (esistente)
│   ├── agents/rl_agent.py                (esistente)
│   │
│   ├── monitor/
│   │   ├── __init__.py
│   │   └── main.py                        (M — simulazione pazienti + Redis)
│   │
│   ├── analyzer/
│   │   ├── __init__.py
│   │   └── main.py                        (A+P — Q-table + safety + Redis)
│   │
│   └── executor/
│       ├── __init__.py
│       ├── main.py                        (E — REST + WS + storico)
│       └── ws_manager.py                  (gestione connessioni WebSocket)
│
├── static/
│   ├── index.html                         (Alpine.js SPA — MAPE-K visibile)
│   ├── style.css
│   └── app.js
│
└── data/
    ├── telemedicina.db
    └── models/q_table.pkl
```

## UI — Card paziente con MAPE-K

Per ogni paziente la card mostra esplicitamente le 4 fasi:

```
┌──────────────────────────────────────────────────┐
│  Mario Rossi                         🔴 ALTO     │
│                                                    │
│  ┌─ MONITOR ──────────────────────────────┐       │
│  │ PA 185/118 · FC 133 · T 39.8°C         │       │
│  │ SpO2 87% · Glicemia 210                │       │
│  └────────────────────────────────────────┘       │
│                                                    │
│  ┌─ ANALYZE ───────────────────────────────┐      │
│  │ Q-values:                                 │      │
│  │ monitoring   ██░░░░░░  -2.3               │      │
│  │ contatta_m   █████░░░  +1.5               │      │
│  │ pronto_socc  ████████  +8.2               │      │
│  │ emergenza    ██████████ +15.0  ← MAX     │      │
│  └──────────────────────────────────────────┘      │
│                                                    │
│  ┌─ PLAN ──────────────────────────────────┐       │
│  │ 🚨 EMERGENZA                             │       │
│  │ Chiamare il 118 IMMEDIATAMENTE           │       │
│  └──────────────────────────────────────────┘       │
│                                                    │
│  ┌─ EXECUTE ───────────────────────────────┐       │
│  │ ✅ WS push · ✅ Allerta · 🕐 2s fa      │       │
│  └──────────────────────────────────────────┘       │
│                                                    │
│  [ Dettaglio → ]                                   │
└──────────────────────────────────────────────────┘
```

## Codice da rimuovere

- `dashboard.py` — sostituito dal nuovo frontend Alpine.js
- `src/telemedicina/monitoring/` — sostituito da `monitor/`, `analyzer/`, `executor/`

## Avvio

```bash
docker compose up --build
# → http://localhost:80
```
