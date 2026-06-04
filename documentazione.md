# Telemedicina – Documentazione

## 1. Panoramica

Il sistema implementa un agente intelligente per telemedicina che:

1. **Gestisce parametri vitali** (pressione arteriosa, frequenza cardiaca, temperatura, SpO2, glicemia)
2. **Restituisce un messaggio di risposta** al paziente con livello di rischio e raccomandazioni
3. **Allerta il medico** in caso di valori significativamente squilibrati
4. **Registra ogni interazione** su database per tracciabilità e analisi storica

---

## 2. Agente Intelligente – Aspetti di IA

L'agente (`agents/intelligent_agent.py`) è un **sistema esperto basato su regole** (symbolic AI / rule-based expert system).

### 2.1 Knowledge Base (Base di Conoscenza)

La conoscenza medica è codificata esplicitamente:

- **Range di normalita**: valori di riferimento per ogni parametro (es. pressione sistolica 90-140 mmHg, frequenza 60-100 bpm)
- **Soglie di gravita**: deviazioni classificate come lievi, moderate o critiche
- **Pattern patologici**: combinazioni di parametri che indicano condizioni cliniche specifiche

### 2.2 Inference Engine (Motore Inferenziale)

L'agente valuta i parametri in sequenza:

1. **Validazione fisiologica**: verifica che i valori siano umanamente possibili (es. frequenza 30-200 bpm)
2. **Identificazione anomalie**: ogni parametro viene confrontato con il range normale e classificato per gravita
3. **Analisi correlazioni**: ricerca pattern critici che coinvolgono piu parametri
4. **Calcolo rischio**: albero decisionale che assegna il livello basso/medio/alto

### 2.3 Pattern Critici Rilevati

| Pattern | Condizione | Rischio |
|---------|-----------|---------|
| Shock ipovolemico | PA < 90 + FC > 100 | ALTO |
| Crisi ipertensiva | PA >= 180 o >= 110 | ALTO |
| Insuff. respiratoria | SpO2 < 92 + FC > 100 | ALTO |
| Infezione sistemica | T >= 38.5 + FC > 100 | ALTO |
| Ipoglicemia severa | Glicemia < 60 + FC > 100 | ALTO |
| Ipertermia critica | T >= 39.5 | ALTO |
| Ipossia severa | SpO2 < 88 | ALTO |

### 2.4 Agente RL (Q-learning) — Componente Centrale

L'agente RL (`agents/rl_agent.py`) implementa **Q-learning tabulare** ed è il motore decisionale principale del sistema. La Q-table, addestrata offline, viene caricata dal servizio **analyzer** all'avvio e interrogata ogni 3 secondi per ogni paziente.

#### Architettura

| Componente | Descrizione |
|-----------|-------------|
| **Stato** | 1296 stati discreti (4×4×3×3×3×3 bins) |
| **Azioni** | `monitoring`, `contatta_medico`, `pronto_soccorso`, `emergenza` |
| **Q-table** | Matrice 1296×4 di valori float appresi via Q-learning |
| **Policy** | Greedy (argmax) in esecuzione, epsilon-greedy in training |
| **Learning rate** α | 0.1 |
| **Discount factor** γ | 0.9 |
| **Epsilon decay** | 1.0 → 0.05 (decay: 0.99996) |

#### Discretizzazione dello Stato

Ogni parametro vitale viene mappato in un bin discreto:

| Parametro | Bins | Soglie |
|-----------|------|--------|
| PA Sistolica | 4 | <90, 90-140, 140-180, ≥180 |
| PA Diastolica | 4 | <60, 60-90, 90-110, ≥110 |
| Frequenza Cardiaca | 3 | <60, 60-100, ≥100 |
| Temperatura | 3 | <36.0, 36.0-37.5, ≥37.5 |
| SpO2 | 3 | <90, 90-95, ≥95 |
| Glicemia | 3 | <70, 70-140, ≥140 |

Il combinate di 6 bins produce un indice univoco da 0 a 1295 che identifica lo stato.

#### Training

La Q-table viene addestrata con 50000 episodi di esplorazione casuale che coprono l'intero range fisiologico dei parametri (PA 70-220, FC 35-180, T 34.5-41.5, SpO2 75-100, Glicemia 30-350). A ogni episodio:

1. Vengono generati parametri vitali casuali
2. Viene calcolato un reward euristico basato su quanti parametri sono in range normale
3. L'agente aggiorna la Q-table con la formula di Q-learning:
   `Q(s,a) ← Q(s,a) + α · [r + γ · max Q(s',a') - Q(s,a)]`
4. ε decade gradualmente per passare da esplorazione a sfruttamento

Risultato: **1281/1296 stati coperti** con valori non-zero.

```bash
# Addestramento
docker compose build analyzer  # include la Q-table pre-addestrata
```

#### Safety Override

Prima di consultare la Q-table, l'agente applica soglie critiche hardcoded. Se attivate, la risposta è 'alto' con allerta medico, bypassando la policy appresa:

| Condizione | Override | Azione |
|-----------|----------|--------|
| PA ≥ 180 o PA dia ≥ 110 | crisi_ipertensiva | pronto_soccorso |
| PA < 90 e FC > 100 | shock | emergenza |
| FC ≥ 130 o ≤ 45 | frequenza_critica | pronto_soccorso |
| SpO2 < 88 | ipossia_severa | emergenza |
| T ≥ 39.5 | ipertermia_critica | pronto_soccorso |
| Glicemia ≤ 55 o ≥ 250 | glicemia_critica | pronto_soccorso |

Nel frontend, quando il safety override è attivo, l'azione corrispondente viene evidenziata e il suo Q-value artificialmente incrementato per mostrare visivamente che l'override ha priorità.

#### Ruolo nel Ciclo MAPE-K

| Fase | Ruolo dell'Agente RL |
|------|---------------------|
| **ANALYZE** | Discretizza i parametri in stato, interroga Q-table → 4 Q-values |
| **PLAN** | Argmax sui Q-values (o safety override) → azione, rischio, raccomandazioni |
| **KNOWLEDGE** | La Q-table è la base di conoscenza appresa (policy) |

---
### 2.5 Ciclo MAPE-K (Autonomic Computing)

L'intero sistema è modellato sul framework **MAPE-K** dell'autonomic computing, dove le fasi sono implementate da microservizi separati che comunicano via Redis.

```
┌─────────────────────────────────────────────────────────────┐
│                     MAPE-K LOOP                               │
│                                                               │
│  MONITOR     ANALYZE       PLAN        EXECUTE                │
│  ┌──────┐   ┌────────┐   ┌───────┐   ┌────────┐              │
│  │Sensor │──▶│Q-table │──▶│Argmax │──▶│WS push │              │
│  │simul. │   │lookup  │   │+safe  │   │+notify │              │
│  └──────┘   └────────┘   └───────┘   └────────┘              │
│                    │                                          │
│                    ▼                                          │
│              ┌─────────────┐                                  │
│              │ KNOWLEDGE    │                                  │
│              │ Q-table.pkl  │                                  │
│              │ DB SQLite    │                                  │
│              │ Safety Rules │                                  │
│              └─────────────┘                                  │
└─────────────────────────────────────────────────────────────┘
```

| Fase | Servizio | Componente AI |
|------|----------|---------------|
| **M**onitor | `monitor:8001` | — (solo simulazione sensori) |
| **A**nalyze | `analyzer:8010` | Discretizzazione + Q-table lookup |
| **P**lan | `analyzer:8010` | Argmax + Safety Override |
| **E**xecute | `executor:8020` | — (solo coordinamento) |
| **K**nowledge | Volume condiviso | Q-table, DB storico, regole di safety |

Il ciclo viene eseguito ogni 3 secondi: il monitor simula i pazienti, l'analyzer li valuta con la Q-table, l'executor distribuisce i risultati via WebSocket.

---

## 3. Architettura

```
CLI (main.py / cli.py)
    |
Services (analysis_service.py)  – orchestrazione del flusso
    |
+-------------------+-------------------+
|                   |                   |
Agent              Database           Notifications
(regole)           (SQLite)           (log file)
```

### 3.1 Componenti

**main.py / cli.py**: Entry point CLI. Raccoglie input, mostra risultati.

**models/vital_parameters.py**: Modello dati `VitalParameters` (dataclass). Contiene i range normali e i metodi di validazione e identificazione anomalie.

**agents/intelligent_agent.py**: Sistema esperto. Analizza i parametri, identifica pattern critici, calcola rischio, genera raccomandazioni e decide l'allerta medico.

**services/analysis_service.py**: Orchestratore. Chiama agente → salva su DB → invia notifiche in sequenza. Separa la logica di coordinamento dalla CLI.

**database/db_manager.py**: Gestore SQLite. Salva interazioni, recupera storico, alert, statistiche e trend.

**utils/notifications.py**: Sistema notifiche. Logga messaggi paziente su `data/logs/notifiche_pazienti.log` e alert medico su `data/logs/alert_medici.log`.

---

## 4. Database

### 4.1 Schema

```sql
CREATE TABLE interazioni (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paziente_id TEXT NOT NULL,
    nome_paziente TEXT NOT NULL,
    pressione_sistolica REAL NOT NULL,
    pressione_diastolica REAL NOT NULL,
    frequenza_cardiaca REAL NOT NULL,
    temperatura REAL NOT NULL,
    saturazione_ossigeno REAL NOT NULL,
    glicemia REAL NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    livello_rischio TEXT NOT NULL,
    raccomandazioni TEXT,
    allerta_medico BOOLEAN DEFAULT 0
);
```

### 4.2 Indici

- `idx_paziente` su `paziente_id`
- `idx_rischio` su `livello_rischio`
- `idx_allerta` su `allerta_medico, livello_rischio`
- `idx_timestamp` su `timestamp DESC`

---

## 5. Flusso di Funzionamento

```
1. PAZIENTE INSERISCE DATI (CLI)
2. Agenete analizza (validazione → anomalie → pattern → rischio → raccomandazioni)
3. Database salva interazione
4. Notifiche: sempre al paziente, solo se rischio ALTO al medico
5. CLI mostra risultati
```

---

## 6. Installazione e Utilizzo

### Requisiti

- Python 3.8+
- `numpy>=1.24` (richiesto sempre, anche per rule-based)

### Esecuzione

```bash
python main.py                # Rule-based
python main.py -RL            # RL, auto-training se Q-table assente
python main.py -RL --build    # RL, forza retrain della Q-table
```

Menu disponibile:
- `1`: Inserisci parametri vitali
- `2`: Storico paziente
- `3`: Alert critici attivi
- `4`: Area medico (pazienti critici)
- `5`: Statistiche sistema
- `0`: Esci

### Test

```bash
python -m tests.test_examples
```

14 test automatici che coprono: validazione parametri, agente intelligente, database, notifiche e integrazione.

---

## 7. Struttura del Progetto

```
progetto_telemedicina/
  main.py                           # Entry point
  src/telemedicina/
    config.py                       # Path centralizzati
    cli.py                          # Interfaccia CLI
    models/vital_parameters.py      # Modello parametri vitali
    agents/intelligent_agent.py     # Agente intelligente (rule-based, primario)
    agents/rl_agent.py              # Agente RL Q-learning (opzionale)
    agents/rl_environment.py        # Ambiente simulato per training RL
    database/db_manager.py          # Gestione database SQLite
    services/analysis_service.py    # Orchestrazione servizi
    utils/notifications.py          # Sistema notifiche
  data/                             # Dati runtime (db, log)
    telemedicina.db
    logs/
  tests/test_examples.py            # Suite di test
  README.md                         # Questo file
  documentazione.md                 # Documentazione completa
```

---

*Prototipo educativo. Non sostituisce il parere medico professionale.*
