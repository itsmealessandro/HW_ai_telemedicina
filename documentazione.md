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

### 2.4 Perche un Sistema a Regole?

- **Interpretabilita totale**: ogni decisione e spiegabile e tracciabile
- **Nessun dato di training**: funziona immediatamente con conoscenza medica codificata
- **Deterministico**: stesso input produce sempre stesso output, requisito per dispositivi medici
- **Manutenibile**: aggiungere o modificare regole e immediato e trasparente

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
- Nessuna libreria esterna (solo standard library)

### Esecuzione

```bash
python main.py
```

Menu disponibile:
- `1`: Inserisci parametri vitali
- `2`: Storico paziente
- `3`: Alert critici attivi
- `4`: Area medico (pazienti critici)
- `5`: Statistiche sistema

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
    agents/intelligent_agent.py     # Agente intelligente (rule-based)
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
