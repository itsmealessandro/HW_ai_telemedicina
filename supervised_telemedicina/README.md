# supervised_telemedicina

Knowledge distillation di un **teacher rule-based** in un **MLP numpy**, con
**safety gate deterministico a precedenza assoluta**, persistenza SQLite e
notifiche per casi critici.

Il progetto sostituisce l'applicazione legacy (rule-based / Q-learning,
archiviata in `../archive/legacy_qtable_rule_based/` e **mai importata** dal
nuovo runtime) con un percorso supervised completo: dataset sintetico
etichettato dal teacher, training riproducibile, inferenza con fallback
sicuro e tracciamento di ogni analisi.

> **Stato: tutte le fasi 0-7 complete** (piano in 8 fasi, Fase 8 =
> documentazione). Suite: **127/127 test OK** (incluse le 4 fasi della
> dashboard di visualizzazione); regressione legacy: **14/14**.

## Vincolo di progetto

- **Solo numpy + standard library.** Vietato sklearn, joblib e qualunque
  dipendenza nuova; `sqlite3` (stdlib) è l'unico strumento di persistenza.
- **Fonte unica delle soglie cliniche**: `src/telemedicina_supervised/safety/safety_rules.py`.
  Nessuna classe, range o soglia è duplicata altrove (guard test
  `tests/test_single_source.py`).

## Architettura

```
supervised_telemedicina/
├── main.py                          # CLI: training esplicito e runtime
├── src/telemedicina_supervised/
│   ├── safety/safety_rules.py       # FONTE UNICA: range, soglie, pattern, classi
│   ├── agents/supervised_agent.py   # safety gate + MLP + fallback (AgentOutcome)
│   ├── ml/                          # MLP numpy, scaler, labels, gradient check
│   ├── services/                    # AnalysisService + notifiche
│   ├── database/analisi_db.py       # SQLite (tabelle analisi/notifiche)
│   ├── models/vital_parameters.py   # oggetto parametri vitali
│   └── config.py                    # path, seed, griglia, soglia di incertezza
├── training/                        # teacher, generatore dataset, training, metriche
├── tests/                           # 109 test (unittest)
└── data/                            # gitignored: dataset congelato, artifact, DB
```

Flusso applicativo unico: **CLI → `AnalysisService.analizza()` → `SupervisedAgent`
(safety gate → MLP) → SQLite + notifiche**. Il safety gate ha precedenza
assoluta: un caso critico non passa mai dall'MLP e non può essere declassato.

## Comandi

Training esplicito (sovrascrive artifact + report in `data/models/`):

```bash
python main.py --build-ml
```

Runtime con parametri JSON (opzioni mutuamente esclusive con `--build-ml`):

```bash
python main.py -ML --parametri '{"pressione_sistolica":120,"pressione_diastolica":80,"frequenza_cardiaca":75,"temperatura":36.8,"saturazione_ossigeno":98,"glicemia":95}'
```

Opzioni utili:

- `--modello-path PATH` — artifact alternativo (default `data/models/mlp_telemedicina.npz`).
- `--db-path PATH` — database SQLite (default `data/analisi.db`).
- `--seed N` / `--seed-base N` — seed del training (default 41).
- `--data-dir` / `--out-dir` — dataset e output del training.

Se l'artifact manca o non è valido, il runtime **fallback testuale** sul
rule-based con messaggio esplicito (`modello non trovato ...; eseguire
python main.py --build-ml`). I casi critici (`alto`) e gli input invalidi
generano una **notifica** (riga in tabella `notifiche` + stampa su stderr) e
vengono sempre persistiti su SQLite.

## Test

```bash
PYTHONPATH=src python -m unittest discover -s tests   # 109/109
```

Regressione della suite legacy (dal root del repo):

```bash
cd archive/legacy_qtable_rule_based && python -m tests.test_examples   # 14/14
```

## Dashboard di visualizzazione (completa, V1-V4)

Strumento didattico **read-only** che visualizza il sistema supervised in 6
viste, in un unico file HTML autonomo (CSS/JS inline, nessuna richiesta di
rete). Generato da `tools/genera_dashboard.py`:

```bash
python tools/genera_dashboard.py                          # build statico
python tools/genera_dashboard.py --serve                  # build + http://127.0.0.1:8000
python tools/genera_dashboard.py --out OUT --db-path DB   # output/database custom
python tools/genera_dashboard.py --metadati-path M --report-path R --test-path D
```

Opzioni: `--out` (default `data/dashboard.html`), `--db-path` (default
`data/analisi.db`), `--metadati-path` (default `data/processed/metadati.json`),
`--report-path` (default `data/models/report.json`), `--test-path` (directory
con `X_test.npy`/`y_test.npy`, default `data/processed`), `--serve` (aggiunge
l'endpoint `GET /api/analisi` per il pulsante "Aggiorna" della vista Live).

Prerequisito opzionale: **matplotlib** (solo per le figure, import lazy) —
`pip install -r requirements_dashboard.txt`. Senza matplotlib le figure
diventano segnaposto con messaggio, mai un crash; il runtime (`src/`,
`training/`, `main.py`) non importa mai matplotlib.

Le 6 viste (tutte implementate, nessun segnaposto):

1. **Analisi Live** — tabella delle analisi dal DB (read-only) con badge
   MLP / GATE / FALLBACK / NOTIFICA.
2. **Flusso decisionale** — percorso del caso (safety gate → MLP → soglia di
   incertezza → regola max), con i valori reali dai metadati.
3. **Teacher vs MLP** — distillation: banner metriche, matrice di confusione,
   scatter degli errori e regioni di decisione affiancate.
4. **Training** — curve di loss, confronto grid, gradient check, architettura.
5. **Incertezza** — istogramma della confidenza softmax (soglia da config),
   contatori di sistema e tabella dei mancati dell'MLP ricalcolata.
6. **Riproducibilità** — seed, split e hash di provenienza.

Tutte le metriche dichiarate sono lette da `data/models/report.json` (mai
ricalcolate né hardcoded); le soglie cliniche non sono mai duplicate (riuso
di `training.metrics` e `training.teacher_rules`).

## Riproducibilità

- Seed fisso **41** (configurabile), split congelati (train 40000 / val 10000 /
  test 20000, seed derivati 41/42/43).
- `data/processed/metadati.json` registra gli hash di provenienza di
  `safety_rules`, `synthetic_generator` e `teacher_rules`.
- Report deterministico (`data/models/report.json`): stesso seed → stessi
  numeri. Accuracy test congelato **0.9851**, slice no-buffer **0.9324**,
  kappa **0.9775**.

## Documentazione

- `documentazione.md` — documentazione tecnica (schema DB, pipeline, design decisions).
- `docs/contratto.md` — contratto di dominio (feature, classi, safety).
- `docs/review_finale.md` — review informale finale con limiti dichiarati.