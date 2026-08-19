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
> documentazione). Suite: **109/109 test OK**; regressione legacy: **14/14**.

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