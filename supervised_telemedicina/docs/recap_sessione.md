# Recap sessione di lavoro

## 1. Contesto del progetto

Progetto didattico di telemedicina che sostituisce l'applicazione legacy
(`archive/legacy_qtable_rule_based/`, rule-based + Q-learning) con un nuovo
progetto autonomo (`supervised_telemedicina/`).

**Obiettivo:** knowledge distillation — un classificatore supervised (MLP in puro
numpy) impara ad approssimare le label generate da un teacher rule-based offline
deterministico.

**Vincoli architetturali:**

- solo numpy + standard library (VIETATO sklearn/joblib);
- tutte le soglie cliniche vivono SOLO in `safety/safety_rules.py` (fonte unica,
  guard test `tests/test_single_source.py`);
- nessun commit automatico;
- gate oracle + scansione explorer dopo ogni fase;
- stato tracciato in `.slim/deepwork/supervised-telemedicina.md` (locale, ignorato da Git).

## 2. L'idea di implementazione

1. **Teacher offline** → genera le label del dataset sintetico (classi `basso`/`medio`/`alto`).
2. **MLP a 2 strati scritto a mano** in numpy: forward, softmax stabile, cross-entropy
   protetta, backward analitico, minibatch SGD, early stopping con checkpoint dei pesi,
   scaler incluso nell'artifact, gradient check.
3. **Safety gate NON opzionale**: l'MLP opera *dietro* le regole deterministiche —
   input invalidi → `errore` (mai una classe di rischio), casi critici → `alto`
   (mai declassabile).
4. **Zona di incertezza**: se `max(probabilità softmax) < 0.6` → fallback al rule-based
   (il sistema è conservativo dove il modello è debole, cioè sui boundary).
5. **Piano in 8 fasi** con gate di review dopo ogni fase e test incrementali.

## 3. Cosa abbiamo fatto (fase per fase)

### Fase 0 — Archivio legacy + bootstrap
- Legacy spostato con `git mv`: 14/14 test passanti dalla nuova posizione.
- Nessun import runtime da `archive/`.

### Fase 1 — Contratto dominio, safety rules, teacher
- `docs/contratto.md`, `safety/safety_rules.py` (fonte unica di FEATURE_ORDER, classi,
  range e soglie), `training/teacher_rules.py` (riusa `classifica_rischio`, nessuna
  soglia propria), `agents/supervised_agent.py` (baseline rule-based), `analysis_service.py`,
  guard test `test_single_source.py`.

### Fase 2 — Dataset sintetico congelato
- 12 profili correlati, buffer zone, seed doppi.
- Split congelati e riproducibili: train 40000 / val 10000 / test 20000, seed 41/42/43.
- `metadati.json` con 3 hash di provenienza (safety_rules, synthetic_generator, teacher_rules).
- Riproducibilità su disco verificata (sha256 identici).

### Fase 3 — Core MLP numpy
- `ml/mlp.py` (forward/backward, softmax stabile, early stopping, serializzazione con
  validazione forme), `ml/scaler.py`, `ml/labels.py`, `ml/gradient_check.py`.
- Gradient check verificato dell'ordine di 1e-10.

### Fase 4 — Training e valutazione
- Grid di 6 config (hidden 16/32/64, lr 0.01/0.05), selezione su `min(loss_val)`,
  valutazione UNA volta sul test congelato.
- Report riproducibile con seed-base 41 (eseguito più volte con risultati identici):
  - accuracy test congelato **0.9851**; slice no-buffer **0.9324**;
  - kappa **0.9775**; recall `alto` 0.9958 (30 mancati, 30 declassati);
  - teacher baseline 1.0 (upper bound); 298 errori su 20000;
  - distanza media dagli errori ~**0.524** (dopo la remediation della metrica).
- Correzioni applicate: distanza dalle soglie ora coerente con dati grezzi; selezione
  per loss minima invece dell'ultima epoca; docstring e indici derivati da `CLASSI`.

### Fase 5 — SupervisedAgent con MLP reale (COMPLETA)
- `SupervisedAgent` ora: valida input → safety gate rule-based → carica/valida artifact →
  scaler + MLP + softmax → soglia di incertezza (0.6) → regola `max(rule-based, MLP)`
  (nessun declassamento) → metadati completi.
- Fallback robusto per artifact assente, corrotto, versione errata, classi errate,
  scaler mancante, dimensioni errate, pesi/bias non finiti.
- `main.py` passa il path dell'artifact all'agente; `analizza_parametri()` come API di compatibilità.
- Suite: **102/102 test** passanti; guard test OK; nessun import sklearn/joblib.

### Fase 6 — CLI e configurazione (PARZIALE)
- `main.py`: comando di training esplicito `--build-ml`, runtime `-ML/--run-ml`,
  opzioni mutuamente esclusive, fallback testuale se il modello manca.
- `src/telemedicina_supervised/config.py`: path, seed, soglia di incertezza, grid.
- Restano da consolidare: configurazione unificata (grid duplicata tra `config.py` e
  `training/train_model.py`), percorso unico CLI/AnalysisService, validazione artifact
  più rigorosa, runtime che usa realmente l'MLP in ogni flusso.

## 4. Risultati e commit

- Suite: **102/102 test passanti**.
- Riproducibilità: report e artifact identici tra esecuzioni ripetute.
- Commit: `8a2e1fb Completata integrazione MLP supervised` (9 file, +634/−112).
- `git diff --check` pulito; legacy intatto; nessuna dipendenza esterna installata.

## 5. Cosa manca (prossimi passi)

1. **Chiusura Fase 6**: unificare la configurazione (GRID, seed, path), rendere la CLI
   e `AnalysisService` un unico percorso applicativo, validazione artifact più severa.
2. **Fase 7**: database persistente (`sqlite3`, path configurabile), notifiche per casi
   critici, test end-to-end (CLI → agente → servizio → DB), regressione suite legacy.
3. **Fase 8**: aggiornare `README.md` e `documentazione.md` (oggi fermi a "Fase 0"),
   aggiornare `knowledge/09_implicazioni_progetto.md`, creare il file di review informale
   finale con i limiti dichiarati (nessuna pretesa diagnostica).
4. **Decisione aperta**: una webapp visiva come il progetto legacy **non** è nel piano
   attuale (che prevede CLI); va deciso se aggiungerla come fase extra.
