# Review dello stato di avanzamento — supervised_telemedicina

**Data:** 2026-08-19
**Riferimento:** `docs/recap_sessione.md`, piano in `.slim/deepwork/supervised-telemedicina.md`
**Verdetto sintetico:** il lavoro fatto è **solido e allineato agli obiettivi**; si può
proseguire. Non ci sono problemi strutturali bloccanti, ma ci sono **4 punti concreti da
chiudere nella Fase 6 prima di passare alla Fase 7**, più alcune note minori.

---

## 1. Cosa ho verificato direttamente (non solo letto dal recap)

| Verifica | Esito |
|---|---|
| Suite test (`PYTHONPATH=src python -m unittest discover -s tests`) | **102/102 OK** — confermato |
| Report di training (`data/models/report.json`) | numeri del recap confermati: accuracy 0.9851, kappa 0.9775, recall `alto` 0.9958, 30 mancati, slice no-buffer 0.9324 |
| Runtime reale (`main.py -ML --parametri '{...}'`) | funziona end-to-end: artifact caricato, scaler+softmax, probabilità e metadati corretti |
| Assenza sklearn/joblib | grep pulito su `src/`, `training/`, `tests/`, `main.py` |
| Fonte unica delle soglie | nessuna soglia duplicata trovata; `teacher_rules.py` riusa `classifica_rischio` |
| Dataset congelato | `metadati.json` presente con 3 hash di provenienza e conteggi coerenti col report |
| Safety gate nel codice | `supervised_agent.py`: input invalidi → `errore`; `alto` rule-based → ritorno immediato senza MLP; confidenza < 0.6 → fallback; classe finale = `max(rule-based, MLP)` |
| Git | commit `8a2e1fb` presente, working tree pulito (solo `recap_sessione.md` non tracciato) |

## 2. Cosa va bene (allineato agli obiettivi)

1. **La proprietà di sicurezza chiave è vera nel codice, non solo sulla carta.** Il gate
   deterministico ha precedenza assoluta: un caso `alto` per le regole non passa mai
   dall'MLP, quindi il recall di sistema sui critici è 1.0 *per costruzione*,
   indipendentemente dal 0.9958 dell'MLP. La regola `max(rule-based, MLP)` può solo
   alzare la classe, mai abbassarla. Questo è esattamente il design dichiarato nel
   contratto (`docs/contratto.md` §1.4).
2. **Onestà metodologica.** Il report e il recap distinguono correttamente tra fedeltà
   al teacher (knowledge distillation) e accuratezza diagnostica; lo slice no-buffer
   (0.9324) è riportato accanto al numero ottimistico (0.9851). I 298 errori su 20000
   con distanza media ~0.52 dalle soglie confermano che gli errori stanno sui boundary,
   dove anche la zona di incertezza (fallback < 0.6) è progettata per intervenire.
3. **Riproducibilità reale.** Seed fissi, split congelati, hash di provenienza nei
   metadati, report deterministico. Per un progetto didattico questo è il criterio
   più importante ed è rispettato.
4. **Validazione difensiva dell'artifact.** L'agente rifiuta artifact assente, corrotto,
   con classi/dimensioni/scaler errate o pesi non finiti, con fallback esplicito e
   motivato. Il degrado è grazioso, mai silenzioso.
5. **Processo a gate rispettato.** Il file deepwork mostra gate oracle e remediation
   documentati per ogni fase, con note applicate e verifiche indipendenti.

## 3. Problemi da risolvere (in ordine di priorità)

### P1 — Chiudere la Fase 6 prima della Fase 7 (bloccante per il proseguimento ordinato)

Il gate Fase 4–5 è stato approvato a condizione che la Fase 6 sia validata **prima**
di database e notifiche. Ad oggi la Fase 6 è parziale, e i punti aperti sono reali,
non cosmetici:

1. **GRID duplicata e già divergente.** `config.TRAINING_HYPERPARAMETERS` (tupla) e
   `training/train_model.py::GRID` (lista) sono due copie separate: il training via
   `--build-ml` usa la GRID locale di `train_model.py`, **ignorando** quella in
   `config.py`. La "configurazione unificata" esiste sulla carta ma non è usata dal
   percorso principale. Rischio: qualcuno modifica `config.py` aspettandosi effetto
   sul training e non succede nulla. → Un'unica griglia in `config.py`, importata da
   `train_model.py`.
2. **Percorso unico CLI/AnalysisService.** Oggi `main.py` chiama direttamente
   `SupervisedAgent` + `analizza()` (doppia analisi dei parametri: una in CLI per il
   display, una dentro `predict`). In Fase 7 persistenza e notifiche vivranno nel
   service: se la CLI non passa dal service, i casi critici lanciati da CLI **non**
   genererebbero notifica né record sul DB. → La CLI deve diventare un thin wrapper
   su `AnalysisService.analizza()`. Questo è il punto strutturale più importante da
   sistemare ora, perché dopo sarà più costoso.
3. **Validazione artifact più severa**: già buona nell'agente, ma `carica_modello()` in
   `config.py` ha le classi hardcoded `("basso", "medio", "alto")` invece di importare
   `CLASSI` da `safety_rules` — duplicazione minore che sfugge al guard test
   (che protegge le soglie numeriche, non le tuple di classi).

### P2 — Documentazione pubblica falsa (non bloccante, ma urgente)

`README.md` e `documentazione.md` dichiarano ancora "**Fase 0** — scheletro non
funzionale, test da implementare", mentre il progetto ha un MLP funzionante e 102 test.
Chiunque apra il repo oggi (compreso un docente) legge informazioni sbagliate. Il recap
lo schedula in Fase 8, ma suggerisco di **anticipare almeno il README** subito dopo la
chiusura della Fase 6: bastano 20 righe (stato reale, comandi di training/runtime,
come lanciare i test).

### P3 — Note minori (registrare, non bloccare)

- **I test richiedono `PYTHONPATH=src`**: senza, un modulo (`test_agent_service`) fallisce
  l'import mentre gli altri passano, dando un falso senso di parzialità. Il risultato
  "102/102" dipende quindi dall'ambiente. Va documentato nel README o risolto con un
  `pyproject.toml` minimale.
- **`_esegui_runtime` analizza due volte** gli stessi parametri (una per le anomalie da
  mostrare, una dentro `predict`). Ininfluente sulle prestazioni a questo volume, ma
  scomparirà naturalmente se la CLI passa dal service (P1.2).
- **Limite da dichiarare esplicitamente nella review finale (Fase 8):** la regola
  `max(rule-based, MLP)` implica che il sistema integrato avrà **falsi positivi in più**
  rispetto al teacher (l'MLP può alzare `basso`→`medio`/`alto`). È il prezzo corretto
  della conservatività, ma va scritto nei limiti dichiarati.

## 4. Sulla decisione aperta (webapp)

Concordo con il recap: la webapp **non** è nel piano e non va aggiunta. Il piano in 8
fasi è già completo per gli obiettivi didattici (distillation, safety gate,
riproducibilità); una webapp aggiungerebbe superficie senza valore metodologico e
rimanderebbe Fase 7-8. Se proprio servirà una demo, basta la CLI che già funziona.

## 5. Ordine consigliato per proseguire

1. Unificare la griglia in `config.py` (eliminare `GRID` da `train_model.py`).
2. Rifare la CLI come wrapper di `AnalysisService` (un solo percorso applicativo).
3. Sistemare `carica_modello` per usare `CLASSI` dalla fonte unica.
4. Gate oracle di Fase 6 → poi Fase 7 (SQLite, notifiche, test end-to-end, regressione legacy).
5. Aggiornare README (subito dopo Fase 6) e completare la documentazione in Fase 8.

## Conclusione

Il progetto è in ottima salute: architettura coerente col contratto, proprietà di
sicurezza verificate nel codice, risultati riproducibili e riportati con onestà.
**Si può proseguire**, a condizione di chiudere i tre punti della Fase 6 (config unica,
percorso unico CLI/service, validazione artifact) prima di toccare database e notifiche:
sono esattamente i punti dove rimandare costerebbe doppio lavoro dopo.
