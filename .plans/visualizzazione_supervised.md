# Piano di realizzazione: dashboard di visualizzazione del sistema supervised (Opzione C)

**Stato:** in attesa di approvazione
**Data:** 2026-08-19
**Obiettivo:** vetrina grafica del sistema supervised che metta in risalto gli **aspetti di
intelligenza artificiale** del progetto (knowledge distillation, MLP numpy, safety gate,
zona di incertezza, riproducibilità), consumando in sola lettura ciò che il sistema già
produce (report, DB, metadati, dataset congelato, artifact).

> **Scelta tecnologica approvata dall'utente (2026-08-19): Opzione C** — grafici con
> `matplotlib` (dipendenza dev, NON una libreria ML) + generazione HTML/JS in stdlib
> (`http.server`, `json`, `sqlite3`). Motivazione: il vincolo "solo numpy + stdlib" del
> runtime resta intatto nella lettera e nello spirito (matplotlib non è sklearn/joblib),
> qualità visiva professionale senza scrivere canvas a mano, artifact HTML autonomo e
> robusto in demo. Alternative scartate: A (Streamlit+pandas: decine di dipendenze, indebolisce
> la narrazione del vincolo), B (HTML/JS puro: costo alto e resa povera).
> Dettaglio in `supervised_telemedicina/docs/idea_visualizzazione.md`.

---

## 1. Contesto

Il progetto legacy aveva una dashboard Streamlit (Q-table a colori, pazienti, policy finale).
Il nuovo sistema supervised (Fasi 0-8 complete, suite 109/109, legacy 14/14) ha un percorso
applicativo unico **CLI → AnalysisService → SupervisedAgent → SQLite + notifiche** e produce
già tutti i dati necessari alla visualizzazione:

| Fonte | Contenuto | Usata per |
|---|---|---|
| `data/analisi.db` | tabelle `analisi` e `notifiche` (Fase 7) | Vista Live, flusso decisionale |
| `data/models/report.json` | metriche, loss, matrice, analisi errori per distanza | Viste distillation/training/incertezza |
| `data/processed/metadati.json` | seed, split, hash di provenienza | Vista riproducibilità |
| `data/processed/X_test.npy` + `y_test.npy` | test congelato (20000) | scatter errori, istogramma confidenze, heatmap |
| `data/models/mlp_telemedicina.npz` | artifact MLP (pesi, scaler, classi) | heatmap MLP, confidenze |
| `training/teacher_rules.py` + `safety_rules.py` | teacher offline (fonte unica) | heatmap teacher |

## 2. Vincoli da preservare (inviolabili)

1. **Runtime intoccato**: nessuna dipendenza nuova entra in `src/`, `training/`, `main.py`.
   `matplotlib` esiste SOLO in `tools/` e in `requirements_dashboard.txt` (mai in
   `requirements.txt`).
2. **Fonte unica delle soglie**: la dashboard non definisce né mostra soglie proprie; per le
   heatmap usa `safety_rules`/`teacher_rules` (già fonti uniche) e per i numeri il
   `report.json`. Il guard test `tests/test_single_source.py` resta invariato e verde.
3. **Sola lettura**: la dashboard non scrive nel DB né rigenera dataset/artifact. Al massimo
   genera il proprio HTML in `data/` (già gitignored).
4. **Nessuna pretesa diagnostica**: banner didattico in cima alla dashboard, coerente con
   `docs/review_finale.md`.
5. **Test verdi senza matplotlib**: la suite usa `unittest.skipUnless` per i test che
   richiedono matplotlib, così resta verde anche senza la dipendenza dev.
6. **Isolamento dal legacy**: nessun import da `archive/`.

## 3. Struttura proposta

```text
supervised_telemedicina/
├── tools/                              # (nuovo) strumenti dev fuori dal runtime
│   └── genera_dashboard.py             # genera data/dashboard.html; --serve per live
├── requirements_dashboard.txt          # (nuovo) matplotlib (SOLO per la dashboard)
├── tests/
│   └── test_dashboard.py               # (nuovo) build reale + guard "no matplotlib in runtime"
└── data/
    └── dashboard.html                  # (generato, gitignored) artifact autonomo
```

### `tools/genera_dashboard.py` — contratto CLI

```text
python tools/genera_dashboard.py              # build statico -> data/dashboard.html
python tools/genera_dashboard.py --serve      # build + http.server su localhost:PORT
                                              # (aggiunge /api/analisi per il refresh live dal DB)
python tools/genera_dashboard.py --out PATH   # percorso output custom (default data/dashboard.html)
```

- backend matplotlib `Agg` (nessun display), figure salvate in buffer → **base64 PNG
  incorporato** nell'HTML: un solo file autonomo, apribile offline, nessun CDN.
- HTML: template con CSS/JS inline (vanilla JS): tab delle sezioni, filtro per classe,
  selezione di un caso → flusso decisionale.
- Degrado grazioso: se artifact o dataset mancano, la sezione mostra un messaggio esplicito
  (stesso pattern del fallback runtime), mai un crash.

## 4. Fasi di lavoro (con gate)

### Fase V1 — Scheletro e Vista Live

**Scopo:** la dashboard esiste e mostra le analisi reali dal DB.

Attività:
- creare `tools/` e `requirements_dashboard.txt` (`matplotlib>=3.7`), `tools/genera_dashboard.py`
  con: caricamento dati, shell HTML (header + banner didattico + CSS), layout a tab;
- **Vista 1 — Live**: tabella da `analisi.db` (timestamp, classe colorata
  basso/medio/alto/errore, allerta_medico, errori, probabilita, modello_usato,
  motivo_fallback, override_sicurezza) con badge **MLP / FALLBACK / GATE / NOTIFICA**
  (notifiche dalla tabella `notifiche` per id/timestamp);
- guard test in `test_dashboard.py`: `src/`, `training/` e `main.py` non importano
  matplotlib (grep); il resto della suite resta verde senza matplotlib (`skipUnless`).

**Gate:** `python tools/genera_dashboard.py` genera l'HTML; apertura offline; badge corretti
su un caso `alto` e uno con fallback (usando i dati reali esistenti).

### Fase V2 — Flusso decisionale e Riproducibilità

**Scopo:** la dashboard spiega il *perché* di ogni decisione e dichiara il "contratto
scientifico" del progetto.

Attività:
- **Vista 2 — Perché ha deciso così?**: selezione di una riga → diagramma verticale
  `input → validazione → safety gate → MLP/softmax → soglia 0.6 → max(rule, MLP) →
  DB+notifica`, con i valori reali del caso (classe regola, classe MLP, barre delle 3
  probabilità, confidenza con la soglia 0.6 disegnata, override);
- **Vista 6 — Riproducibilità**: seed 41, split 40000/10000/20000, hash da
  `metadati.json` (safety_rules, synthetic_generator, teacher_rules), versione artifact,
  badge "report deterministico".

**Gate:** per un caso critico e uno con fallback i percorsi mostrati sono corretti; gli hash
mostrati coincidono con `metadati.json`.

### Fase V3 — Knowledge distillation e Training

**Scopo:** le due viste che mostrano l'MLP "che impara ad imitare".

Attività:
- **Vista 3 — Distillation (teacher vs MLP)**:
  - matrice di confusione del test congelato (heatmap, da `report.json`);
  - banner: accuracy 0.9851 (bufferizzato) vs 0.9324 (no-buffer) + kappa 0.9775, con nota
    sull'ottimismo del primo;
  - scatter 2D del test (sistolica × glicemia) con i 298 errori evidenziati, colorati per
    distanza dalle soglie (da `analisi_errori_distanza`);
  - **heatmap delle regioni di decisione affiancate**: griglia fine su 2 feature (le altre a
    valori normali), colorata con la classe del teacher (via `teacher_rules`) e con la
    classe dell'MLP (via artifact) — la prova visiva della distillation. Calcolo in numpy
    puro, deterministico (dati congelati + artifact).
- **Vista 4 — Training**:
  - curve loss train/val della config migliore (da report, 25 epoche, early stopping);
  - confronto grid (6 config, best evidenziata con min loss_val);
  - badge gradient check ~1e-10 + diagramma architettura (input 6 → hidden ReLU → softmax 3).

**Gate:** i numeri delle viste coincidono esattamente con `report.json`; heatmap teacher e
MLP visivamente simili sui dati reali; se artifact manca la vista 3 degrada con messaggio.

### Fase V4 — Incertezza/Sicurezza e rifiniture finali

**Scopo:** la proprietà di sicurezza più importante diventa visibile; chiusura e docs.

Attività:
- **Vista 5 — Zona di incertezza e sicurezza**:
  - istogramma della confidenza softmax sul test congelato, soglia 0.6 evidenziata;
  - contatori: fallback per confidenza, `override_sicurezza`, notifiche totali;
  - testo esplicito "richiamo di sistema sui critici = 1.0 per costruzione" (il gate non
    lascia mai passare un 'alto' dall'MLP);
  - tabella dei **30 mancati dell'MLP** (dal report: classe regola alto, MLP no) con
    distanza dalle soglie — ognuno intercettato dal gate;
- header rifinito: vincoli di progetto (numpy+stdlib nel runtime, matplotlib solo per i
  grafici) e banner limiti (dati sintetici, nessuna pretesa diagnostica);
- aggiornare `README.md` (sezione Dashboard: comandi, prerequisito matplotlib) e
  `documentazione.md` (sezione visualizzazione);
- verifica finale completa (vedi §6).

**Gate:** tutti i contatori verificati sui dati reali; suite completa verde; documentazione
allineata.

## 5. Rischi e mitigazioni

| Rischio | Perché è reale | Mitigazione |
|---|---|---|
| matplotlib assente nell'ambiente | dipendenza dev nuova | `requirements_dashboard.txt` + istruzioni README; test in `skipUnless` → suite verde comunque |
| Rottura del vincolo "no dipendenze" percepita | matplotlib non è stdlib | Confine netto dichiarato: solo `tools/`, mai nel runtime; guard test dedicato; doc esplicita |
| Heatmap/scatter lenti o pesanti | 20000 punti + griglia 2D | numpy vettorizzato, risoluzione griglia limitata (~200×200), figure moderate; nessun loop Python |
| HTML troppo grande | figure base64 | PNG compressi (dpi ~100, formato compatto); target < 2-3 MB |
| Numeri incoerenti col report | doppia fonte (report vs ricalcolo) | I numeri mostrati vengono LETTI da `report.json`; il ricalcolo numpy serve solo per scatter/heatmap/confidenze (non per le metriche dichiarate) |
| Artefact/dataset assenti | ambiente pulito | Degrado grazioso per sezione (messaggio + suggerimento comando), mai crash |
| Guard test interferito | LITERAL_NOTI protegge i literal | Nessun literal clinico nuovo nel codice della dashboard; guard invariato |

## 6. Criteri di accettazione

1. `python tools/genera_dashboard.py` genera `data/dashboard.html` senza errori.
2. L'HTML è autonomo (immagini incorporate) e apribile offline con qualunque browser.
3. Le 6 viste sono presenti; i numeri chiave coincidono con `report.json`: accuracy
   0.9851/0.9324, kappa 0.9775, recall alto 0.9958, 298 errori, 30 mancati, distanza media
   0.524.
4. Vista Live corretta su DB reale: badge GATE sul caso `alto`, badge MLP sui casi col
   modello, notifiche coerenti.
5. **Nessuna dipendenza nuova nel runtime**: grep matplotlib su `src/`, `training/`,
   `main.py` = 0 match; `requirements.txt` invariato.
6. Suite completa verde: 109/109 + nuovi test dashboard (o skippati senza matplotlib).
7. Guard test `test_single_source.py` invariato e verde.
8. `git diff --check` pulito; nessun commit automatico; nessuna modifica a
   `archive/legacy_qtable_rule_based/`.
9. Documentazione aggiornata (README + documentazione.md) con comandi reali e prerequisito
   matplotlib dichiarato.

## 7. Ordine delle dipendenze e gate

```text
Fase V1 (scheletro + Vista Live)          -> gate: HTML generato e badge corretti
        ↓
Fase V2 (flusso decisionale + riprod.)    -> gate: percorsi corretti, hash coerenti
        ↓
Fase V3 (distillation + training)         -> gate: numeri == report.json, heatmap ok
        ↓
Fase V4 (incertezza/sicurezza + docs)     -> gate: verifica finale completa
```

Non iniziare la V3 prima che la V1 abbia dimostrato che l'HTML si genera dai dati reali.
Non aggiornare la documentazione (V4) prima che le viste siano tutte funzionanti.