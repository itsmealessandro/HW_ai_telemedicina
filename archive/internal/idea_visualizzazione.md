# Idea — Visualizzazione grafica del sistema supervised (aspetti IA)

**Data:** 2026-08-19
**Stato:** idea / proposta — estensione opzionale fuori dal piano 0-8
**Decisione presa (2026-08-19):** **Opzione C** — matplotlib (dipendenza dev, solo
`tools/`) + HTML/JS autogenerato in stdlib. Piano operativo in
`.plans/visualizzazione_supervised.md`.
**Riferimento:** il progetto legacy aveva una dashboard Streamlit
(`dashboard.py`: Q-table storica a colori, pazienti dal DB, policy finale);
questa proposta è l'equivalente per il nuovo sistema supervised, con le
viste pensate per **mettere in risalto gli aspetti di intelligenza
artificiale** del progetto (distillation, MLP, safety gate, incertezza,
riproducibilità).

---

## 1. Obiettivo

Una dashboard **read-only** che consuma ciò che il sistema già produce
(report.json, analisi.db, metadati.json, artifact) e lo mostra in modo da
evidenziare cosa rende "intelligente" il sistema:

1. **Knowledge distillation**: l'MLP impara dal teacher rule-based.
2. **MLP numpy a 2 strati**: training, loss, early stopping, gradient check.
3. **Safety gate a precedenza assoluta**: i casi critici non passano mai
   dall'MLP e non vengono mai declassati.
4. **Zona di incertezza**: confidenza softmax < 0.6 → fallback rule-based.
5. **Regola max(rule-based, MLP)**: il modello può solo alzare la classe.
6. **Riproducibilità**: seed, split congelati, hash di provenienza.

La dashboard NON modifica il sistema: è un consumatore esterno, come nel
legacy (i dati arrivano da DB e report, mai dal runtime).

## 2. Tecnologia — due opzioni

### Opzione A: Streamlit (come il legacy) — *consigliata per parità con il precedente*

`dashboard.py` con `streamlit` + `pandas`, identico pattern al legacy
(`st.dataframe` stilizzato, colori per rischio, tab). Vantaggi: rapidità,
interattività, coerenza col progetto precedente.

**Vincolo rispettato dichiarando il confine**: streamlit/pandas NON entrano
nel runtime (che resta numpy+stdlib, VIETATO sklearn/joblib). La dashboard
vive in `requirements_dashboard.txt` separato e non viene mai importata dal
package `src/`; il guard test `test_single_source.py` continua a proteggere
le soglie, e un nuovo guard test può verificare che `src/` non importi
streamlit/pandas.

### Opzione B: HTML/JS autogenerato, zero dipendenze (più "puro")

Un singolo script `tools/genera_dashboard.py` (solo stdlib: `http.server`,
`json`, `sqlite3`) che genera un `dashboard.html` autonomo (JS inline,
nessun CDN) e lo serve su `localhost`. Vantaggi: coerente al 100% con il
vincolo "solo numpy + standard library" — la visualizzazione stessa è
un esempio di ingegneria senza dipendenze. Svantaggi: più codice, grafici
manuali (canvas/SVG), meno interattività.

**Raccomandazione: Opzione A** — "come ho fatto per il precedente progetto",
migliore resa visiva per un progetto didattico, e il confine delle
dipendenze è dichiarabile in modo pulito.

> **Scelta finale (2026-08-19): Opzione C** — matplotlib (grafici) + HTML
> autogenerato in stdlib. Motivi: vincolo runtime intatto (matplotlib non è
> una libreria ML, non tocca `src/`), qualità visiva senza scrivere canvas a
> mano, artifact HTML autonomo e robusto in demo (niente server/porte),
> un'unica dipendenza dev leggera (`pip install matplotlib`, niente
> pandas/streamlit/pyarrow). Vedi `.plans/visualizzazione_supervised.md`.

## 3. Le viste della dashboard (6 sezioni)

### 1. Live — Analisi recenti (dal DB `analisi.db`)

Come la sezione "Pazienti" del legacy, ma arricchita con i metadati della
Fase 5-7:

- colonne: timestamp, classe (colore: verde/arancio/rosso per
  basso/medio/alto, grigio per errore), allerta_medico, errori,
  probabilita (dict), modello_usato, motivo_fallback, override_sicurezza;
- badge distintivi per i casi che evidenziano l'IA:
  - **MLP** — quando il modello è stato usato;
  - **FALLBACK** — quando la confidenza era < 0.6 (zona di incertezza);
  - **GATE** — quando il safety gate ha bypassato l'MLP (caso critico);
  - **NOTIFICA** — casi con riga in tabella `notifiche`.

> Aspetto IA evidenziato: il sistema "pensa" in tre modi diversi
> (regole, MLP, fallback) e la dashboard rende visibile quale ha deciso
> per ogni caso.

### 2. Perché ha deciso così? (flusso decisionale di un caso)

Selezionando una riga, un diagramma verticale mostra il percorso del caso:

```
input → validazione (fisiologico?) → safety gate (critico?)
       → MLP (softmax) → confidenza ≥ 0.6? → max(rule, MLP) → DB + notifica
```

Ogni nodo mostra i valori reali del caso selezionato: classe della regola,
classe dell'MLP, probabilità softmax (barre per basso/medio/alto),
confidenza (con la soglia 0.6 disegnata), override_sicurezza.

> Aspetto IA evidenziato: trasparenza del classificatore ibrido
> neuro-symbolic — il perché di ogni decisione è sempre spiegabile.

### 3. Knowledge distillation — teacher vs MLP

Confronto diretto tra le regole (teacher) e l'approssimazione (MLP):

- **matrice di confusione** del test congelato (calda, con conteggi);
- **kappa 0.9775**, accuracy 0.9851 (test bufferizzato) e **0.9324**
  (slice no-buffer) affiancati, con nota sul perché il primo è ottimistico;
- **dove sbaglia l'MLP**: scatter 2D del test su due feature (es.
  sistolica × glicemia) con i 298 errori evidenziati e colorati per
  distanza dalle soglie (già calcolata in `analisi_errori_distanza` del
  report) — mostra che gli errori stanno sui boundary delle regole;
- **heatmap 2D delle regioni di decisione**: griglia fine di input su due
  feature (le altre a valori normali) colorata con la classe del teacher e
  con la classe dell'MLP, affiancate — la somiglianza visiva delle due
  mappe è la "prova" della distillation.

> Aspetto IA evidenziato: l'apprendimento per imitazione — l'MLP
> riproduce il comportamento del teacher, errori compresi sui boundary.

### 4. Training dell'MLP (la "macchina che impara")

- **curve loss train/val** della config migliore (dal report: 25 epoche,
  early stopping con pazienza 5) — mostra convergenza e dove si ferma;
- **confronto grid**: 6 piccole curve (hidden 16/32/64 × lr 0.01/0.05)
  con la config selezionata evidenziata (min loss_val);
- **gradient check**: badge con l'errore ~1e-10 (l'MLP è scritto a mano
  in numpy — il check dimostra che il backward è analiticamente corretto);
- architettura disegnata: input 6 → hidden (ReLU) → softmax 3 classi, con
  i pesi salvati nell'artifact.

> Aspetto IA evidenziato: il training non è una scatola nera — si vede
> la rete che impara, epoca per epoca.

### 5. Zona di incertezza e sicurezza

- **istogramma della confidenza softmax** sul test: quanti casi sotto 0.6
  (zona di incertezza → fallback) e quanti sopra;
- contatori di sistema: casi critici totali, `override_sicurezza` (MLP che
  ha alzato la classe), fallback per confidenza;
- **richiamo di sistema sui critici = 1.0 per costruzione**: testo esplicito
  che spiega perché (il gate non lascia mai passare un 'alto' dall'MLP) —
  la proprietà più importante del progetto, resa visibile;
- tabella dei **30 mancati dell'MLP** (record che l'MLP avrebbe
  declassato): ognuno con la distanza dalle soglie, per mostrare che il
  gate li intercetta.

> Aspetto IA evidenziato: il sistema è conservativo dove il modello è
> debole — incertezza misurata e gestita esplicitamente.

### 6. Riproducibilità (il "contratto scientifico")

- seed 41, split congelati (40000/10000/20000), hash di provenienza da
  `metadati.json` (safety_rules, synthetic_generator, teacher_rules);
- versione dell'artifact, timestamp del report, badge "deterministico".

> Aspetto IA evidenziato: i risultati non sono fortuiti — stesso seed →
> stessi numeri, e la dashboard lo dichiara.

## 4. Fonti dati (tutte già esistenti, nessun nuovo output)

| Vista | Fonte |
|---|---|
| 1. Live | `data/analisi.db` (tabelle `analisi`, `notifiche`) |
| 2. Flusso decisionale | record della riga selezionata (probabilita, metadati JSON) |
| 3. Distillation | `data/models/report.json` (matrice, metriche, `analisi_errori_distanza`) + rigenerazione scatter/heatmap con numpy puro dal dataset congelato |
| 4. Training | `report.json` (loss_train/val_finale, config migliore) |
| 5. Incertezza | `report.json` + calcolo confidenze sul test congelato (numpy) |
| 6. Riproducibilità | `data/processed/metadati.json` + `report.json` |

Il calcolo di scatter/heatmap/istogrammi avviene in numpy puro (nessuna
dipendenza ML nuova) e viene cachato come nel legacy (`@st.cache_data`).

## 5. Piano di implementazione (4 fasi, stesso stile gate)

1. **Fase V1 — Scheletro e Live**: `dashboard.py` con vista 1 (tabella
   stilizzata da `analisi.db`, badge MLP/FALLBACK/GATE/NOTIFICA) +
   `requirements_dashboard.txt` + guard test "nessun import di streamlit
   da src/". *Gate: vista 1 funzionante su DB reale.*
2. **Fase V2 — Decisione spiegata**: vista 2 (flusso del caso selezionato)
   + vista 6 (riproducibilità). *Gate: un caso alto e uno con fallback
   mostrano percorsi corretti.*
3. **Fase V3 — Distillation**: vista 3 (confusione, kappa, scatter errori,
   heatmap teacher-vs-MLP) + vista 4 (curve training). *Gate: heatmap
   calcolata da report e dataset, numeri coerenti col report.json.*
4. **Fase V4 — Sicurezza e rifiniture**: vista 5 (incertezza, richiamo di
   sistema, tabella dei 30 mancati) + intestazione con vincoli e limiti
   dichiarati (nessuna pretesa diagnostica, banner in cima come nel
   contratto). *Gate: tutti i contatori verificati sui dati reali.*

## 6. Vincoli rispettati e decisioni aperte

- **Runtime intoccato**: la dashboard è un consumatore read-only; le soglie
  cliniche restano solo in `safety_rules.py`; nessuna dipendenza nuova
  entra in `src/`.
- **Da decidere**: Opzione A (Streamlit, come legacy) vs Opzione B
  (HTML/JS zero-dip) — vedi §2; e se la dashboard sostituisce o affianca
  la CLI come canale dimostrativo.
- **Coerenza con la review finale**: la webapp era dichiarata "fuori piano"
  in `docs/review_finale.md`; questa proposta la fa rientrare come
  estensione opzionale post-Fase 8, senza rinnegare il piano: la CLI resta
  il canale applicativo, la dashboard è la vetrina didattica.