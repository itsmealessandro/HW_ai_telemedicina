# HW_persia_privato — Progetto di Intelligenza Artificiale: Telemedicina

Progetto d'esame del corso **Artificial Intelligence (DT0171)**, A.A. 2025/2026,
Università dell'Aquila — DISIM, Prof. **Fabio Persia**.

L'obiettivo (dal task in `projectDescription/`): progettare e implementare un
**agente intelligente per la telemedicina** che gestisca una serie di parametri
vitali, restituisca un messaggio al paziente, informi il medico in caso di valori
significativamente squilibrati e tenga traccia di tutte le interazioni in un
database.

Il repository contiene l'implementazione completa: un sistema **supervised** che
combina un *teacher* rule-based, un **MLP numpy scritto da zero**, un **safety
gate deterministico** e una **dashboard di visualizzazione** didattica — più gli
appunti di teoria del corso e un documento LaTeX che spiega gli aspetti di IA
partendo dalle basi.

---

## Struttura del repository

```
HW_persia_privato/
├── projectDescription/          # Task d'esame (PDF + testo)
├── supervised_telemedicina/     # IL PROGETTO: sistema supervised completo
│   ├── main.py                  #   CLI: training esplicito e runtime
│   ├── src/telemedicina_supervised/
│   │   ├── safety/              #   safety_rules.py = FONTE UNICA delle soglie
│   │   ├── agents/              #   SupervisedAgent: safety gate + MLP + fallback
│   │   ├── ml/                  #   MLP numpy, scaler, labels, gradient check
│   │   ├── services/            #   AnalysisService + notifiche
│   │   ├── database/            #   SQLite (tabelle analisi/notifiche)
│   │   └── config.py            #   seed, soglia di incertezza, griglia
│   ├── training/                #   teacher, generatore dataset, training, metriche
│   ├── tests/                   #   129 test (unittest)
│   ├── tools/                   #   dashboard (genera_dashboard.py, _figure.py, launcher)
│   ├── docs/                    #   contratto, review finale
│   └── data/                    #   gitignored: dataset congelato, artifact, DB
├── archive/
│   └── legacy_qtable_rule_based/ # App legacy (rule-based/Q-learning) — archiviata,
│                                 # MAI importata dal nuovo runtime
├── knowledge/                   # Appunti di teoria del corso (markdown 01-09)
├── docs/
│   └── latex/                   # Documento LaTeX didattico sugli aspetti di IA
├── .plans/                      # Piani di sviluppo (supervised + visualizzazione)
├── progetto.sh                  # Script principale: test/training/analisi/dashboard/serve
└── README.md                    # Questo file
```

---

## Di cosa tratta il progetto

Il sistema classifica **6 parametri vitali** in **3 classi di rischio**
(`basso`, `medio`, `alto`):

| Feature | Unità |
|---------|-------|
| pressione_sistolica | mmHg |
| pressione_diastolica | mmHg |
| frequenza_cardiaca | bpm |
| temperatura | °C |
| saturazione_ossigeno | % |
| glicemia | mg/dL |

Per chi non conosce l'IA, il punto chiave è questo: **il sistema non "indovina"
a caso**. Parte da una conoscenza clinica esplicita (regole scritte a mano) e
la trasforma in un modello che ha imparato a riconoscere i pattern. Il percorso
è interamente **supervised** (supervisionato): qualcuno mostra al modello
esempi già etichettati e il modello impara da quelli.

Ecco i passaggi, dal punto di partenza al verdetto finale:

### 1. Il teacher rule-based — la "conoscenza di partenza"

Un classificatore a regole cliniche: range e soglie scritte a mano da un
esperto (in `safety_rules.py`, fonte unica mai duplicata). Per ogni
combinazione di valori vitali dice subito se il rischio è `basso`, `medio` o
`alto`. È trasparente al 100% (ogni regola si può leggere e capire), ma è
rigido: va scritto a mano e non generalizza a casi mai visti.

### 2. Il dataset sintetico — gli "esempi" su cui imparare

Per insegnare al modello servono tanti esempi. Il teacher genera un **dataset
sintetico** di 70 000 pazienti fittizi ma realistici (train 40 000 / val
10 000 / test 20 000, seed 41, split congelati) e li etichetta con le sue
regole. Ogni esempio è una coppia: *valori vitali → classe di rischio*.

### 3. L'MLP — la "rete neurale" che impara

L'**MLP (Multi-Layer Perceptron)** è una rete neurale scritta da zero in numpy:
una funzione matematica con centinaia di "manopole" regolabili (i pesi) che,
mostrandole gli esempi del dataset, impara a riconoscere i pattern che portano
a una classe di rischio. Architettura **6→n→3**: 6 ingressi (i parametri
vitali), uno strato nascosto, 3 uscite (una per classe). L'uscita softmax
esprime la **confidenza**: quanto il modello è sicuro di ogni classe.

Il training (backpropagation, cross-entropy, early stopping, ricerca a griglia
su 6 configurazioni) regola le manopole finché le previsioni non coincidono
con le etichette del teacher. Risultato sul test congelato: **accuracy
0.9851**, kappa 0.9775, recall classe alto 0.9958.

### 4. La knowledge distillation — "le regole insegnano alla rete"

Il MLP non inventa nulla: impara dalle etichette del teacher. È la
**knowledge distillation**: un sistema a regole trasparente "dista" la propria
conoscenza in una rete veloce e robusta, che ha imparato la stessa logica ma
può generalizzare a casi mai visti (confronto delle regioni di decisione nella
dashboard).

### 5. Il safety gate — la "rete di sicurezza"

Una rete neurale può sbagliare. Per questo a runtime il **safety gate
deterministico** ha **precedenza assoluta**: se le regole cliniche dicono che
il caso è critico (`alto`), il caso **non passa mai dall'MLP** e non può essere
declassato (richiamo 1.0 per costruzione). La macchina impara, ma l'ultima
parola sui casi gravi resta alle regole.

### 6. Incertezza e fallback — "quando il modello non è sicuro"

Se la confidenza softmax è sotto la soglia di incertezza (0.6), il sistema non
si fida del modello e fa **fallback** sulle regole testuali. Ogni analisi è
persistita su SQLite con metadati (modello usato, confidenza, fallback) e i
casi critici generano **notifiche** per il medico.

### Il flusso completo, in una riga

```
parametri vitali
   → safety gate: caso critico? ── sì ──→ classe "alto" (senza passare dall'MLP)
   │
   └─ no → MLP: previsione + confidenza
              → confidenza < 0.6? ── sì ──→ fallback sulle regole testuali
              │
              └─ no → classe del MLP
   → persistenza su SQLite + notifica se critico
```

Il tutto rispetta un vincolo di progetto rigoroso: **solo numpy + standard
library** nel runtime (vietati sklearn, joblib, pandas); matplotlib è ammesso
solo come dipendenza di sviluppo per la dashboard.

---

## Come si usa

### Script principale (`progetto.sh`)

Tutte le operazioni comuni sono gestite da un unico script alla root del
repository (funziona da qualsiasi directory corrente; richiede solo bash e
python3):

```bash
./progetto.sh test              # suite principale (129 test) + regressione legacy (14)
./progetto.sh train             # training: rigenera artifact MLP + report.json
./progetto.sh analisi '<json>'  # analizza un paziente via CLI (persiste su SQLite)
./progetto.sh dashboard         # genera la dashboard statica (data/dashboard.html)
./progetto.sh serve [porta]     # dashboard + API su http://127.0.0.1:<porta> (default 8000)
./progetto.sh help              # aiuto completo
```

Esempio di analisi runtime:

```bash
./progetto.sh analisi '{"pressione_sistolica":120,"pressione_diastolica":80,"frequenza_cardiaca":75,"temperatura":36.8,"saturazione_ossigeno":98,"glicemia":95}'
```

Nota: la tab **"7. Analisi interattiva"** della dashboard richiede
`./progetto.sh serve` — l'endpoint `POST /api/valuta` esiste solo con il
server attivo. I comandi manuali qui sotto restano validi e mostrano cosa lo
script fa sotto il cofano.

### Test

```bash
cd supervised_telemedicina
PYTHONPATH=src python -m unittest discover -s tests    # 129/129
cd ../archive/legacy_qtable_rule_based && python -m tests.test_examples   # 14/14
```

### Training (rigenera artifact + report)

```bash
cd supervised_telemedicina
python main.py --build-ml
```

### Runtime (analisi di un paziente)

```bash
python main.py -ML --parametri '{"pressione_sistolica":120,"pressione_diastolica":80,"frequenza_cardiaca":75,"temperatura":36.8,"saturazione_ossigeno":98,"glicemia":95}'
```

### Dashboard di visualizzazione (7 viste didattiche)

```bash
cd supervised_telemedicina
tools/avvia_dashboard.sh          # build + server + browser (Ctrl+C per chiudere)
# oppure, manualmente:
python tools/genera_dashboard.py  # genera data/dashboard.html
python tools/genera_dashboard.py --serve   # + endpoint /api/analisi
```

La dashboard mostra: analisi live con badge, flusso decisionale interattivo,
teacher vs MLP (heatmap, errori), training (curve loss), incertezza
(istogramma confidenze, mancati), riproducibilità (seed e hash) e una
**pagina interattiva** in cui l'utente inserisce un campione e vede
passo-passo come viene valutato (validazione → safety gate → rete → soglia →
risposta; richiede `--serve`).

### Documento LaTeX sugli aspetti di IA

```bash
cd docs/src
make          # compila e scrive documento.pdf in docs/
make open     # compila e apre docs/documento.pdf
```

Il documento è in due parti. La **Parte I** è una spiegazione semplice per chi
non conosce l'IA: con esempi di tutti i giorni spiega cosa fa il sistema e di
cosa si occupa ciascun componente (teacher, dataset, rete neurale,
distillazione, safety gate). La **Parte II** è l'analisi tecnica vera e
propria per chi sa programmare: machine learning supervisionato, dati, reti
neurali, backpropagation, valutazione, knowledge distillation, sicurezza e
incertezza, riproducibilità. Le figure sono generate dai dati reali del
progetto (`genera_figure.py`).

---

## Documentazione

| Dove | Cosa |
|------|------|
| `supervised_telemedicina/README.md` | README tecnico del progetto (comandi, architettura, dashboard) |
| `supervised_telemedicina/documentazione.md` | Documentazione tecnica (schema DB, pipeline, design decisions) |
| `supervised_telemedicina/docs/contratto.md` | Contratto di dominio (feature, classi, safety) |
| `supervised_telemedicina/docs/review_finale.md` | Review informale finale con limiti dichiarati |
| `knowledge/README.md` | Indice degli appunti di teoria del corso |
| `docs/documento.pdf` | Documento didattico sugli aspetti di IA del progetto (sorgenti in `docs/src/`) |
| `.plans/` | Piani di sviluppo (fasi 0-8 + dashboard V1-V4) |

---

## Stato

- **Fasi 0-8 del progetto**: complete (piano in `supervised_mlp_telemedicina.md`).
- **Dashboard di visualizzazione (V1-V4)**: completa e integrata su `main`.
- **Pagina interattiva + script principale**: tab 7 con valutazione
  passo-passo del campione; `progetto.sh` come punto d'ingresso unico.
- **Suite**: 129/129 test OK; regressione legacy 14/14.
- **Documento LaTeX**: compilato (45 pagine, 11 capitoli).