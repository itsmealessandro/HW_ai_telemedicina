# Piano di realizzazione: modello supervised MLP per la telemedicina

**Stato:** piano adattato alla riorganizzazione – perimetro archivio confermato  
**Data:** 2026-08-18  
**Obiettivo:** archiviare il progetto precedente e realizzare da zero un nuovo progetto autonomo
con classificatore supervised MLP implementato con `numpy`.

> **Nota di revisione (2026-08-18):** il piano è stato sottoposto a revisione architetturale
> indipendente. Verdetto: **fattibile**, con 7 correzioni obbligatorie integrate nelle sezioni
> seguenti (contrassegnate con ⚠️). Le correzioni riguardano: dichiarazione esplicita
> dell'obiettivo come distillazione del rule-based, fonte unica delle soglie, generatore a
> profili correlati, test set congelato, zona di incertezza, gradient checking e parità di
> sicurezza, disclaimer clinico.

> **Cambio di perimetro richiesto dall'utente:** questa non sarà un'estensione del progetto esistente.
> Il vecchio sistema verrà congelato in `archive/` e il nuovo sistema avrà codice, test, training,
> artifact e documentazione propri.

## 1. Riorganizzazione del repository

### 1.1 Struttura proposta

```text
HW_persia_privato/
├── archive/
│   └── legacy_qtable_rule_based/       # vecchio sistema, sola consultazione
├── supervised_telemedicina/            # nuovo progetto autonomo
│   ├── main.py
│   ├── README.md
│   ├── documentazione.md
│   ├── requirements.txt
│   ├── src/telemedicina_supervised/
│   │   ├── agents/
│   │   ├── database/
│   │   ├── ml/
│   │   ├── models/
│   │   ├── safety/
│   │   ├── services/
│   │   └── utils/
│   ├── training/
│   └── tests/
├── knowledge/                          # knowledge base del corso, condivisa
├── projectDescription/                 # task originale, condiviso
└── .plans/
```

### 1.2 Cosa archiviare e cosa lasciare condiviso

La proposta predefinita è archiviare tutti gli artefatti dell'applicazione precedente:

```text
main.py
src/
training/
tests/
README.md
documentazione.md
requirements.txt
mise.toml
```

in `archive/legacy_qtable_rule_based/`, aggiungendo un `README.md` che descriva la provenienza,
le modalità disponibili (rule-based e Q-learning) e il commit di riferimento.

`knowledge/`, `projectDescription/`, `.plans/`, `.git/` e `.gitignore` restano invece risorse
condivise del repository, non parti del vecchio runtime. Il perimetro è stato confermato: non verranno
spostati nell'archivio `knowledge/` e `projectDescription/`; saranno utilizzabili dal nuovo progetto.

### 1.3 Regole di isolamento

- Il nuovo progetto **non importerà moduli** da `archive/legacy_qtable_rule_based/`.
- Il vecchio progetto sarà verificato dopo lo spostamento, ma poi resterà immutato.
- Il nuovo progetto avrà il proprio `main.py`, package Python, `requirements.txt`, test, database,
  log, dataset e artifact.
- Le sole conoscenze trasferite saranno concetti documentati e requisiti del task, non codice runtime.
- Le regole per label sintetiche e sicurezza verranno reimplementate nel nuovo progetto e saranno
  documentate come componenti offline/runtime del nuovo sistema.

### 1.4 Fase di migrazione e gate

Prima di spostare file:

1. controllare `git status` e registrare il commit di riferimento;
2. eseguire la suite legacy dal percorso originale;
3. spostare con `git mv` gli artefatti applicativi nell'archivio;
4. verificare che il vecchio progetto continui a funzionare dalla nuova posizione;
5. creare lo scheletro `supervised_telemedicina/` senza copiare automaticamente il codice legacy;
6. verificare che il nuovo package non risolva import dal percorso archivio.

**Gate:** archivio leggibile e test legacy passanti; solo dopo si inizia il nuovo progetto.

## 2. Contesto e decisione progettuale

Il progetto precedente contiene (e sarà archiviato):

- `IntelligentAgent`: agente rule-based, attualmente modalità predefinita;
- `QLearningAgent`: agente Q-learning opzionale;
- `AnalysisService`: coordinatore agente → database SQLite → notifiche;
- `VitalParameters`: sei parametri vitali, validazione e soglie di gravità;
Il nuovo progetto partirà invece da un package vuoto e avrà come dipendenza iniziale solo `numpy`.

La nuova modalità supervised dovrà stimare il livello di rischio a partire da:

```text
X = [pressione_sistolica, pressione_diastolica,
     frequenza_cardiaca, temperatura,
     saturazione_ossigeno, glicemia]
```

Target multiclass:

```text
Y = {basso, medio, alto}
```

La scelta prevista è un **Multilayer Perceptron (MLP)** in `numpy` puro, con:

- input di dimensione 6;
- uno o due hidden layer con attivazione ReLU;
- output di dimensione 3 con softmax;
- loss cross-entropy;
- addestramento minibatch SGD;
- gradient descent e backpropagation implementati esplicitamente.

Questa scelta applica direttamente i concetti delle slide 9 e 10: supervised learning,
classificazione multiclass, train/validation/test, bias-variance, funzioni di attivazione,
cross-entropy, gradient descent e backpropagation.

## 3. Vincoli e decisioni da preservare

1. **Nessuna dipendenza obbligatoria nuova:** il primo percorso usa Python standard + `numpy`.
2. **Isolamento dal legacy:** il nuovo progetto non deve importare o eseguire moduli dall'archivio;
   il vecchio progetto viene verificato separatamente e poi congelato.
3. **Validazione prima del modello:** valori fisiologicamente impossibili non devono essere inviati
   all'MLP come se fossero dati normali.
4. **Safety-first:** un modello appreso non può declassare un caso critico già riconosciuto dalle
   regole deterministiche.
5. **Nessuna pretesa clinica:** il dataset iniziale sarà sintetico e le label deriveranno dalle regole
   esistenti; i risultati misurano principalmente la fedeltà al teacher, non l'accuratezza clinica reale.
6. **Riproducibilità:** seed, split, iperparametri, normalizzazione e versione del formato del modello
   devono essere registrati nell'artifact e nel report di training.
7. ⚠️ **Fonte unica delle soglie nel nuovo progetto:** le discrepanze del legacy (es. glicemia
    critica: teacher ≤60, override RL ≤55) saranno usate come elemento di audit, ma non verranno
    importate. Il nuovo progetto dovrà definire una singola fonte di verità in
    `supervised_telemedicina/src/telemedicina_supervised/safety/safety_rules.py`, usata da validatore,
    teacher offline e safety gate, per evitare label contraddittorie.
8. ⚠️ **Disclaimer obbligatorio:** la documentazione deve dichiarare che il sistema usa dati sintetici
    e non è destinato a uso clinico reale.

## 4. Problema logico principale: da dove provengono le label

Il nuovo progetto non dispone di un dataset clinico annotato. Si prevede quindi di creare un teacher
**offline e interno al nuovo progetto**, in `supervised_telemedicina/training/teacher_rules.py`, che
utilizzi esclusivamente la nuova fonte di verità `safety_rules.py`:

1. si generano combinazioni di parametri vitali plausibili;
2. il teacher rule-based assegna `basso`, `medio` o `alto`;
3. l'MLP apprende l'approssimazione di questa funzione;
4. si valuta l'MLP su combinazioni tenute fuori dal training.

Il vecchio `IntelligentAgent` archiviato non sarà importato dal generatore e non sarà una dipendenza
del nuovo sistema. Potrà essere usato soltanto per un audit storico separato, se necessario.

Questa è fattibile come dimostrazione didattica di supervised learning e knowledge distillation,
ma è una limitazione metodologica importante: se le label sono generate dalle regole, il modello
non sta scoprendo nuova conoscenza medica e un test sullo stesso teacher non dimostra validità clinica.
Il report dovrà dichiararlo esplicitamente.

⚠️ **Inquadramento corretto (revisione):** l'obiettivo è formalmente una **distillazione
teacher-student**: l'MLP impara a imitare il rule-based. Poiché il teacher è una funzione
deterministica delle 6 feature, ci si attende accuracy ~95-100% sul test: questo è *atteso e privo
di significato clinico*. Il confronto "MLP vs baseline rule-based" è **tautologico** se la baseline
è il teacher stesso → va rietichettato come *upper bound di imitazione*, non come gara clinica.

Per evitare un risultato tautologico o poco informativo:

- il dataset deve coprire anche le zone vicine alle soglie, non soltanto quattro profili fissi;
- il test deve contenere combinazioni mai usate nel training;
- devono essere inclusi test manuali sui pattern critici (shock, ipossia, crisi ipertensiva, ecc.);
- la metrica principale sarà la fedeltà al teacher, affiancata da metriche di sicurezza;
- ⚠️ riportare anche la **concordanza MLP-teacher** (es. Cohen's kappa) oltre ad accuracy/precision/recall;
- ⚠️ analizzare gli errori **per distanza dalle soglie**, non solo per classe (gli errori si
  concentreranno sui boundary: 139.9 vs 140.1 mmHg, SpO2 95, glicemia 140);
- un eventuale dataset clinico reale sarà una fase successiva e non un prerequisito per questa versione.

## 5. Fasi di lavoro

### Fase 0 – Archiviazione e bootstrap del nuovo progetto

**Scopo:** separare senza ambiguità il legacy dal nuovo sistema supervised.

Attività:

- registrare commit e stato del repository;
- eseguire la suite legacy prima dello spostamento;
- spostare con `git mv` gli artefatti applicativi in `archive/legacy_qtable_rule_based/`;
- aggiungere nell'archivio un README con contenuto, commit e comandi legacy;
- eseguire nuovamente i test dalla directory archivio;
- creare `supervised_telemedicina/` con package, `main.py`, `training/`, `tests/`, documentazione e
  configurazione proprie;
- creare una nuova struttura dati e un nuovo database per il sistema supervised;
- verificare che nessun import del nuovo package risolva percorsi sotto `archive/`.

**Gate:** il legacy è recuperabile e testato dalla nuova posizione; il nuovo progetto si avvia da
solo, anche se l'archivio non è presente nel `PYTHONPATH`.

### Fase 1 – Contratto funzionale e baseline del nuovo sistema

**Scopo:** fissare il comportamento prima di aggiungere apprendimento.

Attività:

- definire l'ordine ufficiale delle sei feature e delle tre classi;
- definire il trattamento dei valori invalidi: risposta `errore` e allerta, senza predizione MLP;
- definire la precedenza delle regole di sicurezza sui risultati del modello;
- definire una nuova baseline offline: `teacher_rules.py` + `safety_rules.py`;
- definire il contratto del nuovo `SupervisedAgent` e del nuovo servizio;
- verificare l'isolamento dall'archivio.

**Gate:** il contratto input/output è scritto e la nuova baseline non importa il legacy.

### Fase 2 – Dataset sintetico etichettato

**Scopo:** costruire un dataset riproducibile usando la conoscenza rule-based esistente.

Attività previste:

- creare `supervised_telemedicina/training/generate_dataset.py` e moduli separati dal training;
- ⚠️ **generatore a profili correlati** (mixture di profili come `SimPatientEnv._genera_parametri`):
  il campionamento indipendente dei 6 parametri viola il vincolo `sistolica > diastolica`
  (`vital_parameters.py:79`) e produce combinazioni fisiologicamente assurde; la diastolica va
  campionata condizionata alla sistolica;
- ⚠️ rispettare i range di `valida_parametri` (SpO2 70-100, glicemia 20-500): i campioni fuori range
  producono label `'errore'` e vanno gestiti/esclusi esplicitamente;
- campionare valori nei range fisiologici e in fasce mirate attorno alle soglie;
- includere esempi normali, anomalie singole, anomalie multiple e combinazioni dei pattern critici;
- ⚠️ **buffer zone attorno alle soglie**: escludere dal training i punti entro un margine dalle soglie
  (es. ±2 mmHg sistolica, ±1 bpm FC, ±1% SpO2) o etichettarli con la classe del lato "più sicuro",
  per ridurre il rumore di label sul gradino;
- scartare o separare gli input che falliscono `valida_parametri()`;
- assegnare la label chiamando `training/teacher_rules.py`, senza duplicare le soglie nel generatore;
- ⚠️ **stratificazione sulle label effettive** (post-label, non su quelle attese: il teacher può
  riclassificare diversamente); oversampling delle regioni anomale nel generatore; se serve, pesi di
  classe nella cross-entropy (SMOTE non disponibile senza sklearn e non necessario);
- usare un seed configurabile; ⚠️ settare **entrambi** i generatori (`random.seed()` e
  `np.random.seed()` o `default_rng(seed)`): il codice esistente usa il modulo `random` di Python;
- controllare il conteggio per classe e fallire con un errore chiaro se una classe è assente;
- ⚠️ **congelare il dataset su disco** (`data/processed/` con seed registrato) e usare split per
  blocchi di generazione: rigenerare a ogni run rende il tuning non confrontabile;
- salvare, se utile per la riproducibilità, un dataset in `data/` (runtime/generated e ignorato da Git)
  con metadati di seed e distribuzione.

**Gate:** dataset valido, classi presenti, distribuzione riportata e casi di confine verificati.

### Fase 3 – Core MLP in `numpy`

**Scopo:** implementare e testare il modello indipendentemente dal progetto telemedico.

Componenti:

- conversione delle label in indici stabili (`basso=0`, `medio=1`, `alto=2`);
- standardizzazione delle feature; media e deviazione standard calcolate **solo sul train** e
  ⚠️ **salvate nell'artifact** (se il SupervisedAgent ricalcolasse la normalizzazione al volo
  sarebbe leakage);
- forward pass per gli hidden layer ReLU;
- forward pass output con softmax numericamente stabile;
- loss cross-entropy con protezione da `log(0)`;
- backward pass con chain rule per calcolare i gradienti;
- ⚠️ **gradient checking**: verifica della backpropagation con differenze finite su una rete piccola
  (test automatico) — criterio oggettivo di correttezza;
- aggiornamento dei pesi con minibatch SGD;
- inizializzazione riproducibile dei pesi;
- early stopping o selezione del miglior modello sulla validation loss;
- salvataggio/caricamento di pesi, bias, normalizzatore, classi, architettura e versione del formato.

Possibile collocazione: `supervised_telemedicina/src/telemedicina_supervised/ml/`, separato dalla
logica dell'agente, così il core matematico rimane riusabile e testabile.

**Gate:** su un piccolo dataset toy noto, la loss diminuisce, le forme delle matrici sono corrette,
la softmax restituisce probabilità finite la cui somma è circa 1, il gradient checking passa e il
modello può essere ricaricato senza cambiare le predizioni.

### Fase 4 – Training, model selection e valutazione

**Scopo:** applicare correttamente il workflow supervised delle slide 9.

Workflow:

1. split stratificato train/validation/test, ad esempio 80/10/10;
2. ⚠️ **test set congelato**: generato con seed diverso e dimensionato bene (≥10-20k campioni),
   salvato su disco e **mai toccato durante il tuning** — le metriche finali si calcolano una sola
   volta dopo la scelta degli iperparametri;
3. calcolo dello scaler sul solo train;
4. confronto di pochi iperparametri predefiniti (hidden size, learning rate, batch size, epoche);
5. scelta sulla validation, mai sul test;
6. eventuale k-fold sul solo blocco train+validation per stimare la variabilità;
7. riaddestramento/configurazione finale senza usare il test per prendere decisioni;
8. valutazione finale una sola volta sul test congelato.

Il report dovrà contenere:

- accuracy;
- precision, recall e macro-F1 per le tre classi;
- matrice di confusione;
- ⚠️ **Cohen's kappa vs teacher** (concordanza MLP-teacher);
- ⚠️ **analisi degli errori per distanza dalle soglie** (non solo per classe);
- andamento train/validation loss;
- distribuzione delle classi;
- risultato del baseline rule-based sullo stesso insieme, presentato come **upper bound di
  imitazione** (il teacher sulle proprie label ottiene per costruzione ~100%);
- metriche specifiche di sicurezza, soprattutto mancato riconoscimento della classe `alto` e dei
  pattern critici;
- media e deviazione standard se viene usata cross-validation.

**Interpretazione corretta:** poiché il teacher produce le label, un'elevata accuracy indica buona
approssimazione delle regole sintetiche. Non va presentata come accuratezza diagnostica su pazienti reali.

**Gate:** nessun leakage, test usato solo alla fine, metriche finite e report riproducibile con lo stesso seed.

### Fase 5 – Integrazione dell'agente supervised

**Scopo:** esporre il modello con la stessa interfaccia degli agenti esistenti.

Prevedere `supervised_telemedicina/src/telemedicina_supervised/agents/supervised_agent.py` con
`analizza_parametri(parametri) -> dict`, compatibile con il nuovo `AnalysisService`.

Contratto di sicurezza previsto:

1. eseguire la validazione dei parametri;
2. eseguire il controllo rule-based dei casi critici e dei pattern pericolosi;
3. se input invalido o caso critico, restituire la decisione deterministica del rule-based;
4. altrimenti eseguire standardizzazione + MLP;
5. ⚠️ **zona di incertezza**: se `max(prob) < 0.6` (soglia in config), il SupervisedAgent **delega
   al rule-based** — il sistema è conservativo dove il modello è debole (i boundary), che è il
   comportamento desiderato in ambito medico;
6. restituire classe, probabilità e metadati del modello;
7. mantenere anomalie e raccomandazioni coerenti con il sistema esistente;
8. non permettere a una predizione `basso` di annullare una condizione di sicurezza `medio/alto` già
   riconosciuta dalle regole;
9. ⚠️ **parità di sicurezza**: per ogni caso di override, l'output del SupervisedAgent deve essere
   identico a quello del rule-based (test automatico);
10. ⚠️ **fallback**: modello assente → rule-based, mai eccezione (pattern già usato per la Q-table
    in `cli.py:337`).

Il comportamento ibrido deve essere esplicito nel report: il modello supervised fornisce la
classificazione appresa, mentre le regole restano il livello di sicurezza e spiegazione. Le due
decisioni non devono essere fuse in modo implicito o non tracciabile.

### Fase 6 – CLI e configurazione del nuovo progetto

Modifiche previste:

- configurazione per path del modello, seed e iperparametri;
- comando di training esplicito, ad esempio `python main.py --build-ml` o uno script dedicato;
- modalità runtime esplicita, ad esempio `python main.py -ML`;
- messaggio chiaro se il modello non esiste, con possibilità di fallback rule-based sicuro;
- il nuovo progetto espone solo le modalità deliberate per il supervised MLP;
- il legacy continua a essere avviato soltanto dai propri percorsi archiviati;
- eventuale gruppo di opzioni mutuamente esclusive per evitare combinazioni ambigue.

Il training non deve partire silenziosamente a ogni avvio: deve essere un'operazione esplicita e
osservabile, perché produce un artifact e un report.

### Fase 7 – Test e verifica end-to-end

Test unitari:

- forward, softmax, loss e backpropagation;
- ⚠️ **gradient checking** con differenze finite su rete piccola;
- aggiornamento dei pesi e diminuzione della loss su dataset toy;
- split/scaler senza contaminazione del test;
- generazione delle label e copertura delle classi;
- serializzazione e ricaricamento con predizioni identiche;
- output dell'agente: classi, probabilità, messaggi e metadati;
- safety override per input invalidi e scenari critici;
- ⚠️ **parità di sicurezza**: ogni caso di override produce output identico al rule-based;
- ⚠️ **boundary**: input a 139.9/89.9 vs 140.1/90.1 non producono flip di classe senza passare
  dalla zona di incertezza.

Test di integrazione:

- il nuovo `AnalysisService` salva nel nuovo database e invia le notifiche;
- un caso critico non viene declassato dall'MLP;
- il modello mancante produce fallback/errore documentato;
- il nuovo sistema funziona senza importare il legacy;
- la suite legacy continua a passare dalla directory archivio.

⚠️ I nuovi test possono usare un runner custom autonomo del nuovo progetto; non devono dipendere
dal runner o dai moduli del progetto archiviato.

Verifica finale prevista:

```bash
python -m tests.test_supervised
python training/train_mlp.py --seed 42
```

I comandi vanno eseguiti dalla directory `supervised_telemedicina/`. Il training deve usare una
directory/artifact temporanea durante i test, per non contaminare il database o il repository.

### Fase 8 – Documentazione dell'elaborato

Aggiornare:

- `supervised_telemedicina/README.md` con comandi, modalità disponibili e struttura;
- `supervised_telemedicina/documentazione.md` con motivazioni, teoria MLP/backprop/SGD, dataset, split, iperparametri, metriche,
  risultati, limiti e safety architecture;
- `knowledge/09_implicazioni_progetto.md` con la decisione finale supervised vs RL e il nuovo perimetro;
- fonti del codice e riferimenti alle slide 9 e 10.

La documentazione dovrà distinguere chiaramente:

- **teacher rule-based** = fonte delle label sintetiche;
- **MLP supervised** = modello che approssima le label;
- **safety rule layer** = protezione deterministica in produzione;
- **validazione didattica** ≠ validazione clinica.

## 6. Verifica di fattibilità e problemi logici

### Fattibilità tecnica

Il piano è tecnicamente fattibile con Python 3.10 e `numpy` già presente:

- matrici, attivazioni, loss e gradienti richiedono solo `numpy`;
- serializzazione con `numpy.savez`/`numpy.load` e metadati JSON non richiede `joblib`;
- sei feature e una rete piccola rendono il costo computazionale trascurabile;
- il contratto `analizza_parametri` può essere definito da zero nel nuovo progetto;
- database, notifiche, CLI e modello dati sono componenti piccoli e indipendenti da ricostruire;
- il codice legacy non è necessario al runtime del nuovo sistema.

### Problemi individuati e mitigazioni

| Problema | Perché è reale | Mitigazione obbligatoria |
|---|---|---|
| Label circolari | Il modello apprende le regole, non la medicina | Dichiarare knowledge distillation e limitare le conclusioni alla fedeltà al teacher |
| Mancanza di ground truth clinico | Le soglie nel codice non sostituiscono linee guida validate | Nessuna pretesa diagnostica; dataset reale + revisione esperta come fase futura |
| Classe `alto` sottorappresentata | Campionamento casuale può produrre poche emergenze | Campionamento stratificato e controllo conteggi prima del training |
| Confini delle soglie | Piccoli cambiamenti possono cambiare la label | Campionare sistematicamente intorno alle soglie e testare i boundary cases |
| Data leakage | Scaler o tuning sul test gonfiano le metriche | Fit dello scaler e scelta iperparametri solo su train/validation |
| Overfitting | Dataset sintetico semplice e rete troppo grande | Rete piccola, validation, early stopping, seed e confronto train/validation |
| Declassamento di emergenze | L'MLP può sbagliare una classe critica | Rule-based safety gate e test espliciti dei falsi negativi critici |
| Output incoerenti | Classe ML, anomalie e raccomandazioni potrebbero divergere | Contratto ibrido dichiarato, metadati e precedenza deterministica documentata |
| Artifact mancante/corrotto | Runtime non può predire senza pesi | Comando di training esplicito, verifica formato e fallback sicuro |
| Baseline tautologica | Il rule-based sulle label generate da sé ottiene per costruzione il 100% | Presentare il confronto come fidelity-to-teacher, non come gara clinica |
| Riproducibilità | Random sampling e inizializzazione cambiano i risultati | Seed unico registrato, artifact versionato e report dei parametri |
| ⚠️ Soglie duplicate e divergenti | Le soglie sono in 3 punti e già divergono (glicemia ≤60 vs ≤55) | Fonte unica delle soglie (modulo `safety_rules.py`) usata da teacher, override e validatore |
| ⚠️ Campionamento indipendente | Viola `sistolica > diastolica` e produce input assurdi | Generatore a profili correlati (mixture), diastolica condizionata alla sistolica |
| ⚠️ Rumore di label sui boundary | Il gradino delle soglie rende l'MLP instabile vicino ai confini | Buffer zone nel training + zona di incertezza in inference con delega al rule-based |
| ⚠️ Doppio generatore casuale | `rl_agent.py`/`rl_environment.py` usano `random`, non `np.random` | Settare entrambi i seed (`random.seed()` + `np.random.seed()` o `default_rng`) |
| ⚠️ Backpropagation non verificata | Errori di segno/forma nei gradienti sono silenziosi | Gradient checking con differenze finite (test automatico) |
| ⚠️ Normalizzazione ricalcolata in inference | Media/std ricalcolate al volo = leakage | Salvare media/std nell'artifact e riusarle in inference |

### Decisione go/no-go

**GO condizionato:** il piano è fattibile e logicamente coerente se l'obiettivo è dimostrare supervised
learning su un problema didattico e costruire un classificatore che apprende il comportamento del sistema
esperto.

**BLOCKER:** il piano non è sufficiente per sostenere che l'MLP sia diagnostico o clinicamente accurato.
Per quella conclusione servirebbero dati reali annotati, linee guida validate, gestione della popolazione,
validazione indipendente e supervisione clinica. Questi elementi sono fuori dallo scope attuale e devono
restare esplicitamente fuori dalle conclusioni dell'elaborato.

## 7. Criteri di accettazione del lavoro futuro

L'implementazione potrà considerarsi completa solo se:

1. il training è riproducibile con un comando e un seed;
2. train/validation/test sono separati correttamente;
3. il modello salva e ricarica identicamente;
4. il report contiene distribuzione classi, loss, confusion matrix, macro-F1 e metriche di sicurezza;
5. gli input invalidi e gli scenari critici non vengono declassati;
6. il nuovo database e le nuove notifiche continuano a funzionare;
7. il nuovo progetto funziona senza import dal legacy;
8. i test automatici passano;
9. la documentazione distingue fedeltà alle regole da validità clinica;
10. i limiti del dataset sintetico sono dichiarati;
11. ⚠️ zero dipendenze nuove (nessun import sklearn/torch nel codice ML);
12. ⚠️ gradient checking superato (backprop verificata con differenze finite);
13. ⚠️ stesso seed → stessa loss finale a parità di ambiente; dataset e artifact salvati con metadati;
14. ⚠️ boundary: input a 139.9/89.9 vs 140.1/90.1 non producono flip di classe senza passare dalla
    zona di incertezza;
15. ⚠️ parità di sicurezza: tutti i casi di override producono output identico al rule-based;
16. ⚠️ il nuovo SupervisedAgent passa i test del nuovo runner (normali→basso,
    crisi→alto, invalidi→errore, fallback senza modello);
17. ⚠️ report onesto: accuracy + macro-F1 + kappa vs teacher + analisi errori per distanza dalle
    soglie, con la dicitura "imitazione del rule-based" nel titolo.

## 8. Ordine delle dipendenze e gate di approvazione

```text
Fase 0 (archiviazione + bootstrap)
        ↓
Fase 1 (contratto/baseline nuova)
        ↓
Fase 2 (dataset + audit)
        ↓
Fase 3 (MLP toy tests)
        ↓
Fase 4 (training + valutazione)
        ↓
Fase 5 (SupervisedAgent + safety)
        ↓
Fase 6 (CLI/config)
        ↓
Fase 7 (test end-to-end)
        ↓
Fase 8 (documentazione finale)
```

Non iniziare l'implementazione della Fase 3 prima che la Fase 2 abbia dimostrato che il dataset contiene
tutte le classi e che il significato delle label è accettato. Non integrare il modello nella CLI prima
che il suo artifact e il safety contract siano verificati in isolamento. Non spostare file nel nuovo
progetto senza aver superato il gate della Fase 0.
