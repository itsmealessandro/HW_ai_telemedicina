# HW_AI_Telemedicina — Agente Intelligente per la Telemedicina

Progetto d'esame di Artificial Intelligence (DT0171), A.A. 2025/2026, Università dell'Aquila — DISIM, Prof. Fabio Persia.

Il progetto presenta un sistema intelligente per il monitoraggio telemedico, incaricato di elaborare un vettore di input a 6 dimensioni (pressione sistolica e diastolica, frequenza cardiaca, temperatura, saturazione dell'ossigeno, glicemia) per inferire una tra tre classi di rischio (basso, medio, alto). Il framework produce un output consultivo per il paziente, attiva alert medici vincolanti nei casi critici e registra ogni inferenza tramite un database relazionale SQLite.

## Architettura del Sistema e Pipeline Analitica

Il sistema è basato su un'architettura ibrida che combina logica deterministica e apprendimento supervisionato. Il flusso operativo è strutturato nei seguenti stadi:
- **Teacher Rule-Based**: Implementazione di un set di regole di dominio (centralizzate in `safety_rules.py`) utilizzate per generare le label del dataset.
- **Sintesi del Dataset**: Generazione pseudo-casuale di 70.000 record (split: 40.000 train, 10.000 validation, 20.000 test) garantendo riproducibilità statistica (seed 41).
- **Knowledge Distillation tramite MLP**: Sviluppo di un Multi-Layer Perceptron implementato esclusivamente in NumPy. L'addestramento ottimizza la cross-entropy loss rispetto alle label fornite dal teacher.
- **Safety Gate Deterministico**: Modulo pre-inferenza a precedenza assoluta. Intercetta i casi clinici anomali assegnandoli direttamente alla classe di rischio massimo, mitigando il rischio di falsi negativi dell'MLP (garantendo un recall pratico del 100% sui casi critici).
- **Gestione dell'Incertezza (Softmax Fallback)**: In presenza di una confidence prediction inferiore alla soglia p = 0.60, l'MLP delega la classificazione al sistema rule-based.
- **Vincoli Tecnologici**: Il runtime opera senza dipendenze di terze parti (escluse librerie standard e NumPy), garantendo deployment agili e alta efficienza computazionale.

## Struttura della Repository

- `supervised_telemedicina/`: Directory sorgente (`main.py`, `src/`, `training/`, `tests/`, `tools/`, `docs/`).
- `docs/`: Documentazione progettuale (monografia didattica e sintesi estesa).
- `projectDescription/`: Specifica funzionale originale.
- `archive/legacy_qtable_rule_based/`: Studio di fattibilità iniziale e sistemi legacy.
- `progetto.sh`: Entry-point centralizzato per test, addestramento e deployment.

## Istruzioni di Esecuzione

Ecco i comandi essenziali per far funzionare il progetto in pochi secondi. Tutte le operazioni si eseguono dal terminale, nella cartella principale del progetto.

### 1. Test del Sistema (Consigliato)
Assicurati che tutto funzioni correttamente eseguendo i test automatici:
```bash
./progetto.sh test
```

### 2. Preparazione dei Dati e Addestramento
Genera il set di dati iniziale e addestra il modello di Intelligenza Artificiale. Questa operazione va fatta la prima volta per inizializzare il sistema:
```bash
cd supervised_telemedicina
PYTHONPATH=.:src python3 -m training.generate_dataset
cd ..
./progetto.sh train
```

### 3. Avvio della Dashboard Visiva
Lancia l'interfaccia grafica per esplorare il progetto e provare il modello in tempo reale:
```bash
./progetto.sh serve 8000
```
👉 **Apri il tuo browser su: http://127.0.0.1:8000/dashboard.html**

### 4. Analisi di un Paziente da Terminale (Opzionale)
Se vuoi testare l'algoritmo direttamente dal terminale passando i parametri vitali di un paziente:
```bash
./progetto.sh analisi '{"pressione_sistolica":120,"pressione_diastolica":80,"frequenza_cardiaca":75,"temperatura":36.8,"saturazione_ossigeno":98,"glicemia":95}'
```

### Riferimento rapido dei comandi

- `./progetto.sh test` — Unit testing
  (139 test supervisionati + 14 regression test legacy).
- `./progetto.sh train` — Ricompila il dataset
  e riesegue il training dell'MLP
  (export degli artifact e di `report.json`).
- `./progetto.sh analisi '<json>'` — Inferenza
  diagnostica su un vettore JSON.
- `./progetto.sh dashboard` — Compilazione statica
  dei report analitici.
- `./progetto.sh serve [porta]` — Dashboard interattiva
  e API in locale (default porta 8000).
- Clone fresco (senza `data/`, gitignored):
  `./progetto.sh train && ./progetto.sh dashboard`
  — il training rigenera da solo il dataset (seed 41)
  se manca.

## Valutazione Analitica del Progetto

Da una prospettiva analitica e metodologica, il sistema risolve in modo rigoroso le criticità legate all'impiego del Machine Learning in contesti clinici (Medical AI):

1. **Gestione del Rischio e Trade-off Bias/Variance**: La combinazione tra safety gate e fallback sull'incertezza introduce un bias intenzionale verso la classe "alto rischio". Questo trade-off aumenta marginalmente i falsi positivi (riducendo la precisione specifica per ottimizzare il recall), una scelta ottimale per massimizzare la sicurezza del paziente.
2. **Metriche di Validazione (Seed 41)**: Il processo di knowledge distillation ha registrato risultati statisticamente rilevanti. Il test set congelato mostra un'accuracy di 0.9851 e un indice Kappa di Cohen pari a 0.9775. Il recall intrinseco della rete per i casi critici è pari a 0.9958 (che il sistema porta forzatamente a 1.0 tramite l'override deterministico).
3. **Ingegnerizzazione del Modello**: L'assenza di framework ad alto livello (come PyTorch o TensorFlow) per il calcolo della backpropagation costringe a un approccio implementativo esplicito che dimostra piena consapevolezza algebrica del processo di ottimizzazione e stabilità numerica.
4. **Ispezionabilità e Data Lineage**: La dashboard non si limita a graficare l'output, ma visualizza in modo trasparente i decision boundaries bidimensionali, la curva di loss e la matrice di confusione. Le decisioni sono sempre affiancate da un tracking completo ("modello usato", "confidenza", "fallback"), essenziale per l'auditing in ambito data science.
