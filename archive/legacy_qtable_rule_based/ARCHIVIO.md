# Archivio: progetto legacy (Q-table / Rule-based)

Questa cartella contiene **l'applicazione legacy** del sistema di telemedicina,
archiviata nella Fase 0 del nuovo progetto. Il codice qui presente **non deve
essere importato, eseguito o referenziato dal nuovo runtime**
(`supervised_telemedicina/`).

> Nota: il file `README.md` presente in questa cartella è il README originale
> del progetto legacy, spostato senza modifiche. Questo file (`ARCHIVIO.md`) è
> la descrizione dell'archivio.

## Contenuto

- `main.py` — entry point legacy (modalità rule-based, RL e build Q-table).
- `src/telemedicina/` — package Python legacy (agenti, database, modelli,
  servizi, utilità, CLI).
- `training/` — script di training dell'agente RL.
- `tests/` — test dell'applicazione legacy.
- `README.md`, `documentazione.md` — documentazione originale.
- `requirements.txt`, `mise.toml` — dipendenze e configurazione tooling.

## Modalità di funzionamento legacy

- **Rule-based**: analisi basata su regole deterministiche sui parametri vitali.
- **Q-learning (RL)**: agente RL con Q-table, con auto-training se la Q-table è
  assente (`python main.py -RL`) o training esplicito (`python main.py --build-qt`).

## Commit di riferimento

- Ultimo commit prima dell'archiviazione: `ba3981c` ("ignore knowledge").
- Commit principali dell'implementazione RL/Q-table: `39c9adf`
  ("RL implementation with Qtable") e `bb02e74`
  ("q table con costruzione evidenziata").

## Vincoli

- **Non modificare** i file legacy spostati in questa cartella.
- **Non creare dipendenze** dal nuovo progetto verso `archive/`.
- La cartella è conservata solo per riferimento storico.