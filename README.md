# Telemedicina – Intelligent Agent

Sistema di telemedicina con agente intelligente per monitoraggio parametri vitali, risposta al paziente, allerta medico e storico su database.

## Task Requirements

| # | Requirement | Implementation |
|---|-------------|----------------|
| 1 | Intelligent agent for telemedicine | Rule-based expert system (symbolic AI) + optional RL Q-learning agent |
| 2 | Manage vital parameters | 6 parameters: blood pressure, heart rate, temperature, SpO2, blood glucose – validated and classified by severity |
| 3 | Return response message to patient | Risk level (low/medium/high) + personalised text recommendations shown on CLI |
| 4 | Inform doctor on unbalanced values | Automatic alert when risk = high or critical patterns detected (shock, hypertensive crisis, sepsis, etc.) |
| 5 | Track interactions on database | SQLite (database/db_manager.py): full history per patient, alert list, aggregate statistics |
| 6 | Submit documentation | This file + documentazione.md with detailed descriptions, rationale, and code sources |

## AI Approach – Rule-Based Expert System (default)

The primary agent uses **symbolic AI** (a rule-based expert system):

- **Knowledge base**: medical reference ranges, severity thresholds, pathological correlation rules encoded as Python dictionaries and conditionals.
- **Inference engine**: sequential rule evaluation – validates input, identifies single-parameter anomalies, checks multi-parameter patterns (e.g. hypotension + tachycardia → possible shock), then computes risk via a heuristic decision tree.
- **No training data required**: works immediately because medical knowledge is explicitly coded.
- **Fully interpretable**: every decision can be traced back to a specific rule.

## RL (Q-learning) Agent – Optional

An RL agent (`agents/rl_agent.py`) using tabular Q-learning with state discretisation, epsilon-greedy policy, and a hardcoded safety override for critical thresholds. See `documentazione.md` for details.

## Requirements

```
pip install -r requirements.txt
```

Python 3.8+ and numpy (required).

## Quick Start

```bash
# Rule-based (default)
python main.py

# RL – auto-trains Q-table if missing, then starts menu
python main.py -RL

# Build Q-table only (training + riepilogo, then exit)
python main.py --build-qt
```

## Menu (once running)

```
1. Inserisci nuovi parametri vitali (Paziente)
2. Visualizza storico parametri
3. Visualizza alert attivi
4. Area Medico – Visualizza pazienti critici
5. Statistiche sistema
0. Esci
```

## Project Structure

```
main.py                            # entry point, argparse with -RL / --build-qt
src/telemedicina/
  config.py                         # paths, RL hyperparameters
  cli.py                            # CLI interface (menu, input, training, riepilogo)
  models/vital_parameters.py        # dataclass + validation + anomaly detection
  agents/intelligent_agent.py       # rule-based expert system (default)
  agents/rl_agent.py                # RL Q-learning agent (optional)
  agents/rl_environment.py          # simulated patient environment for RL training
  database/db_manager.py            # SQLite CRUD operations
  services/analysis_service.py      # orchestrates agent → database → notifications
  utils/notifications.py            # patient/doctor notification logging
data/                               # runtime: telemedicina.db, logs/, models/ (gitignored)
training/train_rl_agent.py          # standalone training script
tests/test_examples.py              # 14 tests covering all components
documentazione.md                   # full project documentation
```

## Testing

```
python -m tests.test_examples
```

14 tests – all pass.

---

*Prototype for educational purposes. Does not substitute professional medical advice.*
