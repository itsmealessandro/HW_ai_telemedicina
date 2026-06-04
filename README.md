# Telemedicina – Intelligent Agent

Sistema di telemedicina con agente intelligente per monitoraggio parametri vitali, risposta al paziente, allerta medico e storico su database.

## Task Requirements

| # | Requirement | Implementation |
|---|-------------|----------------|
| 1 | Intelligent agent for telemedicine | Rule-based expert system (symbolic AI) in `agents/intelligent_agent.py` |
| 2 | Manage vital parameters | 6 parameters: blood pressure, heart rate, temperature, SpO2, blood glucose – validated and classified by severity |
| 3 | Return response message to patient | Risk level (low/medium/high) + personalised text recommendations shown on CLI |
| 4 | Inform doctor on unbalanced values | Automatic alert when risk = high or critical patterns detected (shock, hypertensive crisis, sepsis, etc.) |
| 5 | Track interactions on database | SQLite (`database/db_manager.py`): full history per patient, alert list, aggregate statistics |
| 6 | Submit documentation | This file + `documentazione.md` with detailed descriptions, rationale, and code sources |

## AI Approach – Rule-Based Expert System

The intelligent agent uses **symbolic AI** (a rule-based expert system) rather than machine learning:

- **Knowledge base**: medical reference ranges, severity thresholds, pathological correlation rules encoded as Python dictionaries and conditionals.
- **Inference engine**: sequential rule evaluation – validates input, identifies single-parameter anomalies, checks multi-parameter patterns (e.g. hypotension + tachycardia → possible shock), then computes risk via a heuristic decision tree.
- **No training data required**: the agent works immediately because medical knowledge is explicitly coded.
- **Fully interpretable**: every decision can be traced back to a specific rule – important for medical accountability.

## Project Structure

```
main.py                        # entry point, delegates to src/telemedicina/cli.py
src/telemedicina/
  config.py                     # paths for db and logs (absolute via pathlib)
  cli.py                        # CLI interface (menu, input, output)
  models/vital_parameters.py    # dataclass + validation + anomaly detection
  agents/intelligent_agent.py   # rule-based expert system (the AI agent)
  database/db_manager.py        # SQLite CRUD operations
  services/analysis_service.py  # orchestrates agent → db → notifications
  utils/notifications.py        # patient/doctor notification logging
data/                           # runtime: telemedicina.db + logs/ (gitignored)
tests/test_examples.py          # 14 tests covering all components
documentazione.md               # full project documentation
```

## Quick Start

```
python main.py
```

Follow the menu: insert patient data and vital parameters, get risk analysis, view history and alerts.

## Testing

```
python -m tests.test_examples
```

14 tests – all pass.

---

*Prototype for educational purposes. Does not substitute professional medical advice.*
