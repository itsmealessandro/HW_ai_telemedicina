#!/usr/bin/env python3
"""CLI per training esplicito e inferenza supervised."""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

# Permette ``python main.py`` dalla root del progetto senza installazione.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from telemedicina_supervised.agents.supervised_agent import SupervisedAgent
from telemedicina_supervised.config import (
    DATA_PROCESSED_DIR,
    MODEL_ARTIFACT_PATH,
    MODELS_DIR,
    SEED_DEFAULT,
)
from telemedicina_supervised.services.analysis_service import AnalysisService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Telemedicina supervised CLI")
    azioni = parser.add_mutually_exclusive_group()
    azioni.add_argument("--build-ml", action="store_true", help="addestra e salva il modello")
    azioni.add_argument("-ML", "--run-ml", action="store_true", help="esegue una predizione")
    parser.add_argument("--data-dir", type=Path, default=DATA_PROCESSED_DIR)
    parser.add_argument("--out-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--seed", "--seed-base", dest="seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--modello-path", type=Path, default=MODEL_ARTIFACT_PATH)
    parser.add_argument("--parametri", help="parametri vitali in formato JSON")
    return parser


def _esegui_runtime(parametri_json: str, modello_path: Path) -> int:
    try:
        parametri = json.loads(parametri_json)
    except (json.JSONDecodeError, TypeError) as exc:
        print(f"JSON non valido: {exc}", file=sys.stderr)
        return 2
    if not isinstance(parametri, dict):
        print("Input non valido: --parametri deve contenere un oggetto JSON.", file=sys.stderr)
        return 2

    # Percorso applicativo unico: la CLI è un thin wrapper sul servizio,
    # che orchestra validazione -> SupervisedAgent -> esito (in Fase 7
    # persistenza e notifiche vivranno qui, senza duplicare l'analisi).
    servizio = AnalysisService(agent=SupervisedAgent(modello_path=modello_path))
    esito = servizio.analizza(parametri)
    outcome = esito.esito_agente
    if outcome is None:
        print("Errore interno: esito agente non disponibile.", file=sys.stderr)
        return 1

    print(f"Classe: {esito.classe}")
    if outcome.anomalie:
        anomalie = "; ".join(
            f"{anomalia['parametro']}={anomalia['valore']}"
            f" ({anomalia['gravita']}, {anomalia['direzione']})"
            for anomalia in outcome.anomalie
        )
        print("Anomalie: " + anomalie)
    else:
        print("Anomalie: nessuna")
    print("Pattern: " + ("; ".join(outcome.pattern) if outcome.pattern else "nessuno"))

    if esito.errori:
        print("Errori: " + "; ".join(esito.errori))

    if outcome.probabilita is None:
        print("Probabilità: non disponibile (baseline rule-based)")
    else:
        probabilita = "; ".join(
            f"{classe}={valore:.4f}"
            for classe, valore in outcome.probabilita.items()
        )
        print("Probabilità: " + probabilita)

    if outcome.fallback:
        if outcome.motivo_fallback and outcome.motivo_fallback.startswith("modello non trovato"):
            print(f"{outcome.motivo_fallback}; eseguire python main.py --build-ml")
        else:
            print(f"Fallback rule-based: {outcome.motivo_fallback or 'attivo'}")
    else:
        print("Modello MLP usato: " + ("si" if outcome.modello_usato else "no"))
    print("Metadati: " + json.dumps(outcome.metadati, ensure_ascii=False))
    print(f"Raccomandazione: {outcome.messaggio}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.build_ml and not args.run_ml:
        parser.print_help()
        return 0
    if args.build_ml:
        if args.parametri is not None:
            parser.error("--parametri è valido solo con --run-ml/-ML")
        from training import train_model
        print("Avvio training MLP esplicito...")
        try:
            train_model.main(
                [
                    "--seed-base",
                    str(args.seed),
                    "--data-dir",
                    str(args.data_dir),
                    "--out-dir",
                    str(args.out_dir),
                ]
            )
        except Exception as exc:
            print(f"Errore durante il training: {exc}", file=sys.stderr)
            return 1
        return 0
    if not args.parametri:
        print("Errore: --parametri JSON è richiesto con --run-ml.", file=sys.stderr)
        return 2
    return _esegui_runtime(args.parametri, args.modello_path)


if __name__ == "__main__":
    sys.exit(main())
