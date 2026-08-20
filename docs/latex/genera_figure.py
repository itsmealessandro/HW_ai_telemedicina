#!/usr/bin/env python3
"""Genera le figure PNG del documento LaTeX dai dati reali del progetto.

Riusa tools/_figure.py (le stesse figure della dashboard): nessuna logica
duplicata. Output: figure/generate/*.png

Uso:
    python3 figure/genera_figure.py
"""
import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]          # repo root
SUP = ROOT / "supervised_telemedicina"
sys.path.insert(0, str(SUP / "tools"))
sys.path.insert(0, str(SUP / "src"))

import _figure  # noqa: E402
from genera_dashboard import _carica_test_e_modello  # noqa: E402

OUT = Path(__file__).resolve().parent / "generated"
OUT.mkdir(exist_ok=True)


def salva(nome: str, b64: str) -> bool:
    if not b64:
        print(f"  MANCANTE: {nome}")
        return False
    (OUT / nome).write_bytes(base64.b64decode(b64))
    print(f"  OK: {nome}")
    return True


def main() -> int:
    report = json.loads(
        (SUP / "data" / "models" / "report.json").read_text(encoding="utf-8")
    )
    X_test, y_test, modello, errore = _carica_test_e_modello()
    if errore:
        print(f"Errore: {errore}", file=sys.stderr)
        return 1

    print("Figure generate:")
    salva("loss_curve.png", _figure.figura_loss_curve(report))
    salva(
        "confusione.png",
        _figure.figura_confusione(
            report["metriche_test_congelato"]["matrice_confusione"],
            "Matrice di confusione (test congelato)",
        ),
    )
    salva(
        "scatter_errori.png",
        _figure.figura_scatter_errori(X_test, y_test, modello, modello.scaler),
    )
    salva(
        "heatmap_regioni.png",
        _figure.figura_heatmap_regioni(X_test, modello, modello.scaler),
    )
    salva(
        "architettura.png",
        _figure.figura_architettura(
            report.get("config_migliore", {}).get("n_hidden")
        ),
    )
    salva(
        "istogramma_confidenze.png",
        _figure.figura_istogramma_confidenze(X_test, modello, modello.scaler),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())