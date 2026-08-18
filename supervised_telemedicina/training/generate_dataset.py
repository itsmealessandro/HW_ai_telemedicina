"""
training/generate_dataset.py — Genera e congela il dataset sintetico.

Crea i tre blocchi train/validation/test con seed DERIVATI e DIVERSI
(train=seed_base, val=seed_base+1, test=seed_base+2) e li salva su disco in
`data/processed/` insieme ai metadati di riproducibilità (seed, dimensioni,
conteggi per classe, margini buffer, hash di safety_rules).

Il test set è CONGELATO: generato con seed diverso e mai toccato durante il
tuning (Fase 4). Rigenerare con lo stesso seed-base produce file identici.

Uso:
    PYTHONPATH=src python -m training.generate_dataset --seed-base 41

Nessuna soglia numerica è duplicata qui: il generatore (synthetic_generator)
deriva i limiti da safety_rules e le label dal teacher.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from training.synthetic_generator import MARGINI_BUFFER, genera_batch

CLASSI: List[str] = ["basso", "medio", "alto"]

# Seed di default: 41 (evita collisioni con i literal di soglia controllati
# dal guard test tests/test_single_source.py).
SEED_BASE_DEFAULT: int = 41


def _hash_file(percorso: Path) -> str:
    """Hash sha256 (12 char) di un file: identifica la versione del codice."""
    return hashlib.sha256(percorso.read_bytes()).hexdigest()[:12]


def _hash_safety_rules() -> str:
    """Hash del file safety_rules.py: identifica la versione delle soglie."""
    percorso = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "telemedicina_supervised"
        / "safety"
        / "safety_rules.py"
    )
    return _hash_file(percorso)


def _hash_training(relativo: str) -> str:
    """Hash di un modulo di training (es. 'synthetic_generator.py')."""
    percorso = Path(__file__).resolve().parent / relativo
    return _hash_file(percorso)


def _conteggi(y: np.ndarray) -> Dict[str, int]:
    """Conteggi per classe (ordine CLASSI)."""
    return {classe: int(np.sum(y == classe)) for classe in CLASSI}


def genera_e_salva(
    seed_base: int,
    n_train: int,
    n_val: int,
    n_test: int,
    out_dir: Path,
    buffer_zone: bool,
) -> Dict[str, object]:
    """Genera i tre blocchi, li salva e restituisce i metadati."""
    out_dir.mkdir(parents=True, exist_ok=True)

    blocchi = {
        "train": (seed_base, n_train),
        "val": (seed_base + 1, n_val),
        "test": (seed_base + 2, n_test),
    }

    metadati: Dict[str, object] = {
        "seed_base": seed_base,
        "buffer_zone": buffer_zone,
        "margini_buffer": MARGINI_BUFFER,
        "hash_safety_rules": _hash_safety_rules(),
        "hash_synthetic_generator": _hash_training("synthetic_generator.py"),
        "hash_teacher_rules": _hash_training("teacher_rules.py"),
        "blocchi": {},
    }

    print("=" * 56)
    print("GENERAZIONE DATASET SINTETICO (teacher rule-based)")
    print("=" * 56)
    for nome, (seed, n) in blocchi.items():
        X, y, scartati = genera_batch(seed, n, buffer_zone=buffer_zone)

        # Controllo classi PRIMA di scrivere su disco (evita file parziali),
        # esteso a tutti gli split (non solo train).
        conteggi = _conteggi(y)
        classi_assenti = [c for c, v in conteggi.items() if v == 0]
        if classi_assenti:
            raise RuntimeError(
                f"Classe assente nel blocco {nome}: {classi_assenti}. "
                "Il dataset non è utilizzabile per il training."
            )

        np.save(out_dir / f"X_{nome}.npy", X)
        np.save(out_dir / f"y_{nome}.npy", y)

        metadati["blocchi"][nome] = {  # type: ignore[index]
            "seed": seed,
            "n": int(n),
            "scartati": int(scartati),
            "conteggi": conteggi,
        }
        print(f"\n[{nome}] seed={seed} n={n} scartati={scartati}")
        for classe in CLASSI:
            print(f"  {classe:>6}: {conteggi[classe]:>6}")

    with (out_dir / "metadati.json").open("w", encoding="utf-8") as f:
        json.dump(metadati, f, indent=2, ensure_ascii=False)
    print(f"\nSalvato in: {out_dir.resolve()}")
    return metadati


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera e congela il dataset sintetico etichettato."
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=SEED_BASE_DEFAULT,
        help=f"seed base (default: {SEED_BASE_DEFAULT}); train=seed, "
        "val=seed+1, test=seed+2",
    )
    parser.add_argument("--n-train", type=int, default=40000)
    parser.add_argument("--n-val", type=int, default=10000)
    parser.add_argument("--n-test", type=int, default=20000)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed"),
        help="directory di output (default: data/processed)",
    )
    parser.add_argument(
        "--no-buffer",
        action="store_true",
        help="disabilita la buffer zone attorno alle soglie",
    )
    args = parser.parse_args()

    genera_e_salva(
        seed_base=args.seed_base,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        out_dir=args.out_dir,
        buffer_zone=not args.no_buffer,
    )


if __name__ == "__main__":
    main()