"""
training/train_model.py — Training, model selection e valutazione (Fase 4).

Workflow (slide 9):
  1. dataset congelato da Fase 2 (train/val/test già separati, test mai
     toccato durante il tuning);
  2. scaler calcolato SOLO sul train e salvato nell'artifact;
  3. confronto di pochi iperparametri predefiniti (hidden, lr, batch,
     epoche) con early stopping;
  4. scelta sulla validation loss, mai sul test;
  5. riaddestramento finale con la configurazione migliore;
  6. valutazione UNA sola volta sul test congelato;
  7. report JSON riproducibile (stesso seed -> stessi numeri).

Il report include: accuracy, precision/recall/macro-F1 per classe, matrice
di confusione, Cohen's kappa vs teacher, analisi errori per distanza dalle
soglie, andamento train/val loss, distribuzione classi, baseline rule-based
(upper bound di imitazione) e metriche di sicurezza (mancato riconoscimento
di 'alto' e pattern critici).

Uso:
    PYTHONPATH=src python -m training.train_model --seed-base 41
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from telemedicina_supervised.ml.labels import to_indici
from telemedicina_supervised.ml.mlp import MLP
from telemedicina_supervised.ml.scaler import StandardScaler
from telemedicina_supervised.safety.safety_rules import CLASSI

from training.metrics import (
    distanza_dalle_soglie,
    riepilogo_metriche,
)
from training.synthetic_generator import genera_batch

# Iperparametri predefiniti da confrontare (scelta su validation).
GRID: List[Dict[str, Any]] = [
    {"n_hidden": 16, "lr": 0.01, "batch_size": 32, "epoche": 25},
    {"n_hidden": 16, "lr": 0.05, "batch_size": 32, "epoche": 25},
    {"n_hidden": 32, "lr": 0.01, "batch_size": 32, "epoche": 25},
    {"n_hidden": 32, "lr": 0.05, "batch_size": 32, "epoche": 25},
    {"n_hidden": 64, "lr": 0.01, "batch_size": 64, "epoche": 25},
    {"n_hidden": 64, "lr": 0.05, "batch_size": 64, "epoche": 25},
]

SEED_DEFAULT: int = 41
PAZIENZA: int = 5


def carica_dataset(directory: Path) -> Dict[str, np.ndarray]:
    """Carica i blocchi congelati da data/processed/."""
    dati: Dict[str, np.ndarray] = {}
    for nome in ("train", "val", "test"):
        dati[f"X_{nome}"] = np.load(directory / f"X_{nome}.npy")
        dati[f"y_{nome}"] = np.load(directory / f"y_{nome}.npy")
    return dati


def prepara(
    dati: Dict[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], StandardScaler]:
    """Scaler sul solo train + conversione label in indici."""
    scaler = StandardScaler().fit(dati["X_train"])
    preparati: Dict[str, np.ndarray] = {}
    for nome in ("train", "val", "test"):
        preparati[f"X_{nome}"] = scaler.transform(dati[f"X_{nome}"])
        preparati[f"y_{nome}"] = to_indici(dati[f"y_{nome}"], CLASSI)
    return preparati, scaler


def addestra_configurazione(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Dict[str, Any],
    seed: int,
) -> Tuple[MLP, Dict[str, List[float]]]:
    """Addestra una configurazione e restituisce (modello, storia loss)."""
    modello = MLP(
        n_input=X_train.shape[1],
        n_hidden=int(config["n_hidden"]),
        classi=CLASSI,
        seed=seed,
    )
    storia = modello.train(
        X_train,
        y_train,
        epoche=int(config["epoche"]),
        batch_size=int(config["batch_size"]),
        lr=float(config["lr"]),
        X_val=X_val,
        y_val=y_val,
        pazienza=PAZIENZA,
    )
    return modello, storia


def seleziona_migliore(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
) -> Tuple[Dict[str, object], float, Dict[str, List[float]]]:
    """Confronta la GRID sulla validation loss; restituisce la migliore."""
    migliore_config: Dict[str, object] = {}
    migliore_loss = float("inf")
    migliore_storia: Dict[str, List[float]] = {}
    for config in GRID:
        modello, storia = addestra_configurazione(
            X_train, y_train, X_val, y_val, config, seed
        )
        loss_val = min(storia["loss_val"])
        print(
            f"[grid] hidden={config['n_hidden']} lr={config['lr']} "
            f"batch={config['batch_size']} epoche={config['epoche']} "
            f"-> loss_val={loss_val:.4f}"
        )
        if loss_val < migliore_loss:
            migliore_loss = loss_val
            migliore_config = config
            migliore_storia = storia
    return migliore_config, migliore_loss, migliore_storia


def analisi_errori_per_distanza(
    X_test: np.ndarray,
    y_test: np.ndarray,
    predizioni: np.ndarray,
) -> Dict[str, Any]:
    """Errori raggruppati per distanza minima dalle soglie."""
    distanze = distanza_dalle_soglie(X_test)
    errori = y_test != predizioni
    if not errori.any():
        return {"n_errori": 0, "distanza_media_errori": 0.0, "distanza_max_errori": 0.0}
    distanze_errori = distanze[errori]
    return {
        "n_errori": int(errori.sum()),
        "distanza_media_errori": float(distanze_errori.mean()),
        "distanza_max_errori": float(distanze_errori.max()),
    }


def baseline_rule_based(
    X_test: np.ndarray, y_test: np.ndarray
) -> Dict[str, Any]:
    """Baseline: il teacher sulle proprie label (upper bound di imitazione).

    Le label del dataset sono generate dal teacher: la concordanza attesa è
    pressoché totale per costruzione. Serve come riferimento per
    interpretare l'accuracy dell'MLP (approssimazione delle regole, non
    accuratezza diagnostica).
    """
    from telemedicina_supervised.safety.safety_rules import FEATURE_ORDER

    from training.teacher_rules import label as label_teacher

    predette = np.array(
        [
            label_teacher(dict(zip(FEATURE_ORDER, campione)))
            for campione in X_test
        ],
        dtype="U6",
    )
    accordo = float(np.mean(predette == y_test))
    return {"accordo_teacher_label": accordo, "n_campioni": int(y_test.size)}


def genera_test_no_buffer(seed_base: int, n_test: int) -> Tuple[np.ndarray, np.ndarray]:
    """Test aggiuntivo SENZA buffer zone (stesso seed del test congelato).

    Il test congelato è generato con buffer: le metriche su di esso sono
    ottimistiche sui casi near-threshold. Questo slice senza buffer misura
    la generalizzazione sui valori vicini alle soglie (nota gate Fase 2).
    """
    X, y, _ = genera_batch(seed_base + 2, n_test, buffer_zone=False)
    return X, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Training MLP telemedicina")
    parser.add_argument("--seed-base", type=int, default=SEED_DEFAULT)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/models"),
    )
    args = parser.parse_args()

    dati = carica_dataset(args.data_dir)
    preparati, scaler = prepara(dati)

    X_train, y_train = preparati["X_train"], preparati["y_train"]
    X_val, y_val = preparati["X_val"], preparati["y_val"]
    X_test, y_test = preparati["X_test"], preparati["y_test"]

    print("=" * 56)
    print("TRAINING MLP (teacher rule-based, knowledge distillation)")
    print("=" * 56)
    print(f"train={X_train.shape[0]} val={X_val.shape[0]} test={X_test.shape[0]}")

    # 1) Model selection sulla validation.
    migliore, loss_val_migliore, storia = seleziona_migliore(
        X_train, y_train, X_val, y_val, args.seed_base
    )
    print(f"[scelta] config migliore: {migliore} (loss_val={loss_val_migliore:.4f})")

    # 2) Riaddestramento finale con la configurazione migliore.
    modello_finale, storia_finale = addestra_configurazione(
        X_train, y_train, X_val, y_val, migliore, args.seed_base
    )
    modello_finale.scaler = scaler

    # 3) Valutazione UNA volta sul test congelato.
    predizioni = modello_finale.predici_indici(X_test)
    metriche = riepilogo_metriche(
        y_test, predizioni, n_classi=len(CLASSI), indice_alto=2
    )
    # Le distanze dalle soglie vanno calcolate sui valori GREZZI (le soglie
    # di safety_rules sono in unità cliniche, non standardizzate).
    errori_distanza = analisi_errori_per_distanza(
        dati["X_test"], y_test, predizioni
    )
    baseline = baseline_rule_based(dati["X_test"], dati["y_test"])

    # 4) Slice senza buffer (generalizzazione near-threshold).
    X_nb, y_nb = genera_test_no_buffer(args.seed_base, int(dati["y_test"].size))
    X_nb_norm = scaler.transform(X_nb)
    y_nb_idx = to_indici(y_nb, CLASSI)
    predizioni_nb = modello_finale.predici_indici(X_nb_norm)
    metriche_nb = riepilogo_metriche(
        y_nb_idx, predizioni_nb, n_classi=len(CLASSI), indice_alto=2
    )

    # 5) Report riproducibile.
    report = {
        "seed_base": args.seed_base,
        "config_migliore": {k: v for k, v in migliore.items()},
        "loss_val_migliore": loss_val_migliore,
        "loss_train_finale": storia_finale["loss_train"],
        "loss_val_finale": storia_finale["loss_val"],
        "distribuzione_classi": {
            nome: {
                str(i): int((preparati[f"y_{nome}"] == i).sum())
                for i in range(len(CLASSI))
            }
            for nome in ("train", "val", "test")
        },
        "metriche_test_congelato": metriche,
        "metriche_test_no_buffer": metriche_nb,
        "analisi_errori_distanza": errori_distanza,
        "baseline_rule_based": baseline,
        "interpretazione": (
            "Le label sono generate dal teacher rule-based: accuracy alta = "
            "buona approssimazione delle regole sintetiche, NON accuratezza "
            "diagnostica su pazienti reali."
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    percorso_artifact = args.out_dir / "mlp_telemedicina.npz"
    modello_finale.salva(percorso_artifact, scaler=scaler)
    with open(args.out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nArtifact salvato: {percorso_artifact}")
    print(f"Report salvato: {args.out_dir / 'report.json'}")
    print(f"Accuracy test congelato: {metriche['accuracy']:.4f}")
    print(f"Accuracy test no-buffer: {metriche_nb['accuracy']:.4f}")
    print(f"Kappa vs teacher (test congelato): {metriche['kappa_cohen']:.4f}")
    print(f"Recall 'alto' (test congelato): {metriche['sicurezza']['recall_alto']:.4f}")


if __name__ == "__main__":
    main()