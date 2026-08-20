"""tools/_figure.py — Figure matplotlib per la dashboard (Fase V3).

Ogni funzione restituisce una stringa base64 (PNG, senza prefisso) oppure
``None`` se matplotlib non è disponibile o la figura non è generabile.
matplotlib è importato LAZY (mai a livello di modulo): il tool e i test
funzionano anche senza la dipendenza dev (``requirements_dashboard.txt``).

Contratto dei numeri: le metriche dichiarate (accuracy, kappa, recall, ...)
NON vengono ricalcolate qui — arrivano già pronte da ``report.json``. Qui si
ricalcola in numpy puro solo ciò che serve alle figure: predizioni dell'MLP
sul test congelato, distanza dalle soglie (riuso di
``training.metrics.distanza_dalle_soglie``) e label del teacher sulla griglia
(riuso di ``training.teacher_rules.label``, fonte unica). Nessuna soglia
clinica è duplicata in questo modulo.
"""

import base64
import io
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from telemedicina_supervised.config import PAZIENZA  # noqa: E402
from telemedicina_supervised.safety.safety_rules import (  # noqa: E402
    CLASSI,
    FEATURE_ORDER,
    RANGE_NORMALI,
)
from training.metrics import distanza_dalle_soglie  # noqa: E402
from training.teacher_rules import label as label_teacher  # noqa: E402

# Seed fisso e dichiarato per il campionamento dello scatter (dati congelati:
# il campione estratto è sempre lo stesso, nessun seed nuovo non dichiarato).
SEED_CAMPIONE = 41
MAX_PUNTI_SCATTER = 4000
GRIGLIA = 200

COLORI_CLASSI = {
    "basso": "#2e7d32",
    "medio": "#f9a825",
    "alto": "#c62828",
    "errore": "#9e9e9e",
}


def _mpl():
    """Import lazy di matplotlib con backend Agg (alza ImportError se assente)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _png(fig, plt) -> Optional[str]:
    """Figura -> base64 PNG (dpi contenuti per un HTML leggero)."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def figura_loss_curve(report: Dict[str, Any]) -> Optional[str]:
    """Curve loss train/val della config migliore (25 epoche, early stopping).

    I valori arrivano da ``report.json`` (loss_train_finale/loss_val_finale);
    il punto di miglior val loss è evidenziato con la pazienza di early
    stopping letta da ``config.PAZIENZA`` (fonte ufficiale).
    """
    try:
        plt = _mpl()
        train = report.get("loss_train_finale")
        val = report.get("loss_val_finale")
        if (
            not isinstance(train, list)
            or not isinstance(val, list)
            or not train
            or len(train) != len(val)
        ):
            return None
        epoche = list(range(1, len(train) + 1))
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epoche, train, label="loss train", marker="o", markersize=3)
        ax.plot(epoche, val, label="loss val", marker="o", markersize=3)
        migliore = int(np.argmin(val)) + 1
        ax.axvline(migliore, color="#c62828", linestyle="--", linewidth=1)
        ax.annotate(
            f"miglior val loss: epoca {migliore}",
            xy=(migliore, min(val)),
            xytext=(migliore + 0.5, min(val) + 0.03),
            fontsize=9,
        )
        ax.set_xlabel("Epoca")
        ax.set_ylabel("Loss")
        ax.set_title(f"Loss train/val — config migliore (early stopping, pazienza {PAZIENZA})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        return _png(fig, plt)
    except Exception:
        return None


def figura_confusione(matrice: Any, titolo: str) -> Optional[str]:
    """Heatmap della matrice di confusione (da report.json, mai ricalcolata)."""
    try:
        plt = _mpl()
        m = np.asarray(matrice, dtype=np.int64)
        if m.ndim != 2 or m.shape[0] != m.shape[1] or m.shape[0] != len(CLASSI):
            return None
        fig, ax = plt.subplots(figsize=(5, 4.2))
        im = ax.imshow(m, cmap="Blues")
        ax.set_xticks(range(m.shape[1]))
        ax.set_xticklabels(CLASSI)
        ax.set_yticks(range(m.shape[0]))
        ax.set_yticklabels(CLASSI)
        ax.set_xlabel("Predetto")
        ax.set_ylabel("Reale")
        ax.set_title(titolo)
        soglia_colore = m.max() / 2
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(
                    j,
                    i,
                    str(m[i, j]),
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if m[i, j] > soglia_colore else "black",
                )
        fig.colorbar(im, ax=ax, fraction=0.046)
        return _png(fig, plt)
    except Exception:
        return None


def figura_scatter_errori(
    X_test: np.ndarray, y_test: np.ndarray, modello: Any, scaler: Any
) -> Optional[str]:
    """Scatter 2D sistolica × glicemia: errori evidenziati, colore = distanza.

    Gli errori sono le predizioni dell'MLP diverse dalla label del test
    congelato; il colore è la distanza dalle soglie (riuso di
    ``training.metrics.distanza_dalle_soglie``). Campione massimo
    ``MAX_PUNTI_SCATTER`` con seed fisso dichiarato.
    """
    try:
        plt = _mpl()
        X = np.asarray(X_test, dtype=np.float64)
        y = np.asarray(y_test)
        if X.ndim != 2 or X.shape[0] != y.shape[0] or scaler is None:
            return None
        rng = np.random.default_rng(SEED_CAMPIONE)
        n = X.shape[0]
        if n > MAX_PUNTI_SCATTER:
            indice = rng.choice(n, size=MAX_PUNTI_SCATTER, replace=False)
            X, y = X[indice], y[indice]
        pred = np.asarray(modello.predici_etichette(scaler.transform(X)))
        distanze = distanza_dalle_soglie(X)
        errori = y != pred
        i_sist = FEATURE_ORDER.index("pressione_sistolica")
        i_glic = FEATURE_ORDER.index("glicemia")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.scatter(
            X[~errori, i_sist],
            X[~errori, i_glic],
            s=8,
            c="#9e9e9e",
            alpha=0.5,
            label="corretti",
        )
        punti = ax.scatter(
            X[errori, i_sist],
            X[errori, i_glic],
            s=18,
            c=distanze[errori],
            cmap="viridis",
            edgecolors="#1c1e21",
            linewidths=0.4,
            label=f"errori ({int(errori.sum())})",
        )
        fig.colorbar(punti, ax=ax, label="distanza dalle soglie")
        ax.set_xlabel("Pressione sistolica (mmHg)")
        ax.set_ylabel("Glicemia (mg/dL)")
        ax.set_title("Test congelato: errori dell'MLP per distanza dalle soglie")
        ax.legend(loc="upper right")
        return _png(fig, plt)
    except Exception:
        return None


def figura_heatmap_regioni(
    X_test: np.ndarray, modello: Any, scaler: Any
) -> Optional[str]:
    """Regioni di decisione affiancate: teacher (rule-based) vs MLP.

    Griglia ``GRIGLIA``×``GRIGLIA`` su sistolica × glicemia (range dal test
    congelato); le altre 4 feature al centro del range normale
    (``RANGE_NORMALI``, fonte ufficiale). Label del teacher via
    ``training.teacher_rules.label`` (fonte unica, funzione scalare) e label
    dell'MLP via artifact (scaler + ``predici_etichette``, vettorizzato).
    """
    try:
        plt = _mpl()
        X = np.asarray(X_test, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != len(FEATURE_ORDER) or scaler is None:
            return None
        i_sist = FEATURE_ORDER.index("pressione_sistolica")
        i_glic = FEATURE_ORDER.index("glicemia")
        sist = X[:, i_sist]
        glic = X[:, i_glic]
        ampiezza_s = sist.max() - sist.min()
        ampiezza_g = glic.max() - glic.min()
        if ampiezza_s <= 0 or ampiezza_g <= 0:
            return None
        estensione = (
            float(sist.min() - 0.05 * ampiezza_s),
            float(sist.max() + 0.05 * ampiezza_s),
            float(glic.min() - 0.05 * ampiezza_g),
            float(glic.max() + 0.05 * ampiezza_g),
        )
        assi_sist = np.linspace(estensione[0], estensione[1], GRIGLIA)
        assi_glic = np.linspace(estensione[2], estensione[3], GRIGLIA)
        normali = {
            nome: (RANGE_NORMALI[nome][0] + RANGE_NORMALI[nome][1]) / 2
            for nome in FEATURE_ORDER
        }
        griglia = np.empty((GRIGLIA * GRIGLIA, len(FEATURE_ORDER)))
        for r, g in enumerate(assi_glic):
            for c, s in enumerate(assi_sist):
                riga = [normali[nome] for nome in FEATURE_ORDER]
                riga[i_sist] = s
                riga[i_glic] = g
                griglia[r * GRIGLIA + c] = riga

        # Teacher: label rule-based per punto (fonte unica, funzione scalare).
        etichette_teacher = np.array(
            [label_teacher(dict(zip(FEATURE_ORDER, riga))) for riga in griglia]
        )
        # MLP: predizione vettorizzata su input standardizzati dallo scaler.
        etichette_mlp = np.array(modello.predici_etichette(scaler.transform(griglia)))

        mappa = {classe: i for i, classe in enumerate(CLASSI)}

        def _indici(etichette: np.ndarray) -> np.ndarray:
            return np.array([mappa.get(e, len(CLASSI)) for e in etichette])

        z_teacher = _indici(etichette_teacher).reshape(GRIGLIA, GRIGLIA)
        z_mlp = _indici(etichette_mlp).reshape(GRIGLIA, GRIGLIA)
        colori = [COLORI_CLASSI[c] for c in CLASSI] + [COLORI_CLASSI["errore"]]
        from matplotlib.colors import ListedColormap

        cmap = ListedColormap(colori)
        fig, assi = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, z, titolo in (
            (assi[0], z_teacher, "Teacher (rule-based)"),
            (assi[1], z_mlp, "MLP (distillato)"),
        ):
            # z ha forma [glicemia, sistolica] (righe=glic, colonne=sist):
            # imshow mappa la prima dimensione sull'asse y e la seconda su x,
            # quindi con origin="lower" e extent=(sist, glic) gli assi sono
            # corretti senza trasporre.
            ax.imshow(
                z,
                origin="lower",
                extent=estensione,
                aspect="auto",
                cmap=cmap,
                vmin=0,
                vmax=len(CLASSI),
            )
            ax.set_xlabel("Pressione sistolica (mmHg)")
            ax.set_ylabel("Glicemia (mg/dL)")
            ax.set_title(titolo)
        from matplotlib.patches import Patch

        etichette_legenda = list(CLASSI) + ["errore"]
        patch = [
            Patch(color=colori[i], label=etichette_legenda[i])
            for i in range(len(CLASSI) + 1)
        ]
        fig.legend(handles=patch, loc="lower center", ncol=4, frameon=False)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        return _png(fig, plt)
    except Exception:
        return None


def figura_architettura(n_hidden: Optional[int] = None) -> Optional[str]:
    """Diagramma semplice: input 6 → hidden ReLU → softmax 3."""
    try:
        plt = _mpl()
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.axis("off")

        def scatola(x: float, y: float, testo: str, colore: str) -> None:
            from matplotlib.patches import Rectangle

            ax.add_patch(
                Rectangle(
                    (x - 0.9, y - 0.45),
                    1.8,
                    0.9,
                    facecolor=colore,
                    edgecolor="#1c1e21",
                    linewidth=1.2,
                    zorder=2,
                )
            )
            ax.text(x, y, testo, ha="center", va="center", fontsize=9, zorder=3)

        scatola(0, 0, "Input\n6 feature", "#cfe0f0")
        hidden = f"Hidden ReLU\n{n_hidden} neuroni" if n_hidden else "Hidden ReLU"
        scatola(3, 0, hidden, "#c8e6c9")
        scatola(6, 0, "Softmax\n3 classi", "#ffe0b2")
        for x1, x2 in ((0.9, 2.1), (3.9, 5.1)):
            ax.annotate(
                "",
                xy=(x2, 0),
                xytext=(x1, 0),
                arrowprops=dict(arrowstyle="->", color="#1c1e21", lw=1.5),
            )
        ax.text(
            1.5,
            0.6,
            "standardizzazione\n(scaler dal train)",
            ha="center",
            fontsize=8,
            color="#555555",
        )
        ax.set_xlim(-1.5, 7.5)
        ax.set_ylim(-1.2, 1.2)
        return _png(fig, plt)
    except Exception:
        return None