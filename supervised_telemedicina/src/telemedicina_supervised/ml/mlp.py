"""
telemedicina_supervised/ml/mlp.py — MLP a due strati in numpy puro.

Architettura: input -> hidden ReLU -> output softmax.

- forward pass con softmax numericamente stabile (sottrae il massimo);
- loss cross-entropy con clip anti log(0);
- backward pass esplicito con chain rule (gradienti verificati da
  ml/gradient_check.py con differenze finite);
- minibatch SGD con inizializzazione riproducibile (seed su entrambi i
  generatori) e early stopping sulla validation loss;
- serializzazione completa: pesi, bias, normalizzatore, classi,
  architettura e versione del formato (un solo file .npz).

Nessuna soglia clinica è duplicata in questo modulo: il modello lavora su
feature già standardizzate (ml/scaler.py) e le classi sono passate dal
chiamante.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    from telemedicina_supervised.ml.scaler import StandardScaler

VERSIONE_FORMATO: int = 1
_EPS_LOG: float = 1e-12


class MLP:
    """Rete neurale feed-forward 2 strati, addestrata con minibatch SGD."""

    def __init__(
        self,
        n_input: int,
        n_hidden: int,
        classi: Sequence[str],
        seed: int = 41,
    ) -> None:
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.classi = tuple(classi)
        self.n_classi = len(self.classi)
        self.seed = seed
        self.scaler: Optional["StandardScaler"] = None
        rng = np.random.default_rng(seed)
        # Inizializzazione He (adatta a ReLU): varianza ~ 2/fan_in.
        scala_1 = np.sqrt(2.0 / n_input)
        scala_2 = np.sqrt(2.0 / n_hidden)
        self.W1 = rng.normal(0.0, scala_1, size=(n_hidden, n_input))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0.0, scala_2, size=(self.n_classi, n_hidden))
        self.b2 = np.zeros(self.n_classi)

    # ------------------------------------------------------------- forward

    def forward(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Forward pass; restituisce attivazioni e probabilità softmax.

        `z2` contiene i logit GREZZI (prima della sottrazione del massimo):
        utili per debug e curve di loss. La softmax stabile usa internamente
        i logit shiftati.
        """
        z1 = X @ self.W1.T + self.b1
        a1 = np.maximum(z1, 0.0)  # ReLU
        z2 = a1 @ self.W2.T + self.b2
        # Softmax stabile: sottrae il massimo per riga.
        z2_shiftati = z2 - np.max(z2, axis=1, keepdims=True)
        esponenziali = np.exp(z2_shiftati)
        probs = esponenziali / np.sum(esponenziali, axis=1, keepdims=True)
        return {"z1": z1, "a1": a1, "z2": z2, "probs": probs}

    def probabilità(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)["probs"]

    def predici_indici(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.probabilità(X), axis=1)

    def predici_etichette(self, X: np.ndarray) -> np.ndarray:
        indici = self.predici_indici(X)
        return np.array([self.classi[i] for i in indici])

    # ---------------------------------------------------------------- loss

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Cross-entropy media; clip anti log(0)."""
        probs = self.probabilità(X)
        n = probs.shape[0]
        probs_scelte = np.clip(probs[np.arange(n), y], _EPS_LOG, 1.0)
        return float(-np.mean(np.log(probs_scelte)))

    # ------------------------------------------------------------- backward

    def gradienti(self, X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        """Gradienti della loss media rispetto a tutti i parametri."""
        avanti = self.forward(X)
        probs, a1, z1 = avanti["probs"], avanti["a1"], avanti["z1"]
        n = X.shape[0]

        # dL/dz2 per softmax + cross-entropy: (probs - onehot) / n.
        onehot = np.zeros_like(probs)
        onehot[np.arange(n), y] = 1.0
        dz2 = (probs - onehot) / n

        dW2 = dz2.T @ a1
        db2 = np.sum(dz2, axis=0)

        # Chain rule attraverso ReLU (derivata 1 dove z1 > 0).
        da1 = dz2 @ self.W2
        dz1 = da1 * (z1 > 0)

        dW1 = dz1.T @ X
        db1 = np.sum(dz1, axis=0)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    # ---------------------------------------------------------------- train

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epoche: int = 25,
        batch_size: int = 32,
        lr: float = 0.01,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        pazienza: int = 5,
    ) -> Dict[str, List[float]]:
        """Minibatch SGD con early stopping sulla validation loss.

        Restituisce {loss_train: [...], loss_val: [...]} (una voce per
        epoca). I pesi migliori (minor loss_val, o loss_train se la
        validation non è fornita) vengono ripristinati a fine training.
        """
        n = X.shape[0]
        if y.shape[0] != n:
            raise ValueError("X e y hanno lunghezze diverse")
        rng = np.random.default_rng(self.seed + 1)  # seed separato dal primo

        storia: Dict[str, List[float]] = {"loss_train": [], "loss_val": []}
        migliore_loss = float("inf")
        migliore_pesi = self._pesi()
        epoche_senza_miglioramento = 0

        for epoca in range(epoche):
            ordine = rng.permutation(n)
            for inizio in range(0, n, batch_size):
                indice = ordine[inizio : inizio + batch_size]
                X_b, y_b = X[indice], y[indice]
                gradi = self.gradienti(X_b, y_b)
                self._aggiorna(gradi, lr)

            loss_tr = self.loss(X, y)
            storia["loss_train"].append(loss_tr)
            if X_val is not None and y_val is not None:
                loss_va = self.loss(X_val, y_val)
            else:
                loss_va = loss_tr
            storia["loss_val"].append(loss_va)

            if loss_va < migliore_loss:
                migliore_loss = loss_va
                migliore_pesi = self._pesi()
                epoche_senza_miglioramento = 0
            else:
                epoche_senza_miglioramento += 1
                if epoche_senza_miglioramento >= pazienza:
                    break

        self._imposta(migliore_pesi)
        return storia

    def _pesi(self) -> Dict[str, np.ndarray]:
        # Copie indipendenti: altrimenti gli update SGD muterebbero anche il
        # checkpoint scelto per early stopping.
        return {
            "W1": self.W1.copy(),
            "b1": self.b1.copy(),
            "W2": self.W2.copy(),
            "b2": self.b2.copy(),
        }

    def _imposta(self, pesi: Dict[str, np.ndarray]) -> None:
        self.W1, self.b1 = pesi["W1"], pesi["b1"]
        self.W2, self.b2 = pesi["W2"], pesi["b2"]

    def _aggiorna(self, gradi: Dict[str, np.ndarray], lr: float) -> None:
        self.W1 -= lr * gradi["W1"]
        self.b1 -= lr * gradi["b1"]
        self.W2 -= lr * gradi["W2"]
        self.b2 -= lr * gradi["b2"]

    # ---------------------------------------------------------- persistenza

    def salva(self, percorso: Path, scaler: Optional["StandardScaler"] = None) -> None:
        """Salva pesi, bias, architettura, classi, scaler e versione formato.

        Lo scaler viene serializzato DENTRO lo stesso file: in inferenza il
        modello riapplica la normalizzazione calcolata sul solo train. Se
        `scaler` non è passato ma `self.scaler` è già impostato, viene usato
        quello (nessuna perdita silenziosa dello scaler).
        """
        if scaler is None:
            scaler = self.scaler
        dati = {
            "versione_formato": np.array(VERSIONE_FORMATO),
            "n_input": np.array(self.n_input),
            "n_hidden": np.array(self.n_hidden),
            "seed": np.array(self.seed),
            "classi": np.array(self.classi),
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
        }
        if scaler is not None:
            parametri = scaler.to_dict()
            dati["scaler_media"] = parametri["media"]
            dati["scaler_dev"] = parametri["dev"]
            self.scaler = scaler
        percorso = Path(percorso)
        percorso.parent.mkdir(parents=True, exist_ok=True)
        np.savez(percorso, **dati)

    @classmethod
    def carica(cls, percorso: Path) -> "MLP":
        """Ricostruisce il modello da un artifact .npz.

        Se l'artifact contiene lo scaler, questo è esposto come attributo
        `modello.scaler` (None se assente).
        """
        with np.load(percorso) as dati:
            versione = int(dati["versione_formato"])
            if versione != VERSIONE_FORMATO:
                raise ValueError(
                    f"Versione formato {versione} non supportata "
                    f"(attesa {VERSIONE_FORMATO})"
                )
            modello = cls(
                n_input=int(dati["n_input"]),
                n_hidden=int(dati["n_hidden"]),
                classi=[str(c) for c in dati["classi"]],
                seed=int(dati["seed"]),
            )
            modello.W1 = dati["W1"]
            modello.b1 = dati["b1"]
            modello.W2 = dati["W2"]
            modello.b2 = dati["b2"]
            # Validazione coerenza interna: un artifact corrotto deve fallire
            # qui con un errore chiaro, non più tardi con errori oscuri.
            if modello.W1.shape != (modello.n_hidden, modello.n_input):
                raise ValueError(
                    f"W1 con forma {modello.W1.shape} incoerente con "
                    f"architettura ({modello.n_hidden}, {modello.n_input})"
                )
            if modello.W2.shape != (modello.n_classi, modello.n_hidden):
                raise ValueError(
                    f"W2 con forma {modello.W2.shape} incoerente con "
                    f"architettura ({modello.n_classi}, {modello.n_hidden})"
                )
            if modello.b1.shape != (modello.n_hidden,):
                raise ValueError(
                    f"b1 con forma {modello.b1.shape} incoerente con "
                    f"hidden size {modello.n_hidden}"
                )
            if modello.b2.shape != (modello.n_classi,):
                raise ValueError(
                    f"b2 con forma {modello.b2.shape} incoerente con "
                    f"numero classi {modello.n_classi}"
                )
            scaler = None
            if "scaler_media" in dati:
                from telemedicina_supervised.ml.scaler import StandardScaler

                scaler = StandardScaler.from_dict(
                    {"media": dati["scaler_media"], "dev": dati["scaler_dev"]}
                )
        modello.scaler = scaler
        return modello
