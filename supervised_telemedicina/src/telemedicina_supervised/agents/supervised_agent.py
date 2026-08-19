"""Safety-gated supervised agent (rule-based gate + optional MLP)."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from telemedicina_supervised.config import (
    MODEL_ARTIFACT_PATH,
    SOGLIA_INCERTEZZA,
    carica_modello,
)
from telemedicina_supervised.ml.mlp import MLP
from telemedicina_supervised.safety.safety_rules import (
    CLASSI,
    CLASSE_ERRORE,
    FEATURE_ORDER,
    analizza,
)


@dataclass
class AgentOutcome:
    """Esito compatibile con il contratto baseline, con metadati Fase 5."""

    classe: str
    probabilita: Optional[Dict[str, float]] = None
    errore: bool = False
    messaggio: str = ""
    modello_usato: bool = False
    classe_mlp: Optional[str] = None
    classe_regola: Optional[str] = None
    confidenza: Optional[float] = None
    fallback: bool = False
    motivo_fallback: Optional[str] = None
    override_sicurezza: bool = False
    # Dettagli strutturati dell'analisi rule-based (già calcolati in
    # predict): esposti per il display senza duplicare la logica.
    anomalie: List[Dict[str, Any]] = field(default_factory=list)
    pattern: List[str] = field(default_factory=list)

    @property
    def metadati(self) -> Dict[str, Any]:
        return {
            "modello_usato": self.modello_usato,
            "classe_mlp": self.classe_mlp,
            "classe_regola": self.classe_regola,
            "confidenza": self.confidenza,
            "fallback": self.fallback,
            "motivo_fallback": self.motivo_fallback,
            "override_sicurezza": self.override_sicurezza,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {**asdict(self), "metadati": self.metadati}

    # English aliases make the metadata convenient for integrations without
    # changing the established Italian public fields.
    @property
    def model_used(self) -> bool:
        return self.modello_usato

    @property
    def mlp_class(self) -> Optional[str]:
        return self.classe_mlp

    @property
    def rule_based_class(self) -> Optional[str]:
        return self.classe_regola

    @property
    def confidence(self) -> Optional[float]:
        return self.confidenza

    @property
    def fallback_reason(self) -> Optional[str]:
        return self.motivo_fallback


_MESSAGGI_CLASSE: Dict[str, str] = {
    "basso": "Parametri vitali nella norma: proseguire il monitoraggio regolare.",
    "medio": "Rilevate alcune anomalie: contattare il medico e ripetere la misurazione.",
    "alto": "Parametri critici rilevati: richiesta valutazione medica immediata.",
}


class SupervisedAgent:
    """Classificatore con precedenza assoluta del safety gate."""

    def __init__(
        self,
        modello_path: Optional[Path] = None,
        modello: Optional[MLP] = None,
        loader: Callable[[Optional[Path]], Optional[MLP]] = carica_modello,
    ) -> None:
        self.modello_path = (
            Path(modello_path) if modello_path is not None else MODEL_ARTIFACT_PATH
        )
        self._modello = modello
        self._loader = loader

    def _carica(self) -> tuple[Optional[MLP], Optional[str]]:
        if self._modello is not None:
            modello = self._modello
        else:
            if not self.modello_path.is_file():
                return None, f"modello non trovato in {self.modello_path}"
            try:
                modello = self._loader(self.modello_path)
            except Exception:
                modello = None
            if modello is None:
                return None, f"modello non valido in {self.modello_path}"

        try:
            if tuple(modello.classi) != CLASSI:
                return None, "artifact con classi non valide"
            if int(modello.n_input) != len(FEATURE_ORDER) or int(modello.n_classi) != len(CLASSI):
                return None, "artifact con dimensione di input/classi non valida"
            if modello.scaler is None:
                return None, "scaler mancante nell'artifact"
            media = np.asarray(modello.scaler.media, dtype=float)
            dev = np.asarray(modello.scaler.dev, dtype=float)
            if media.shape != (len(FEATURE_ORDER),) or dev.shape != (len(FEATURE_ORDER),):
                return None, "scaler con dimensione non valida"
            if not np.isfinite(media).all() or not np.isfinite(dev).all() or (dev == 0).any():
                return None, "scaler non finito o non valido"
            for nome in ("W1", "b1", "W2", "b2"):
                valori = np.asarray(getattr(modello, nome), dtype=float)
                if not np.isfinite(valori).all():
                    return None, "pesi del modello non finiti"
            if np.asarray(modello.W1).shape != (int(modello.n_hidden), len(FEATURE_ORDER)):
                return None, "pesi con dimensione non valida"
            if np.asarray(modello.b1).shape != (int(modello.n_hidden),):
                return None, "bias con dimensione non valida"
            if np.asarray(modello.W2).shape != (len(CLASSI), int(modello.n_hidden)):
                return None, "pesi con dimensione non valida"
            if np.asarray(modello.b2).shape != (len(CLASSI),):
                return None, "bias con dimensione non valida"
        except Exception:
            return None, "artifact non valido"
        return modello, None

    def _base(self, dettagli: Dict[str, Any], errore: bool = False) -> AgentOutcome:
        classe = CLASSE_ERRORE if errore else dettagli["classe"]
        if errore:
            messaggio = "Parametri non validi: " + "; ".join(dettagli["errori"])
        else:
            messaggio = _MESSAGGI_CLASSE.get(classe, "")
            if dettagli["pattern"]:
                messaggio += " Pattern: " + "; ".join(dettagli["pattern"])
        return AgentOutcome(
            classe=classe,
            errore=errore,
            messaggio=messaggio,
            classe_regola=classe,
            fallback=True,
            motivo_fallback=(
                "safety gate: input non valido"
                if errore
                else "safety gate: classe critica, MLP bypassato"
            ),
            anomalie=list(dettagli["anomalie"]),
            pattern=list(dettagli["pattern"]),
        )

    def predict(self, parametri: Any) -> AgentOutcome:
        dettagli = analizza(parametri)
        if not dettagli["valido"]:
            return self._base(dettagli, errore=True)

        base = self._base(dettagli)
        # Critical and pattern-dangerous outcomes are returned byte-for-byte
        # at the established fields; only additive metadata is populated.
        if dettagli["classe"] == "alto":
            return base

        modello, motivo = self._carica()
        if modello is None:
            base.fallback = True
            base.motivo_fallback = motivo
            return base
        try:
            if isinstance(parametri, Mapping):
                valori = [parametri[nome] for nome in FEATURE_ORDER]
            else:
                valori = [getattr(parametri, nome) for nome in FEATURE_ORDER]
            vettore = np.asarray([valori], dtype=float)
            scaler = modello.scaler
            if scaler is None:
                raise ValueError("scaler mancante nell'artifact")
            normalizzato = scaler.transform(vettore)
            probabilita_array = np.asarray(modello.probabilità(normalizzato), dtype=float)
            if probabilita_array.shape != (1, len(CLASSI)):
                raise ValueError("probabilità con dimensione non valida")
            probabilita_array = probabilita_array[0]
            if not np.isfinite(probabilita_array).all() or (probabilita_array < 0).any():
                raise ValueError("probabilità non finite")
            if not np.isclose(probabilita_array.sum(), 1.0, rtol=1e-6, atol=1e-8):
                raise ValueError("probabilità non normalizzate")
        except Exception as exc:
            base.fallback = True
            base.motivo_fallback = f"inferenza MLP non valida: {exc}"
            return base

        probabilita = {
            classe: float(valore)
            for classe, valore in zip(CLASSI, probabilita_array)
        }
        indice_mlp = int(np.argmax(probabilita_array))
        classe_mlp = CLASSI[indice_mlp]
        confidenza = float(probabilita_array[indice_mlp])
        base.modello_usato = True
        base.classe_mlp = classe_mlp
        base.classe_regola = base.classe
        base.confidenza = confidenza
        base.probabilita = probabilita
        base.override_sicurezza = (
            CLASSI.index(base.classe) > indice_mlp
        )
        if confidenza < SOGLIA_INCERTEZZA:
            base.fallback = True
            base.motivo_fallback = "confidenza sotto la soglia di incertezza"
            return base

        classe_finale = CLASSI[max(CLASSI.index(base.classe), indice_mlp)]
        messaggio = _MESSAGGI_CLASSE[classe_finale]
        if dettagli["pattern"]:
            messaggio += " Pattern: " + "; ".join(dettagli["pattern"])
        return AgentOutcome(
            classe=classe_finale,
            probabilita=probabilita,
            errore=False,
            messaggio=messaggio,
            modello_usato=True,
            classe_mlp=classe_mlp,
            classe_regola=base.classe,
            confidenza=confidenza,
            override_sicurezza=base.override_sicurezza,
            anomalie=list(dettagli["anomalie"]),
            pattern=list(dettagli["pattern"]),
        )

    def analizza_parametri(self, parametri: Any) -> Dict[str, Any]:
        """Compatibility API returning the complete serializable outcome."""
        return self.predict(parametri).to_dict()
