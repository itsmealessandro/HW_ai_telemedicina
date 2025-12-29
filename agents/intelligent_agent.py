
from typing import Dict, List
from models.vital_parameters import VitalParameters
import os
import joblib


class IntelligentAgent:
    """
    Agente intelligente basato su Supervised Learning per il monitoraggio
    dei parametri vitali in un sistema di telemedicina.

    L'agente utilizza:
    - Un modello di machine learning supervisionato (se disponibile)
    - Regole di sicurezza simboliche (fallback e override)
    """

    MODEL_PATH = "models/risk_classifier.pkl"

    def __init__(self):
        """Inizializza l'agente caricando il modello supervisionato."""
        self.model = self._load_model()

    # ============================================================
    # API PUBBLICA (immutabile, usata da main.py)
    # ============================================================

    def analizza_parametri(self, parametri: VitalParameters) -> Dict:
        """
        Analizza i parametri vitali e restituisce una valutazione clinica.

        Returns:
            Dict conforme all'interfaccia attesa dal main.py
        """

        # 1. Validazione
        valido, errori = parametri.valida_parametri()
        if not valido:
            return self._errore_output(errori)

        # 2. Estrazione feature
        features = self._extract_features(parametri)

        # 3. Predizione ML (o fallback)
        rischio_predetto = self._predict_risk(features)

        # 4. Safety override (regole cliniche critiche)
        rischio_finale = self._safety_override(parametri, rischio_predetto)

        # 5. Identificazione anomalie (per spiegabilità)
        anomalie = parametri.identifica_anomalie()

        # 6. Raccomandazioni
        raccomandazioni = self._genera_raccomandazioni(
            rischio_finale, anomalie
        )

        # 7. Allerta medico
        allerta_medico = rischio_finale == "alto"

        return {
            "livello_rischio": rischio_finale,
            "anomalie": self._formatta_anomalie(anomalie),
            "raccomandazioni": raccomandazioni,
            "allerta_medico": allerta_medico,
            "dettagli_analisi": {
                "modello_usato": "supervised_ml" if self.model else "rule_based_fallback",
                "numero_anomalie": len(anomalie),
            }
        }

    # ============================================================
    # MACHINE LEARNING
    # ============================================================

    def _load_model(self):
        """Carica il modello supervisionato se presente."""
        if os.path.exists(self.MODEL_PATH):
            try:
                return joblib.load(self.MODEL_PATH)
            except Exception as e:
                print(f"⚠️ Errore caricamento modello ML: {e}")
        return None

    def _extract_features(self, parametri: VitalParameters) -> List[float]:
        """Trasforma i parametri vitali in feature numeriche."""
        return [
            parametri.pressione_sistolica,
            parametri.pressione_diastolica,
            parametri.frequenza_cardiaca,
            parametri.temperatura,
            parametri.saturazione_ossigeno,
            parametri.glicemia
        ]

    def _predict_risk(self, features: List[float]) -> str:
        """
        Predice il livello di rischio tramite ML.
        Fallback a regole statiche se il modello non è disponibile.
        """
        if self.model:
            return self.model.predict([features])[0]
        return self._rule_based_fallback(features)

    # ============================================================
    # REGOLE DI SICUREZZA (OVERRIDE)
    # ============================================================

    def _safety_override(self, parametri: VitalParameters, rischio_ml: str) -> str:
        """
        Regole cliniche critiche che hanno priorità sul modello ML.
        """

        if parametri.saturazione_ossigeno < 88:
            return "alto"

        if parametri.pressione_sistolica >= 180:
            return "alto"

        if parametri.temperatura >= 39.5:
            return "alto"

        if parametri.glicemia < 60:
            return "alto"

        return rischio_ml

    # ============================================================
    # FALLBACK RULE-BASED (semplice)
    # ============================================================

    def _rule_based_fallback(self, features: List[float]) -> str:
        """
        Valutazione deterministica usata solo se ML non disponibile.
        """
        ps, pd, fc, t, spo2, g = features

        if spo2 < 90 or ps > 170 or t > 39:
            return "alto"

        if fc > 100 or g < 70 or g > 180:
            return "medio"

        return "basso"

    # ============================================================
    # RACCOMANDAZIONI
    # ============================================================

    def _genera_raccomandazioni(self, rischio: str, anomalie: List[Dict]) -> str:
        if rischio == "alto":
            return (
                "⚠️ Parametri critici rilevati. "
                "Recarsi immediatamente al pronto soccorso o contattare il medico."
            )

        if rischio == "medio":
            return (
                "📌 Parametri alterati. "
                "Contattare il medico entro 24 ore e ripetere le misurazioni."
            )

        return (
            "✅ Parametri nella norma. "
            "Continuare il monitoraggio regolare e mantenere uno stile di vita sano."
        )

    # ============================================================
    # UTILITÀ
    # ============================================================

    def _formatta_anomalie(self, anomalie: List[Dict]) -> List[str]:
        descrizioni = []
        for a in anomalie:
            min_n, max_n = a["range_normale"]
            direzione = "alta" if a["valore"] > max_n else "bassa"
            descrizioni.append(
                f"{a['parametro']}: {a['valore']} {a['unita']} "
                f"({direzione}, normale {min_n}-{max_n}) "
                f"[{a['gravita'].upper()}]"
            )
        return descrizioni

    def _errore_output(self, errori: List[str]) -> Dict:
        return {
            "livello_rischio": "errore",
            "anomalie": errori,
            "raccomandazioni": "Parametri non validi. Verificare le misurazioni.",
            "allerta_medico": True,
            "dettagli_analisi": {"errori": errori}
        }
