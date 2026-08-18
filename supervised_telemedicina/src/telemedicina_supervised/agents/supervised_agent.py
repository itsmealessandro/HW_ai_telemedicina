"""
telemedicina_supervised/agents/supervised_agent.py — Agente con il
CONTRATTO DEFINITIVO del nuovo progetto supervised.

In questa fase (Fase 1) l'agente è la BASELINE rule-based: la predizione
coincide con il safety gate deterministico. In Fase 5 l'MLP verrà inserito
DIETRO questo gate:
  - input invalidi -> 'errore' (mai una classe di rischio);
  - casi critici riconosciuti dalle regole -> decisione deterministica
    'alto' (il gate NON è opzionale e NON può essere declassato);
  - solo i casi non critici potranno passare all'MLP per la stima di
    probabilità (con eventuale delega al rule-based in zona di incertezza).

Nessuna soglia numerica è definita qui: tutta la logica e le soglie
vivono in `telemedicina_supervised.safety.safety_rules`.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from telemedicina_supervised.safety.safety_rules import CLASSE_ERRORE, analizza


@dataclass
class AgentOutcome:
    """
    Output del SupervisedAgent (contratto).

    Attributi:
        classe:      'basso' | 'medio' | 'alto' | 'errore'.
        probabilita: dict {'basso': p, 'medio': p, 'alto': p} per il
                     modello probabilistico (Fase 5), oppure None nella
                     baseline rule-based (Fase 1).
        errore:      True se l'input era fisiologicamente invalido.
        messaggio:   spiegazione leggibile della decisione.
    """

    classe: str
    probabilita: Optional[Dict[str, float]] = None
    errore: bool = False
    messaggio: str = ""


_MESSAGGI_CLASSE: Dict[str, str] = {
    "basso": "Parametri vitali nella norma: proseguire il monitoraggio regolare.",
    "medio": "Rilevate alcune anomalie: contattare il medico e ripetere la misurazione.",
    "alto": "Parametri critici rilevati: richiesta valutazione medica immediata.",
}


class SupervisedAgent:
    """
    Agente di classificazione del rischio.

    Contratto del metodo `predict`:
      - input valido e non critico  -> classe di rischio, errore=False;
      - input valido e critico      -> 'alto' (mai declassato);
      - input fisiologicamente invalido -> classe='errore', errore=True,
        probabilita=None (nessuna predizione di classe).
    """

    def predict(self, parametri: Any) -> AgentOutcome:
        """
        Predice il livello di rischio dei parametri vitali.

        Args:
            parametri: mapping (dict) o oggetto con i 6 attributi
                       ufficiali (es. VitalParameters).

        Returns:
            AgentOutcome secondo il contratto.
        """
        # 1. Analisi completa rule-based in UNA sola chiamata alla fonte
        #    unica (safety_rules.analizza): validazione + classificazione.
        #    Input invalido => 'errore', mai una classe di rischio.
        dettagli = analizza(parametri)
        if not dettagli["valido"]:
            return AgentOutcome(
                classe=CLASSE_ERRORE,
                probabilita=None,
                errore=True,
                messaggio="Parametri non validi: " + "; ".join(dettagli["errori"]),
            )

        # 2. Safety gate rule-based (baseline Fase 1). In Fase 5 l'MLP
        #    opererà solo dietro questo gate, che NON è opzionale e ha
        #    precedenza assoluta.
        classe = dettagli["classe"]

        messaggio = _MESSAGGI_CLASSE.get(classe, "")
        if dettagli["pattern"]:
            messaggio += " Pattern: " + "; ".join(dettagli["pattern"])

        # 3. Baseline rule-based: nessuna probabilità softmax (None).
        #    In Fase 5 l'MLP popolerà `probabilita`.
        return AgentOutcome(
            classe=classe,
            probabilita=None,
            errore=False,
            messaggio=messaggio,
        )
