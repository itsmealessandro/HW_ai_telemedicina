"""
training/teacher_rules.py — Teacher OFFLINE per le label del dataset.

Il teacher è la funzione deterministica che, in Fase 2, assegnerà la
label `basso`/`medio`/`alto` a ogni combinazione di parametri vitali del
dataset sintetico. L'MLP (fasi 3-5) impara ad approssimare QUESTA funzione
(knowledge distillation del rule-based).

VINCOLO ARCHITETTURALE: il teacher NON possiede soglie proprie. Tutta la
logica e tutte le soglie vivono esclusivamente in
`telemedicina_supervised.safety.safety_rules`; qui c'è solo il riuso della
stessa API usata dal validatore e dal safety gate runtime. In questo modo
label (offline) e safety gate (runtime) non possono divergere.

Nessun import dal legacy (archive/): il legacy è stato usato solo come
riferimento documentale per fissare le soglie in safety_rules.py.
"""

from typing import Any, Optional

from telemedicina_supervised.safety.safety_rules import classifica_rischio


class TeacherRules:
    """
    Teacher rule-based offline: produce la label per una combinazione di
    parametri vitali, con la STESSA semantica di
    `safety_rules.classifica_rischio` (di cui è un semplice riuso).
    """

    def __init__(self) -> None:
        # Nessuno stato: il teacher è una funzione pura del rule-based.
        pass

    def label(self, parametri: Any) -> str:
        """
        Label di rischio per `parametri`.

        Args:
            parametri: mapping (dict) o oggetto con i 6 attributi ufficiali
                       (es. VitalParameters).

        Returns:
            'basso' | 'medio' | 'alto' | 'errore'.
            'errore' per input fisiologicamente invalidi.
        """
        return classifica_rischio(parametri)

    # Alias esplicito per uso in generazione dataset.
    etichetta = label


# Istanza di comodo condivisa (nessuno stato interno, thread-safe per
# costruzione).
_TEACHER: Optional[TeacherRules] = None


def label(parametri: Any) -> str:
    """
    Funzione di modulo (comoda per la Fase 2): label rule-based di
    `parametri`, identica a `TeacherRules().label`.
    """
    global _TEACHER
    if _TEACHER is None:
        _TEACHER = TeacherRules()
    return _TEACHER.label(parametri)
