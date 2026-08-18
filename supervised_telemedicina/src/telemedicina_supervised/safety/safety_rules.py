"""
telemedicina_supervised/safety/safety_rules.py — FONTE UNICA DI VERITÀ delle soglie.

Questo modulo è l'unico punto del progetto in cui vivono:
  - i range di validità fisiologica (valori "possibili", non normali);
  - i range normali di ciascun parametro;
  - le soglie critiche e moderate di ciascun parametro;
  - i pattern di correlazione pericolosi;
  - la funzione di classificazione rule-based `classifica_rischio`.

Validatore (VitalParameters), teacher offline (training/teacher_rules.py),
safety gate (SupervisedAgent) e servizio (AnalysisService) dipendono TUTTI
da questo modulo: nessuna soglia numerica è duplicata altrove nel runtime.

----------------------------------------------------------------------
AUDIT STORICO (sola consultazione, nessun import dal legacy)
----------------------------------------------------------------------
Valori estratti da archive/legacy_qtable_rule_based (letti solo come
riferimento documentale):

  RANGE_NORMALI  (vital_parameters.py):
    pressione_sistolica   (90, 140)
    pressione_diastolica  (60,  90)
    frequenza_cardiaca    (60, 100)
    temperatura           (36.0, 37.5)
    saturazione_ossigeno  (95, 100)
    glicemia              (70, 140)

  SOGLIE_CRITICHE (vital_parameters.py):
    sistolica >= 180 (alta), diastolica >= 110 (alta),
    FC >= 120 (alta) o <= 50 (bassa), temp >= 38.5 (alta) o <= 35.0 (bassa),
    SpO2 < 90 (bassa), glicemia >= 200 (alta) o <= 60 (bassa)

  SOGLIE moderate (metodo _valuta_gravita_*):
    sistolica >= 160 o < 80; diastolica >= 100 o < 50;
    FC >= 110 o <= 55; temp >= 38.0 o <= 35.5; SpO2 < 93; glicemia >= 180 o <= 65

  Pattern critici (intelligent_agent._analizza_correlazioni):
    shock, crisi ipertensiva, insufficienza respiratoria, sepsi,
    ipoglicemia severa, ipertermia critica, ipossia severa.

Le soglie qui sotto sono REIMPLEMENTAZIONI documentate di quei valori,
adottate come unico riferimento del nuovo progetto.

----------------------------------------------------------------------
DISCREPANZA LEGACY RISOLTA — GLICEMIA CRITICA
----------------------------------------------------------------------
Il piano segnala una divergenza nel legacy sulla soglia di ipoglicemia
critica:
  - teacher rule-based (intelligent_agent.py): critica per glicemia <= 60
    (`_valuta_gravita_glicemia` usa `valore <= 60`; il pattern
    "ipoglicemia severa" usa `glicemia < 60`);
  - override RL (rl_agent.py): `crit.glicemia <= 55` → safety override
    "glicemia_critica".

SCELTA UNIFICATA ADOTTATA: **soglia critica bassa = 60 mg/dL**
(condizione: `glicemia <= 60`).

Motivazione:
  1. Il teacher rule-based è la fonte delle label per la knowledge
     distillation: la soglia del modello studente deve coincidere con
     quella del teacher, altrimenti l'MLP apprenderebbe un bersaglio
     incoerente con il safety gate (label contraddittorie).
  2. Tra le due soglie (60 vs 55), 60 è la più conservativa dal punto di
     vista della sicurezza: individua un paziente critico prima.
  3. Il valore <= 55 dell'override RL era un'accorciatoia specifica del
     solo agente Q-learning, non usata per le label; viene scartato.
La soglia alta unificata resta 200 mg/dL (coerente col teacher; l'override
RL usava 250, scartato per coerenza di label).

----------------------------------------------------------------------
PRECEDENZA DELLE REGOLE DI SICUREZZA
----------------------------------------------------------------------
In questo progetto le regole deterministiche (safety gate) hanno
PRECEDENZA ASSOLUTA sul modello: un caso che le regole giudicano critico
non può MAI essere declassato. `classifica_rischio` implementa proprio
questo livello: in Fase 5 l'MLP opererà solo DOPO questo gate.
"""

from typing import Any, Dict, List, Mapping, Tuple

# Ordine ufficiale delle 6 feature (contratto): NON modificare senza
# aggiornare docs/contratto.md.
FEATURE_ORDER: List[str] = [
    "pressione_sistolica",
    "pressione_diastolica",
    "frequenza_cardiaca",
    "temperatura",
    "saturazione_ossigeno",
    "glicemia",
]

# Classi ufficiali di rischio (contratto).
CLASSI: Tuple[str, ...] = ("basso", "medio", "alto")
# Risposta per input fisiologicamente invalidi: mai una classe di rischio.
CLASSE_ERRORE: str = "errore"

# ----------------------------------------------------------------------
# Range di validità fisiologica: valori "possibili" per un essere umano.
# Fuori da questi range l'input è da considerare una misurazione invalida
# (risposta 'errore'), NON un rischio basso/medio/alto.
# ----------------------------------------------------------------------
RANGE_FISIOLOGICI: Dict[str, Tuple[float, float]] = {
    "pressione_sistolica": (50.0, 250.0),    # mmHg
    "pressione_diastolica": (30.0, 150.0),   # mmHg
    "frequenza_cardiaca": (30.0, 200.0),     # bpm
    "temperatura": (34.0, 42.0),             # °C
    "saturazione_ossigeno": (70.0, 100.0),   # %
    "glicemia": (20.0, 500.0),               # mg/dL
}

# ----------------------------------------------------------------------
# Range normali: valori fisiologicamente "nella norma".
# ----------------------------------------------------------------------
RANGE_NORMALI: Dict[str, Tuple[float, float]] = {
    "pressione_sistolica": (90.0, 140.0),    # mmHg
    "pressione_diastolica": (60.0, 90.0),    # mmHg
    "frequenza_cardiaca": (60.0, 100.0),     # bpm
    "temperatura": (36.0, 37.5),             # °C
    "saturazione_ossigeno": (95.0, 100.0),   # %
    "glicemia": (70.0, 140.0),               # mg/dL
}

# ----------------------------------------------------------------------
# Soglie critiche e moderate per ciascun parametro.
# Struttura: {parametro: {'alta': soglia_superiore|None,
#                         'bassa': soglia_inferiore|None}}
# 'alta' = valore >= soglia → anomalia verso l'alto.
# 'bassa' = valore <= soglia → anomalia verso il basso.
# ----------------------------------------------------------------------
SOGLIE_CRITICHE: Dict[str, Dict[str, Any]] = {
    "pressione_sistolica": {"alta": 180.0, "bassa": None},
    "pressione_diastolica": {"alta": 110.0, "bassa": None},
    "frequenza_cardiaca": {"alta": 120.0, "bassa": 50.0},
    "temperatura": {"alta": 38.5, "bassa": 35.0},
    "saturazione_ossigeno": {"alta": None, "bassa": 90.0},
    # Discrepanza legacy risolta: soglia critica bassa unificata a 60
    # (teacher <= 60; override RL <= 55 scartato — vedi docstring modulo).
    "glicemia": {"alta": 200.0, "bassa": 60.0},
}

SOGLIE_MODERATE: Dict[str, Dict[str, Any]] = {
    "pressione_sistolica": {"alta": 160.0, "bassa": 80.0},
    "pressione_diastolica": {"alta": 100.0, "bassa": 50.0},
    "frequenza_cardiaca": {"alta": 110.0, "bassa": 55.0},
    "temperatura": {"alta": 38.0, "bassa": 35.5},
    "saturazione_ossigeno": {"alta": None, "bassa": 93.0},
    "glicemia": {"alta": 180.0, "bassa": 65.0},
}

# ----------------------------------------------------------------------
# Soglie usate SOLO nei pattern di correlazione (non classificano un
# singolo parametro, ma una combinazione pericolosa).
# ----------------------------------------------------------------------
# SpO2 sotto cui, combinata con tachicardia (>FC normale max),
# si configura insufficienza respiratoria.
PATTERN_IPOSSIA_SPO2_MAX: float = 92.0  # %
# Temperatura >= soglia → ipertermia critica (già coperta da SOGLIE_CRITICHE,
# mantenuta per fedeltà documentale al legacy).
PATTERN_IPERTERMIA_CRITICA_MIN: float = 39.5  # °C
# SpO2 sotto cui si configura ipossia severa (già coperta da
# SOGLIE_CRITICHE, mantenuta per fedeltà documentale al legacy).
PATTERN_IPOSSIA_SEVERA_MAX: float = 88.0  # %


def _get(parametri: Any, nome: str) -> Any:
    """Estrae un parametro da un mapping o da un oggetto con attributi."""
    if isinstance(parametri, Mapping):
        return parametri.get(nome)
    return getattr(parametri, nome, None)


def _valori_numerici(*valori: Any) -> bool:
    """True se tutti i valori sono numeri reali (esclusi i bool)."""
    return all(
        isinstance(v, (int, float)) and not isinstance(v, bool) for v in valori
    )


def valida_parametri(parametri: Any) -> Tuple[bool, List[str]]:
    """
    Valida che i parametri siano fisiologicamente plausibili.

    Non verifica se i valori sono "normali", solo se sono possibili:
      - ogni valore dentro il proprio RANGE_FISIOLOGICI;
      - pressione sistolica strettamente maggiore della diastolica
        (vincolo strutturale, presente anche nel legacy).

    Args:
        parametri: mapping (dict) o oggetto con i 6 attributi ufficiali.

    Returns:
        (True, []) se validi, altrimenti (False, lista dei motivi).
    """
    errori: List[str] = []

    for nome in FEATURE_ORDER:
        valore = _get(parametri, nome)
        if not _valori_numerici(valore):
            errori.append(f"{nome}: valore non numerico ({valore!r})")
            continue
        min_v, max_v = RANGE_FISIOLOGICI[nome]
        if not (min_v <= valore <= max_v):
            errori.append(
                f"{nome}: fuori range fisiologico ({min_v}-{max_v})"
            )

    sistolica = _get(parametri, "pressione_sistolica")
    diastolica = _get(parametri, "pressione_diastolica")
    if _valori_numerici(sistolica, diastolica):
        if sistolica <= diastolica:
            errori.append(
                "pressione_sistolica deve essere maggiore di pressione_diastolica"
            )

    return len(errori) == 0, errori


def _severita(nome: str, valore: float) -> str:
    """
    Gravità di un singolo parametro fuori dal range normale:
    'critica', 'moderata' o 'lieve'. Fedele ai if/elif del legacy
    (_valuta_gravita_*): prima le soglie critiche, poi le moderate.
    """
    critiche = SOGLIE_CRITICHE[nome]
    moderate = SOGLIE_MODERATE[nome]
    if critiche["alta"] is not None and valore >= critiche["alta"]:
        return "critica"
    if critiche["bassa"] is not None and valore <= critiche["bassa"]:
        return "critica"
    if moderate["alta"] is not None and valore >= moderate["alta"]:
        return "moderata"
    if moderate["bassa"] is not None and valore <= moderate["bassa"]:
        return "moderata"
    return "lieve"


def _anomalie(parametri: Any) -> List[Dict[str, Any]]:
    """Lista dei parametri fuori dal range normale, con gravità e direzione."""
    anomalie: List[Dict[str, Any]] = []
    for nome in FEATURE_ORDER:
        valore = _get(parametri, nome)
        if not _valori_numerici(valore):
            continue  # la validità è già gestita da valida_parametri
        min_n, max_n = RANGE_NORMALI[nome]
        if min_n <= valore <= max_n:
            continue
        direzione = "elevata" if valore > max_n else "bassa"
        anomalie.append(
            {
                "parametro": nome,
                "valore": valore,
                "range_normale": (min_n, max_n),
                "gravita": _severita(nome, valore),
                "direzione": direzione,
            }
        )
    return anomalie


def _pattern_critici(parametri: Any) -> List[str]:
    """
    Pattern di correlazione pericolosi (fedeli al legacy
    intelligent_agent._analizza_correlazioni), con soglie derivate dalle
    costanti di questo modulo (nessun numero duplicato).
    """
    sistolica = _get(parametri, "pressione_sistolica")
    diastolica = _get(parametri, "pressione_diastolica")
    frequenza = _get(parametri, "frequenza_cardiaca")
    temperatura = _get(parametri, "temperatura")
    saturazione = _get(parametri, "saturazione_ossigeno")
    glicemia = _get(parametri, "glicemia")

    if not _valori_numerici(
        sistolica, diastolica, frequenza, temperatura, saturazione, glicemia
    ):
        return []  # input invalido: gestito da valida_parametri

    sistolica_min = RANGE_NORMALI["pressione_sistolica"][0]  # 90
    frequenza_max = RANGE_NORMALI["frequenza_cardiaca"][1]   # 100
    sistolica_crit = SOGLIE_CRITICHE["pressione_sistolica"]["alta"]   # 180
    diastolica_crit = SOGLIE_CRITICHE["pressione_diastolica"]["alta"]  # 110
    temperatura_crit = SOGLIE_CRITICHE["temperatura"]["alta"]          # 38.5
    glicemia_crit_bassa = SOGLIE_CRITICHE["glicemia"]["bassa"]          # 60

    pattern: List[str] = []
    if sistolica < sistolica_min and frequenza > frequenza_max:
        pattern.append("SHOCK: ipotensione con tachicardia compensatoria")
    if sistolica >= sistolica_crit or diastolica >= diastolica_crit:
        pattern.append("CRISI IPERTENSIVA: pressione pericolosamente elevata")
    if saturazione < PATTERN_IPOSSIA_SPO2_MAX and frequenza > frequenza_max:
        pattern.append("INSUFFICIENZA RESPIRATORIA: ipossia con tachicardia")
    if temperatura >= temperatura_crit and frequenza > frequenza_max:
        pattern.append("INFEZIONE SISTEMICA: febbre elevata con risposta cardiaca")
    if glicemia <= glicemia_crit_bassa and frequenza > frequenza_max:
        pattern.append("IPOGLICEMIA SEVERA: glicemia critica con tachicardia")
    if temperatura >= PATTERN_IPERTERMIA_CRITICA_MIN:
        pattern.append("IPERTERMIA CRITICA: temperatura pericolosamente elevata")
    if saturazione < PATTERN_IPOSSIA_SEVERA_MAX:
        pattern.append("IPOSSIA SEVERA: saturazione ossigeno critica")
    return pattern


def analizza(parametri: Any) -> Dict[str, Any]:
    """
    Analisi completa rule-based.

    Returns:
        dict con chiavi:
          - 'valido': bool;
          - 'errori': lista motivi (vuota se valido);
          - 'anomalie': parametri fuori range con gravità;
          - 'pattern': pattern critici rilevati;
          - 'classe': 'basso'/'medio'/'alto' oppure 'errore'.
    """
    valido, errori = valida_parametri(parametri)
    if not valido:
        return {
            "valido": False,
            "errori": errori,
            "anomalie": [],
            "pattern": [],
            "classe": CLASSE_ERRORE,
        }

    anomalie = _anomalie(parametri)
    pattern = _pattern_critici(parametri)

    # Precedenza delle regole di sicurezza: critico o pattern => 'alto'.
    if pattern or any(a["gravita"] == "critica" for a in anomalie):
        classe = "alto"
    elif anomalie:  # qualunque anomalia non critica => 'medio' (fedele al legacy)
        classe = "medio"
    else:
        classe = "basso"

    return {
        "valido": True,
        "errori": [],
        "anomalie": anomalie,
        "pattern": pattern,
        "classe": classe,
    }


def classifica_rischio(parametri: Any) -> str:
    """
    Classifica il rischio con logica rule-based.

    Returns:
        'basso' | 'medio' | 'alto' | 'errore'.
        'errore' per input fisiologicamente invalidi: mai una classe
        di rischio in quel caso.
    """
    return analizza(parametri)["classe"]
