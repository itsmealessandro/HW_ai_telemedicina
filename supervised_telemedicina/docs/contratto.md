# Contratto funzionale e baseline — supervised_telemedicina

**Fase:** 1 — Contratto funzionale e baseline del nuovo sistema
**Validazione:** orchestratore
**Stato:** in revisione

Questo documento fissa il comportamento del nuovo progetto **prima** di
introdurre apprendimento. Il punto di riferimento architetturale è la
fonte unica delle soglie:
`src/telemedicina_supervised/safety/safety_rules.py`.

> **Obiettivo metodologico dichiarato (Fase 1, punto 5 del contratto):**
> il progetto implementa una **knowledge distillation del rule-based**,
> NON una validità clinica. I dati saranno sintetici e le label deriveranno
> dal teacher rule-based; i risultati misureranno la fedeltà al teacher,
> non l'accuratezza diagnostica. Nessuna pretesa clinica.

---

## 1. Contratto

### 1.1 Ordine ufficiale delle 6 feature

```text
[pressione_sistolica, pressione_diastolica, frequenza_cardiaca,
 temperatura, saturazione_ossigeno, glicemia]
```

Questo è l'ordine del vettore `VitalParameters.to_vector()`, dell'input MLP
(Fase 3+) e di `FEATURE_ORDER` in `safety_rules.py`. Non modificarlo senza
aggiornare questo documento.

### 1.2 Classi ufficiali

```text
'basso' (0), 'medio' (1), 'alto' (2)
```

### 1.3 Valori fisiologicamente invalidi → 'errore'

Un input con almeno un valore fuori dal range fisiologico (o con
`sistolica <= diastolica`) produce la risposta **'errore'**: nessuna
predizione di classe di rischio, allerta medico. Mai una classe
`basso`/`medio`/`alto` per input invalidi.

### 1.4 Precedenza assoluta delle regole di sicurezza (safety gate)

Le regole deterministiche hanno **precedenza assoluta** sui risultati del
modello: un caso che le regole giudicano critico **non può mai essere
declassato**. Il gate non è opzionale e in Fase 5 l'MLP opererà solo dietro
di esso.

### 1.5 Obiettivo metodologico

Knowledge distillation del rule-based, dati sintetici, nessuna validità
clinica (vedi nota in testata).

---

## 2. Soglie unificate (fonte unica: `safety_rules.py`)

### 2.1 Range di validità fisiologica (input invalidi → 'errore')

| Parametro | Range fisiologico |
|---|---|
| pressione_sistolica | 50–250 mmHg |
| pressione_diastolica | 30–150 mmHg |
| frequenza_cardiaca | 30–200 bpm |
| temperatura | 34.0–42.0 °C |
| saturazione_ossigeno | 70–100 % |
| glicemia | 20–500 mg/dL |

Vincolo strutturale: `pressione_sistolica > pressione_diastolica`.

### 2.2 Range normali

| Parametro | Range normale |
|---|---|
| pressione_sistolica | 90–140 mmHg |
| pressione_diastolica | 60–90 mmHg |
| frequenza_cardiaca | 60–100 bpm |
| temperatura | 36.0–37.5 °C |
| saturazione_ossigeno | 95–100 % |
| glicemia | 70–140 mg/dL |

### 2.3 Soglie critiche (→ 'alto')

| Parametro | Critica |
|---|---|
| pressione_sistolica | ≥ 180 mmHg |
| pressione_diastolica | ≥ 110 mmHg |
| frequenza_cardiaca | ≥ 120 bpm oppure ≤ 50 bpm |
| temperatura | ≥ 38.5 °C oppure ≤ 35.0 °C |
| saturazione_ossigeno | ≤ 90 % (non critica: > 90 %) |
| glicemia | ≥ 200 mg/dL oppure **≤ 60 mg/dL** |

### 2.4 Soglie moderate (→ 'medio')

| Parametro | Moderata |
|---|---|
| pressione_sistolica | ≥ 160 oppure < 80 mmHg |
| pressione_diastolica | ≥ 100 oppure < 50 mmHg |
| frequenza_cardiaca | ≥ 110 oppure ≤ 55 bpm |
| temperatura | ≥ 38.0 oppure ≤ 35.5 °C |
| saturazione_ossigeno | < 93 % |
| glicemia | ≥ 180 oppure ≤ 65 mg/dL |

### 2.5 Pattern critici (→ 'alto')

| Pattern | Condizione |
|---|---|
| Shock | sistolica < 90 e FC > 100 |
| Crisi ipertensiva | sistolica ≥ 180 o diastolica ≥ 110 |
| Insufficienza respiratoria | SpO2 < 92 e FC > 100 |
| Infezione sistemica/sepsi | temperatura ≥ 38.5 e FC > 100 |
| Ipoglicemia severa | glicemia ≤ 60 e FC > 100 |
| Ipertermia critica | temperatura ≥ 39.5 |
| Ipossia severa | SpO2 < 88 |

---

## 3. Discrepanza legacy risolta: soglia glicemia critica

Il piano ha segnalato una divergenza nel legacy:

| Fonte | Soglia ipoglicemia critica |
|---|---|
| Teacher rule-based (`intelligent_agent.py`) | `glicemia <= 60` (gravità critica) |
| Override RL (`rl_agent.py`) | `glicemia <= 55` |

**Scelta unificata adottata: soglia critica bassa = 60 mg/dL**
(`glicemia <= 60` → critica → 'alto').

**Motivazione:**

1. Il teacher rule-based è la fonte delle label per la knowledge
   distillation: il safety gate runtime e il teacher offline devono
   coincidere, altrimenti l'MLP apprenderebbe un bersaglio incoerente con
   il gate (label contraddittorie).
2. Tra 60 e 55, 60 è la soglia più conservativa: individua il paziente
   critico prima, coerentemente con la precedenza delle regole di
   sicurezza.
3. Il valore `<= 55` apparteneva al solo agente Q-learning (override
   specifico), non alle label; viene scartato.
4. La soglia alta resta 200 mg/dL (coerente col teacher; l'override RL a
   250 viene scartato per la stessa ragione di coerenza label/gate).

---

## 4. Esempio input → output (`classifica_rischio`)

Formato input: oggetto `VitalParameters` o dict con le 6 chiavi ufficiali.

| Input (sist, dias, FC, temp, SpO2, glic) | Classe |
|---|---|
| (120, 80, 75, 36.8, 98, 95) — nella norma | `basso` |
| (85, 60, 72, 36.5, 97, 90) — ipotensione lieve isolata | `medio` |
| (185, 110, 80, 36.8, 98, 95) — crisi ipertensiva | `alto` |
| (85, 55, 115, 36.5, 96, 90) — shock (ipotensione + tachicardia) | `alto` |
| (120, 80, 75, 36.8, 98, 55) — ipoglicemia critica (≤ 60) | `alto` |
| (120, 80, 75, 36.8, 98, 10) — glicemia fuori range fisiologico | `errore` |

Note:
- in Fase 1 (`predict` di `SupervisedAgent`) `probabilita` è `None`
  (baseline rule-based); in Fase 5 sarà un dict `{basso, medio, alto}`;
- la baseline non declassa mai un caso critico ('alto' resta 'alto').

---

## 5. Contratti software (baseline Fase 1)

### `SupervisedAgent.predict(parametri) -> AgentOutcome`

| Campo | Tipo | Note |
|---|---|---|
| `classe` | str | `basso`/`medio`/`alto`/`errore` |
| `probabilita` | dict o None | `None` in Fase 1; dict in Fase 5 |
| `errore` | bool | `True` solo per input invalidi |
| `messaggio` | str | spiegazione leggibile |

### `AnalysisService.analizza(parametri) -> ServiceOutcome`

| Campo | Tipo | Note |
|---|---|---|
| `classe` | str | come sopra |
| `messaggio_paziente` | str | messaggio per il paziente |
| `allerta_medico` | bool | `True` per 'alto' o input invalido |
| `errori` | list[str] | motivi di validazione (vuota se valido) |

In Fase 1 il servizio logga su console e registra in memoria
(`AnalysisService.storico`); il database arriverà nelle fasi successive
senza modificare il contratto.
