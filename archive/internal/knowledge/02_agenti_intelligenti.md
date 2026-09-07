# 02 – Agenti Intelligenti

> Fonte: `2_AI_2526_agents.pdf` (Prof. Fabio Persia, DT0171, A.A. 2025/2026)

## 1. Definizione di agente (pag. 6)

> "An agent is anything that can be viewed as perceiving its environment through sensors and acting upon that environment through actuators."

| Tipo | Sensori | Attuatori |
|---|---|---|
| **Umano** | Occhi, orecchie | Mani, gambe, tratto vocale |
| **Robotico** | Telecamere, lidar | Motori |
| **Software** | Contenuti file, packet di rete, input utente | Scrittura file, invio packet, output audio/visivo |

## 2. Concetti fondamentali (pag. 7)

- **Percept**: contenuto percepito dai sensori.
- **Percept sequence**: storia completa di tutto ciò che l'agente ha percepito.
- **Scelta d'azione**: dipende dalla conoscenza incorporata e dall'intera sequenza di percezioni.

## 3. Task agent-based vs non-agent-based (pag. 11-12)

- **Sequential** (agent-based): l'agente gioca un ruolo in sequenze di decisioni; le decisioni influenzano stati successivi.
- **Episodic** (non-agent-based): ogni episodio è indipendente (es. classificazione immagini).

## 4. Agenti razionali (pag. 14-19)

**Definizione** (pag. 16):
> "Per ogni possibile sequenza di percezioni, un agente razionale dovrebbe selezionare un'azione che si attende di massimizzare la sua performance measure, data l'evidenza della sequenza di percezioni e la conoscenza incorporata."

- **Consequentialism** (pag. 14): il valore dell'agente dipende dalle conseguenze delle azioni.
- **Performance measure** (pag. 14): misura che valuta la desiderabilità della sequenza di stati ambientali.
- **Rationality vs Humanity** (pag. 15): gli umani hanno desideri propri, le macchine no → la performance measure
  è nella mente del progettista/utente.
- **Learning** (pag. 19): l'agente razionale impara il più possibile dalle percezioni; la configurazione iniziale
  riflette conoscenza a priori, modificata/aumentata con l'esperienza.

**Regola progettuale** (pag. 17): progettare la performance measure in base a *ciò che si vuole ottenere*,
non a *come si pensa che l'agente debba comportarsi*.

## 5. Framework PEAS (pag. 21-24)

**P**erformance measure, **E**nvironment, **A**ctuators, **S**ensors – framework per progettare un agente.

Esempio – taxi automatizzato (pag. 22):

| Dimensione | Contenuto |
|---|---|
| Performance | Sicurezza, velocità, comfort, legalità, guadagni |
| Environment | Strade, traffico, pedoni, meteo, clienti |
| Attuatori | Freni, volante, acceleratore, segnalatore, display, speaker |
| Sensori | Telecamere, microfoni, infrarossi, velocimetro, GPS, accelerometro, sonar |

## 6. Tipi di ambiente (pag. 25-31)

| Caratteristica | Estremi | Esempio in telemedicina |
|---|---|---|
| **Osservabilità** | Fully vs Partially observable | Dati vitali parziali (solo wearable) |
| **Agenti** | Single vs Multi-agent | Multi (paziente + medico + sistema) |
| **Determinismo** | Deterministic vs Nondeterministic | Nondeterministic (risposta farmaco imprevedibile) |
| **Temporalità** | Episodic vs Sequential | Sequential (trattamento continuo) |
| **Dinamicità** | Static vs Dynamic | Dynamic (condizione paziente cambia) |
| **Spazio** | Discrete vs Continuous | Continuous (valori vitali continui) |
| **Conoscenza** | Known vs Unknown | Parzialmente noto |

**Caso più difficile** (pag. 31): partially observable, multi-agent, nondeterministic, sequential, dynamic, continuous, unknown.

## 7. Funzione e programma dell'agente (pag. 33-35)

- **Agent function**: descrizione astratta del comportamento; mappa sequenze di percezioni → azioni.
- **Agent program**: implementazione concreta della funzione su hardware fisico.
- **Agent = program + architecture** (pag. 35): l'architettura rende disponibili le percezioni, esegue il programma e alimenta gli attuatori.

## 8. Table-Driven Agent (pag. 36-38)

Approccio più semplice: lookup table che mappa ogni sequenza di percezioni all'azione desiderata.

**Formula dimensione tabella** (pag. 38):
- P = insieme dei possibili percepts, T = vita dell'agente (totale percepts)
- Tabella: **Σ(t=1..T) P^t** entries
- Tabella degli scacchi: ≥ 10^150 entries (atomi nell'universo osservabile < 10^82) → non fattibile.

## 9. Tipi di agente (pag. 40-47)

1. **Simple Reflex Agent** (pag. 41): azioni basate solo sulla percezione corrente; regola condizione-azione
   (es. "if car-in-front-is-braking then initiate-braking").
2. **Model-Based Reflex Agent** (pag. 42): mantiene uno **stato interno** per tracciare aspetti del mondo non evidenti
   nella percezione corrente; agisce su percezione corrente + stato interno.
3. **Goal-Based Agent** (pag. 43): agisce per raggiungere obiettivi; la decisione corretta dipende dal goal;
   search e planning trovano sequenze di azioni per raggiungere i goal.
4. **Utility-Based Agent** (pag. 44-45): massimizza "felicità" attesa → **utility function**. I goal danno distinzione
   binaria (happy/unhappy); la utility permette confronti granulari tra stati.
   - **Utility function vs Performance measure** (pag. 45): la utility è l'**internalizzazione** della performance
     measure; se concordano, l'agente che massimizza l'utility è razionale secondo la misura esterna.
5. **Learning Agent** (pag. 46-47): impara dall'esperienza; inizia con conoscenza base e si adatta automaticamente.
   Qualsiasi tipo precedente può essere implementato come learning agent. **Performance standard** distingue
   reward/penalty dal percept → feedback sulla qualità del comportamento.

## 10. Esempio: Paper Buying Agent (pag. 48-62)

Esempio 2.1 del libro Poole & Mackworth; codice Python in `agentBuying.py`.
- Agente che garantisce la scorta di carta; monitora offerte online e stock in magazzino.
- **Percepts**: prezzo carta + quantità in stock. **Action**: numero di unità da ordinare (0 = non ordinare).
- Architettura software (pag. 53-56): `display.py` (visualizzazione), `agents.py` (classe base agente con
  `select_action(percept) → action`, classe ambiente con `do(action) → percept`), simulatore (catena do ↔ select_action),
  `agentBuying.py` (stato ambiente = tempo, stock; storico prezzi e stock), `utilities.py` (funzioni di utilità).

## 11. Concetti chiave per l'esame

1. Definizione di agente (pag. 6).
2. Definizione di agente razionale (pag. 16) e performance measure.
3. Framework PEAS e applicazione a un esempio (pag. 21-24).
4. Le 7 caratteristiche degli ambienti (pag. 25-31).
5. Agent function vs agent program; Agent = program + architecture (pag. 33-35).
6. Dimensione della tabella di un table-driven agent: Σ(t=1..T) P^t (pag. 38).
7. I 5 tipi di agente, dal simple reflex al learning agent (pag. 40-47).
