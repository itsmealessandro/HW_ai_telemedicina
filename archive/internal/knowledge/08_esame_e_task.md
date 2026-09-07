# 08 – Esame e Task di Progetto

> Fonti: `first_exam_2026.pdf`, `AI_tasks - 2026.pdf`, `0 - Course Introduction.pdf`

## 1. Struttura dell'esame (slide 0)

- **Valutazione**: scritto 50% + Homework Persia 25% + Homework Caianiello 25% (pag. 11/17).
- **Pesi per argomento** (pag. 9/17): Introduzione 10%, Agenti 15%, Rappresentazione conoscenza 15%, Quantificazione
  incertezza 15%, Ragionamento probabilistico 15%, Ragionamento probabilistico nel tempo 15%, RL 15%.
- Testi: Russell & Norvig *AIMA* 4ª ed., Aggarwal, Poole & Mackworth, Aggarwal *NN and DL*, Mitchell (pag. 15/17).

## 2. Esame scritto del 15 gennaio 2026 (first_exam_2026.pdf)

Struttura: 6 domande, punteggio massimo 30+L, minimo 18, nessuna penalità per risposte errate/mancanti.

| # | Domanda | Argomento |
|---|---|---|
| 1 | Struttura del **Turing Test** | AI Fundamentals (slide 1) |
| 2 | Acronimo **PEAS** + 2 esempi di agenti | Agenti intelligenti (slide 2) |
| 3 | Vantaggi/svantaggi **content-based vs collaborative filtering** | Recommender Systems (slide 3) |
| 4 | **Relazioni di intervallo di Allen** (quante, elenco, definizione formale) | Knowledge Representation (slide 5) |
| 5 | **Classificazione vs Regressione** + esempi concreti | Machine Learning (slide 9) |
| 6 | **Heuristica di Rosenblatt** per il perceptron | Reti Neurali (slide 10, pag. 22-24) |

→ Tutte queste domande sono trattate in questa knowledge base (file 01, 02, 03, 05, 06).

## 3. Task di progetto – Telemedicina (AI_tasks - 2026.pdf)

### Task 1: Agente intelligente per telemedicina
> "Design and implement an intelligent agent in the context of telemedicine. Specifically, the program will have to
> manage a series of vital parameters and return a response message to the patient, as well as inform the doctor in the
> event of significantly unbalanced parameter values. Keep track of the various interactions that have taken place using
> a database."

**Requisiti funzionali derivati:**
1. **Gestione parametri vitali**: ricevere e processare parametri vitali (pressione, frequenza cardiaca, temperatura,
   SpO₂, glicemia).
2. **Risposta al paziente**: messaggio di feedback basato sui valori (normale/anomalo).
3. **Alert al medico**: notifica quando i valori sono significativamente sbilanciati (soglie configurabili).
4. **Database delle interazioni**: tracciamento persistente di tutte le interazioni paziente-sistema-medico.

### Task 2 e 3: VLM + ISEQL (riferimento slide 12)
- Implementare i primi 2 livelli della pipeline VLM+ISEQL su video.
- Tool VLM consigliati: **Qwen 2.5 (ollama)**, **Gemini 1.5 robotics view (Google AI Studio)** o altri.
- Interfaccia grafica per definire modelli di eventi ISEQL (query come regular expression).

### Documentazione
- Consegna entro 57 ore dalla data dell'esame scelto.
- Descrizione dettagliata di tutti i passi con relativi motivi.
- Ogni membro del gruppo deve saper spiegare tutta la documentazione e il codice.

## 4. Checklist per la valutazione del progetto

**Concetti del corso che il progetto deve mostrare:**
- [ ] Definire l'agente con il framework **PEAS** (Performance, Environment, Actuators, Sensors) – slide 2.
- [ ] Identificare il **tipo di ambiente** (parzialmente osservabile, sequenziale, dinamico, continuo...) – slide 2.
- [ ] Giustificare il **paradigma di apprendimento** scelto (supervised / RL / rule-based) – slide 9.
- [ ] Mostrare consapevolezza di **bias-variance**, train/validation/test – slide 9.
- [ ] Collegare il comportamento a **utility/performance measure** – slide 2.
- [ ] Motivare le scelte con i **principi Responsible AI** e l'**AI Act** (settore medico = alto rischio) – slide 1.
- [ ] Spiegabilità e tracciabilità delle decisioni (importante in ambito medico) – slide 13.

**Requisiti del task:**
- [ ] Gestione multi-parametro con validazione e classificazione di severità.
- [ ] Messaggio al paziente (normale/anomalo + raccomandazioni).
- [ ] Notifica al medico per valori critici/significativamente sbilanciati.
- [ ] Database persistente di tutte le interazioni.
- [ ] Documentazione dettagliata con ragioni e fonti del codice.
