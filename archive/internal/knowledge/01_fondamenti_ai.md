# 01 – Fondamenti dell'Intelligenza Artificiale

> Fonti: `0 - Course Introduction.pdf`, `1_AI-fundamentals_2526_introduction.pdf`
> Docente: Prof. Fabio Persia – Corso DT0171 Artificial Intelligence, A.A. 2025/2026, Univ. dell'Aquila

## 1. Cos'è l'AI

> "Artificial Intelligence (AI) is the field of study which focuses on computing systems able to imitate intelligent human behavior."
> (slide 1, pag. 3)

Un sistema informatico esibisce AI quando svolge compiti che normalmente richiedono **pensiero e ragionamento umano**.

**L'intelligenza** comprende (slide 1, pag. 4): astrazione, logica, comprensione, auto-consapevolezza, apprendimento,
conoscenza emotiva, ragionamento, pianificazione, creatività, pensiero critico, problem-solving.
Descritta come la capacità di *percepire/inferire informazioni, trattenere conoscenza e applicarla in comportamenti adattivi*.

**Ambito multidisciplinare** (slide 1, pag. 5): filosofia, matematica, economia, neuroscienze, psicologia, informatica,
ingegneria, teoria del controllo, cibernetica.

## 2. Le quattro definizioni di AI

Due dimensioni (slide 1, pag. 40-41):

| | **Comportamento** | **Pensiero** |
|---|---|---|
| **Umanità** | Agire come un umano (Test di Turing) | Pensare come un umano (modellazione cognitiva) |
| **Razionalità** | Agire razionalmente (miglior esito) | Pensare razionalmente (logica e ragionamento) |

## 3. Test di Turing (slide 1, pag. 42-46)

- Progettato da **Alan Turing** (1950, "Computing Machinery and Intelligence"); originariamente "imitation game".
- Tre terminali: interrogatore umano, rispondente umano, computer. Il computer supera il test se l'interrogatore
  non distingue tra umano e macchina.

**Capacità richieste** (pag. 44): NLP, knowledge representation, automated reasoning, machine learning.

**Total Turing Test** (pag. 45): aggiunge computer vision, speech recognition e robotica.

**Limiti** (pag. 46): non riproducibile né analizzabile matematicamente; nessuna definizione rigorosa di intelligenza;
presuppone che emulare l'umano sia l'approccio giusto (analogia: gli aerei non imitano gli uccelli).

## 4. Cronologia dell'AI (slide 1, pag. 48-71)

### Inception (1943-1956)
- **McCulloch & Pitts (1943)**: primo modello di neurone artificiale (on/off).
- **Hebb (1949)**: regola di Hebbian learning (modifica della forza delle connessioni).
- **Minsky & Edmonds (1950)**: SNARC, primo computer per reti neurali.
- **Dartmouth Workshop (1956)**: nascita formale dell'AI; termine coniato da **John McCarthy**.
- **Logic Theorist** (Newell & Simon): primo programma che dimostra teoremi matematici.

### Entusiasmo iniziale (1952-1969)
- **Arthur Samuel (1956)**: programma di dama con **reinforcement learning**.
- **GPS** – General Problem Solver (1957, Newell-Shaw-Simon): primo programma "thinking humanly".
- **Physical Symbol System Hypothesis** (1976, Newell-Simon): "un sistema di simboli fisici ha i mezzi necessari
  e sufficienti per un'azione intelligente generale".
- **Lisp** (1958, McCarthy): linguaggio dominante in AI per 30 anni.
- **Perceptron** (1958, Rosenblatt): prima rete neurale artificiale implementata.

### Dose di realtà (1966-1973)
- Le previsioni di Herbert Simon (1957) si avverarono in 40 anni, non 10.
- Cause del fallimento: sistemi basati su "introspezione informata" anziché analisi accurata del compito; mancata
  inattuabilità dei problemi prima della teoria della complessità.
- **Minsky & Papert, "Perceptrons" (1969)**: i perceptron non rappresentano XOR → "neural network winter", crollo dei finanziamenti.

### Expert Systems (1969-1986)
- **DENDRAL** (1969): primo sistema knowledge-intensive (inferenza di struttura molecolare da spettrometria di massa).
- Boom dell'industria AI (1980-1988), poi **AI Winter**: gli expert system non gestivano l'incertezza e non imparavano.

### Ritorno delle reti neurali (1986-oggi)
- **Backpropagation** (Rumelhart, Hinton, Williams, 1986, Nature): "Learning representations by back-propagating errors".

### Ragionamento probabilistico e ML (1987-oggi)
- **Judea Pearl (1988)**: *Probabilistic Reasoning in Intelligent Systems* → Bayesian networks.
- **Rich Sutton (1988)**: collegamento reinforcement learning → **Markov Decision Processes (MDP)**.

### Big Data (2001-oggi)
- Dataset con trilioni di parole, miliardi di immagini.
- **Banko & Brill (2001)**: aumentare i dati di 2-3 ordini di grandezza è più efficace che migliorare l'algoritmo.
- **Watson IBM (2011)**: vince a Jeopardy!

### Deep Learning (2011-oggi)
- **AlexNet** (Krizhevsky et al., 2012/2013): svolta in ImageNet.
- **Transformer** (Vaswani et al., 2017): "Attention Is All You Need" → base di GPT-x, LLaMA.
- **Turing Award 2018**: Bengio, Hinton, LeCun.
- **Nobel Fisica 2024**: Hopfield e Hinton (v. anche 06_reti_neurali.md).

## 5. AGI vs Narrow AI (slide 1, pag. 73-78)

| Concetto | Descrizione |
|---|---|
| **Narrow AI (Weak AI)** | AI applicativa e di dominio specifico (genera testo, gioca a scacchi, categorizza immagini) |
| **AGI (Strong AI)** | Intelligenza generale di scopo generale, capacità simili a quelle umane |
| **Strong AI** | Talvolta riservato a sistemi con sentienza/coscienza |

Framework DeepMind per i livelli AGI (2023): Emerging → Competent → Expert → Virtuoso → Superhuman.
GPT-4 e LLaMA 2 sono considerati "emerging AGI". Esempi storici di AGI: GPS (1957), Cyc (dal 1980).

## 6. Symbolic AI / GOFAI (slide 1, pag. 79-86)

**Definizione** (pag. 80): metodologie basate sulla manipolazione di rappresentazioni simboliche formali e ragionamento
astratto su tali simboli.

**Caratteristiche**: simboli formali interpretabili dall'uomo; simboli atomici combinati in simboli complessi;
regole formali di manipolazione.

**Tecniche**: logic programming, ricerca su alberi/grafi di stato, sistemi esperti, sistemi rule-based, inference engine.
**GOFAI** = Good Old-Fashioned AI (termine coniato da Haugeland, 1985).

| Pro | Contro |
|---|---|
| Conoscenza esplicita e strutturata | Conoscenza incompleta |
| Ragionamento formale | Difficoltà con incertezza/ambiguità |
| Trasparenza e interpretabilità | Scalabilità limitata |
| Flessibilità | Capacità limitata di apprendimento/adattamento |
| Non servono enormi quantità di esempi | |

## 7. Machine Learning e Connectionist AI (slide 1, pag. 87-97)

**Definizione ML** (pag. 88): campo che studia sistemi capaci di imparare da esempi, costruire ipotesi e usarle come
modelli matematici per predizioni su dati non visti.

- **ML vs AI** (pag. 89): ML è un **sotto-campo** dell'AI; non tutti i sistemi AI usano ML.
- **ML vs Symbolic AI** (pag. 90): ML = **bottom-up** (dal particolare al generale); symbolic AI = **top-down**
  (dal generale al particolare).

**Reti neurali artificiali** (pag. 92-94): ispirate a quelle biologiche; neuroni collegati da pesi; segnale reale;
funzione di attivazione non-lineare; il training determina i pesi corretti.

**Deep Learning** (pag. 95): ML con molti strati di semplici elementi di calcolo.
**Connectionist AI** (pag. 96): AI basata su reti neurali.

| Pro Connectionist | Contro Connectionist |
|---|---|
| Capacità di apprendere/adattarsi | Conoscenza non simbolica (codificata nelle connessioni) |
| Gestione incertezza/ambiguità | Non interpretabili dall'uomo |
| Gestione grandi quantità di dati | |

## 8. Neuro-Symbolic AI (slide 1, pag. 98-106)

**Obiettivo** (pag. 100): unire symbolic e connectionist AI. Combinando apprendimento e ragionamento si ottengono
contemporaneamente **explainability + efficiency + generalization** (pag. 101-102).

**Applicazioni**:
- **Auto a guida autonoma** (pag. 103-104): rete neurale per riconoscimento oggetti + AI simbolica per decisioni
  basate su regole del traffico (Sharifi et al., arxiv 2307.01316v2).
- **AlphaGeometry** (pag. 105, Google DeepMind): risolve problemi di geometria a livello medaglia d'oro IMO;
  combina language model + symbolic deduction engine (Nature, gennaio 2024).

**Due categorie** (pag. 106):
1. Metodi che "iniettano" conoscenza simbolica nelle reti neurali;
2. Metodi che "incorporano" pattern neurali nella conoscenza simbolica (lifting + ragionamento simbolico).

## 9. Sotto-campi dell'AI (slide 1, pag. 117-121)

Problem solving, knowledge representation, automated reasoning, automated planning, machine learning, machine
perception, intelligent agents, robotics, NLP, computer vision, social intelligence, cognitive intelligence,
generative AI.

## 10. Etica e Responsible AI (slide 1, pag. 123-127)

**Benefici** (pag. 123-124): diagnosi medica migliorata, previsioni meteo, guida sicura, gestione agricola,
ottimizzazione processi, automazione, accessibilità, traduzione automatica, costo marginale quasi zero.

**Principi Responsible AI** (pag. 126): sicurezza, responsabilità, equità, diritti umani e valori, privacy,
diversità/inclusione, collaborazione, evitare concentrazione di potere, trasparenza, implicazioni legali/policy,
limitare usi dannosi, considerare implicazioni per l'occupazione.

**AI Act** (pag. 127): regolamento UE in vigore dal **1° agosto 2024**; l'AI deve essere sviluppata e usata in modo
sicuro, etico e rispettoso dei diritti fondamentali; sanzioni proporzionali.

## 11. Applicazioni attuali (slide 1, pag. 10-26)

Veicoli autonomi (Waymo), pianificazione autonoma (NASA Remote Agent, EUROPA, SEXTANT), traduzione automatica
(Google Translate), riconoscimento vocale (5.1% WER Microsoft 2017), recommender (Amazon, Netflix, Spotify, YouTube),
spam filter, game playing (Deep Blue 1997, AlphaGo 2017, AlphaZero 2018), generazione contenuti (Midjourney, DALL-E),
chatbot (ChatGPT, Copilot, Gemini), trading, climate science.

**Medicina** (pag. 23): **LYNA** – sistema diagnostico, 99.6% accuratezza sul tumore al seno metastatico. Limiti:
dimostrare miglioramento degli outcome clinici, trasparenza, assenza bias, privacy dati. FDA approvals: 2 nel 2017,
12 nel 2018.

## 12. Concetti chiave per l'esame

1. Turing Test: struttura, capacità richieste, limiti (pag. 42-46).
2. Le 4 definizioni di AI: umanità/razionalità × pensiero/comportamento (pag. 40-41).
3. ML ≠ AI; ML bottom-up vs symbolic AI top-down (pag. 89-90).
4. Symbolic vs Connectionist vs Neuro-symbolic AI (pag. 79-106).
5. Responsible AI principles e AI Act (pag. 126-127).
6. Cronologia: date ed eventi chiave (1943, 1956, 1958, 1969, 1986, 1988, 2011, 2017).
