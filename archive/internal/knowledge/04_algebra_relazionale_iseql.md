# 04 – Algebra Relazionale Estesa, ISEQL e Framework per Eventi

> Fonti: `6 - Relational Algebra Extension.pdf`, `7 - Efficient Implementation of Interval Relationships.pdf`,
> `8 - An Interactive Framework for Video Surveillance Event Detection and Modeling.pdf`
> (Prof. Fabio Persia, DT0171, A.A. 2025/2026)

## 1. Motivazione (PDF6)

**Debolezza degli automi stocastici** (PDF6, pag. 2): rappresentazione della negazione difficile; difficoltà con eventi
concorrenti; ricerca all'indietro dallo stato finale. → Soluzione: **estensione dell'algebra relazionale** per un
event query language (EQL).

**Low-level vs High-level events** (PDF6, pag. 3): la percezione umana raggruppa valori individuali in strutture maggiori
(anche per time series). EQL combina eventi basati su **timepoint** in strutture più grandi tramite **relazioni temporali su intervalli**.

## 2. ISEQL (PDF6)

**Definizione** (PDF6, pag. 7): **I**nterval-based **S**urveillance **E**vent **Q**uery **L**anguage – linguaggio basato
su algebra relazionale estesa con intervalli.

**Feature chiave** (PDF6, pag. 9):
- Estende le relazioni di Allen.
- **Overlap Percentage Constraints**: percentuale di sovrapposizione.
- **Cardinality Constraints**: quante volte un intervallo appare nel risultato.
- **Robustness**: tolleranza a piccole variazioni negli intervalli.
- Più efficiente degli approcci state-of-the-art.

### 2.1 Relazioni di Allen in versione ISEQL (PDF6, pag. 30-31)

ISEQL è un **superset** delle relazioni di Allen:
- Ogni relazione di Allen è coperta da almeno una relazione ISEQL e viceversa.
- Relazioni ISEQL **parametrizzate**: `Bef(δ:0)` → Before; `Bef(δ>0)` → OR(Meet, Before).

### 2.2 Cardinality Constraints (PDF6, pag. 26-27)

**Left Cardinality**: specifica quante volte un intervallo r_i deve apparire nel risultato.

### 2.3 Overlap Percentage Constraints (PDF6, pag. 28-29)

**Left Overlap Percentage**: solo coppie dove l'intervallo sinistro si sovrappone per più di p% della sua dimensione
con l'intervallo destro.

### 2.4 Formalizzazione in algebra relazionale (PDF6, pag. 33-38)

- **Interval-Timestamp Join** (pag. 33): operatore fondamentale.
- Tutti gli operatori ISEQL sono funzioni di questo join:
  - non parametrizzati (pag. 35), parametrizzati (pag. 36), con overlap percentage (pag. 37), con cardinality (pag. 38).

**Applicazioni**: video surveillance (PDF6, pag. 4-5), online social network – spamming e fake users (pag. 6).

## 3. Implementazione efficiente (PDF7)

### 3.1 Endpoint Index (PDF7, pag. 2-3)

Versione semplificata del **Timeline Index** (SAP HANA). Gli intervalli sono mappati su endpoint monodimensionali:
**e = `<timestamp, type, tuple_id>`** (timestamp = Ts/T_e, type = start/end, tuple_id = id tupla).

Esempio (pag. 3): per `r`: `<0,start,1>, <1,end,1>, <1,start,2>, <2,start,3>, <3,end,2>, <5,end,3>`.

### 3.2 JoinByS Algorithm (PDF7, pag. 1, 4-22)

**Obiettivo** (pag. 1): implementazione efficiente dell'**interval-timestamp join**. Ogni operatore ISEQL
(parametrizzato e non) è implementabile come funzione di JoinByS; supporta cardinality e overlap percentage.

- **Start Interval-Timestamp Join** (pag. 6-13): intervalli half-open; struttura dati `activeR` (lista degli intervalli
  attivi); scorre gli endpoint, aggiunge/rimuove da activeR, emette coppie quando il confronto è soddisfatto.
  Esempio con comp `<=`: `{<r2,s1>, <r3,s2>}`.
- **End Interval-Timestamp Join** (pag. 14-21): analogo, usa endpoint di fine con comp `<`. Esempio: `{<r2,s1>, <r3,s1>, <r3,s2>}`.
- **Mapping in JoinByS** (pag. 22): tutti gli operatori ISEQL (right cardinality, left overlap percentage) sono mappabili.

### 3.3 Competitors (PDF7, pag. 23-24)

- **Leung-Muntz algorithm**: per le relazioni di Allen (Leung & Muntz, ICDE 1990).
- **IEJoin algorithm**: per inequality join generiche (Khayyat et al., VLDB J. 2017).
- **JoinByS è più efficiente di entrambi** (pag. 24, grafico performance).

## 4. Framework interattivo per videosorveglianza (PDF8)

Riferimento: **Persia, Bettini, Helmer, CIKM '17**. Architettura a **4 layer** (PDF8, pag. 1-2):
1. **Image Processing Layer**
2. **Interval Action Detector**
3. **High-Level Event Detector**
4. **GUI + Application Core + Database**

### 4.1 Image Processing Layer (PDF8, pag. 3-13)

Usa **OpenCV** per identificare e tracciare persone/oggetti. Pipeline (pag. 4-12): lettura immagine + background →
sottrazione immagini → **algoritmo di Canny** (edge detection) → negazione → sovrapposizione → rilevamento oggetti
(bounding boxes) → disegno bounding boxes.

### 4.2 Interval Action Detector (PDF8, pag. 15-26)

Assembla frame con label di basso livello in **eventi di livello medio** descritti da intervalli. Database: **PostgreSQL**.
**Tre predicati di livello medio** (pag. 16-18):
1. **In**: persona dentro una regione.
2. **HasPkg**: persona porta un pacchetto (match bounding boxes persona-pacchetto).
3. **InsideCar**: persona in macchina (comparizione/scomparsa vicino auto).

Ogni predicato ha versione **offline** e **online**.

### 4.3 High-Level Event Detector (PDF8, pag. 28-34)

Combina eventi di livello medio in eventi complessi usando le relazioni di intervallo ISEQL:
- **BDPE** – Basic Direct Package Exchange (pag. 29-30).
- **BIPE/IDPE** – Basic Indirect Package Exchange (pag. 31-32).
- **Potential Car Theft** (pag. 33).
- **Unattended Package (UP)** (pag. 34): `UP = UPa - UPb` (pacchetto incustodito).

### 4.4 GUI e Application Core (PDF8, pag. 36-39)

- **GUI** (Java Swing): definisce nuovi modelli di eventi e li verifica immediatamente.
- **Application Core** (Java): traduce l'input utente in modello di algebra relazionale → genera **stored procedure
  PL/pgSQL** → connette via JDBC al database.
- L'utente fornisce un'istanza del modello → il sistema la generalizza.

### 4.5 Database (PDF8, pag. 40)

**PostgreSQL 9.4**: ogni operatore dell'event model = stored procedure; operatori assemblabili dinamicamente in modelli diversi.

### 4.6 Funzionalità e performance (PDF8, pag. 41, 43-51)

**Funzionalità** (pag. 41): detection low/medium/high-level, classificazioni utente, definizione di nuovi predicati
atomici/medi e nuovi modelli di eventi ad alto livello.

**Dataset**: ITEA CANDELA, University parking lot, PETS 2006, VIRAT, Facebook Clickstream.
**Performance online**: Image Processing ~**250 ms/frame** (≈4 fps); Interval Action Detector ~**46 ms/frame** (real-time).
**Accuracy** (pag. 49): misurata come m/n (m = eventi identificati, n = ground truth umano).
**Robustness parameter** (pag. 51): impatto positivo sulla rilevazione eventi.

## 5. Concetti chiave per l'esame

1. ISEQL: definizione e feature (overlap %, cardinality, robustness) (PDF6, pag. 7-9).
2. ISEQL come superset parametrizzato delle relazioni di Allen (PDF6, pag. 30-31).
3. Interval-Timestamp Join come operatore fondamentale (PDF6, pag. 33).
4. Endpoint Index: `e = <timestamp, type, tuple_id>` (PDF7, pag. 2-3).
5. JoinByS vs Leung-Muntz vs IEJoin (PDF7, pag. 23-24).
6. Architettura a 4 layer e predicati In/HasPkg/InsideCar (PDF8, pag. 1-2, 16-18).
7. Eventi ad alto livello: BDPE, BIPE, Potential Car Theft, Unattended Package (PDF8, pag. 29-34).
8. Stored procedure PL/pgSQL in PostgreSQL come operatori (PDF8, pag. 40).

## 6. Bibliografia

- Allen (1983), *Maintaining knowledge about temporal intervals*, CACM 26(11).
- Helmer, Persia (2016), *ISEQL*, IJMDEM vol. 7.
- Helmer, Persia (2016), *High-level surveillance event detection*, IEEE ICSC 2016.
- Persia, Bettini, Helmer (2017), *Interactive Framework for Video Surveillance*, CIKM '17.
- Leung, Muntz (1990), *Query processing for temporal databases*, ICDE.
- Khayyat et al. (2017), *Fast and scalable inequality joins*, VLDB J. 26(1).
- Kaufmann et al. (2013), *Timeline Index*, SIGMOD.
