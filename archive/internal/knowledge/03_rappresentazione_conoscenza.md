# 03 – Rappresentazione della Conoscenza

> Fonti: `4_Basics of Knowledge Representation.pdf`, `5_Knowledge Representation.pdf`
> (Prof. Fabio Persia, DT0171, A.A. 2025/2026)

## 1. Knowledge Representation and Reasoning (KRR) – PDF4

**Definizione** (PDF4, pag. 3): sottocampo dell'AI dedicato a rappresentare informazioni sul mondo in forma utilizzabile
da un sistema AI per risolvere compiti intelligenti tramite **reasoning automatizzato**.

- **Knowledge-based agents**: agenti che formano rappresentazioni di un mondo complesso, usano **inference** per
  derivare nuove rappresentazioni e dedurre cosa fare.
- **Logical agents**: knowledge-based agent che usano la **logica** per KRR.

### 1.1 Differenza con i Problem-Solving Agents (PDF4, pag. 4-5)

| | Problem-solving agent | Knowledge-based agent |
|---|---|---|
| KRR | Limitata: solo stati, azioni, goal, utility, euristica (per trovare un cammino) | Completa e riusabile |
| Task | Risolve un singolo problema | Accetta nuovi task sotto forma di **goal espliciti** |
| Competenza | Fissa | Aumenta rapidamente imparando nuova conoscenza |
| Adattamento | Limitato | Si adatta ad ambienti mutevoli aggiornando la KB |

I **CSP agent** sono un passo avanti (rappresentazione domain-independent, algoritmi generali) ma restano limitati.

### 1.2 Knowledge Base (PDF4, pag. 7-8)

- **Definizione** (pag. 7): insieme di frasi espresse in un **linguaggio formale** (knowledge representation language).
- Le frasi sono **axiom** quando date per assunte (non derivate).
- **Operazioni fondamentali** (pag. 8):
  - **TELL**: aggiungere nuove frasi alla KB.
  - **ASK**: interrogare ciò che è noto.
  - Entrambe coinvolgono **inferenza**.
  - Esempio: TELL("quando c'è tuono c'è fulmine"), TELL("oggi non c'è stato fulmine") → ASK("c'è stato tuono?") = No.

### 1.3 Linguaggi di rappresentazione (PDF4, pag. 9)

La **logica** è il modo naturale più usato: frasi KB formali, ben adatte all'inferenza.

**Gerarchia espressività vs efficienza computazionale** (dalla più semplice/efficiente alla più espressiva):
1. Logica proposizionale con solo Horn clauses
2. Logica proposizionale
3. Logica del primo ordine con solo Horn clauses
4. **Logica del primo ordine (FOL)** ← lingua preferita per le KB
5. Logica del secondo ordine

## 2. Ontological Engineering (PDF5)

- **Obiettivo** (PDF5, pag. 6-8): creare rappresentazioni generali e flessibili per domini complessi, concentrandosi
  su concetti astratti generali: **Eventi, Tempo, Oggetti Fisici, Credenze**.
- **Upper ontology**: framework generale di concetti (generici in alto, specifici in basso), analogo ai framework OOP
  (es. Java Swing: come `Window` → l'utente definisce `SpreadsheetWindow`).
- **Limite FOL** (pag. 10): le generalizzazioni hanno eccezioni o valgono solo in parte (es. "i pomodori sono rossi" –
  ma alcuni sono verdi, gialli, arancioni).

### 2.1 General-purpose ontologies (PDF5, pag. 15-19)

Due caratteristiche chiave (pag. 15): (1) applicabile in quasi ogni dominio; (2) unifica diverse aree della conoscenza.

**Quattro vie di costruzione** (pag. 18): team di ontologi → **CYC** (Lenat & Guha, 1990); importazione da database →
**DBPEDIA**; parsing di testi → **TEXTRUNNER**; crowd-sourcing. **Google Knowledge Graph** (pag. 19): >70 miliardi di fatti.

### 2.2 Categorie e oggetti (PDF5, pag. 21-28)

- **Category** (pag. 21): organizzazione degli oggetti in categorie; il ragionamento avviene a livello di categorie,
  l'interazione a livello di oggetti individuali.
- **Inferenza** (pag. 22): input percettivo → inferenza categoria → predizioni sull'oggetto.
- **Reification** (pag. 23): rappresentare categorie come oggetti → `Member(b, Basketballs)`, `Subset(Basketballs, Balls)`.
- **Inheritance** (pag. 24): Food → Fruit → Apples → ogni mela è commestibile per eredità.
- **Taxonomic hierarchy** (pag. 25): relazioni di subclass → tassonomia.
- **Disjoint** (pag. 28): categorie senza membri in comune. **Exhaustive decomposition** (partizione) = disjoint +
  decomposizione esaustiva (es. undergrad + graduate = tutti gli studenti).

### 2.3 Composizione fisica (PDF5, pag. 31-36)

- **PartOf** (pag. 31-32): relazione **transitiva** e **riflessiva**; gerarchia parte-tutto.
- **Struttura** (pag. 33): oggetti compositi caratterizzati da relazioni strutturali tra parti (es. bipede = 2 gambe + corpo).
- **Bunch** (pag. 35-36): raggruppamento fisico (non set matematico). `BunchOf(Apples)` = oggetto composito con le mele come parti, definito in termini di PartOf.

### 2.4 Measurements (PDF5, pag. 37, 46)

- Valori assegnati a proprietà (altezza, massa, costo); misure quantitative come funzione unità che prende un numero.
- Ordinamento anche per scale non quantitative (difficoltà esercizi, bellezza) con `>`.

### 2.5 Natural Kinds (PDF5, pag. 38-42)

- **Problema** (pag. 38-40): le categorie del mondo reale non hanno definizione netta; inevitabile in ambienti
  **parzialmente osservabili**.
- **Soluzione** (pag. 41): separare ciò che è vero per TUTTE le istanze da ciò che è vero solo per istanze tipiche →
  **`Typical(Tomatoes)`**. Si possono scrivere fatti utili senza definizioni esatte (pag. 42).

## 3. Event Calculus (PDF5, pag. 49-57)

**Limite delle azioni discrete** (pag. 49): non descrivono cosa succede DURANTE un'azione, né azioni simultanee.

**Event Calculus**: oggetti = **eventi**, **fluents** (proprietà che cambiano nel tempo), istanti temporali.
- **Fluent**: `At(Shankar, Berkeley)` = fatto che Shankar è a Berkeley.
- **Happens(E, t1, t2)**: l'evento E è accaduto nell'intervallo [t1, t2].
- **T(f, t1, t2)**: il fluent f è vero da t1 a t2.
- **Initiates / Terminates**: un evento inizia/termina un fluent.
- **Frame problem** risolto tramite **assiomi successori** (pag. 53-55).
- Estensioni: eventi simultanei, esogeni, continui, non deterministici (pag. 57).

## 4. Tempo – Relazioni di Allen (PDF5, pag. 58-64)

- **Momenti vs intervalli estesi** (pag. 58): solo i momenti hanno durata zero.
- **Funzioni** (pag. 59): `Begin`, `End`, `Time`, `Duration`.
- **Le 13 relazioni di Allen** (pag. 61-62): Before, Meet, Overlap, During, Starts, Finishes e i loro **inversi**, più **Equals**.

| Relazione | Simbolo | Definizione intuitiva |
|---|---|---|
| **Before** | i < j | i termina prima che j inizi |
| **Meet** | i m j | i termina quando j inizia |
| **Overlap** | i o j | i inizia prima di j e termina durante j |
| **During** | i d j | i è contenuto in j |
| **Starts** | i s j | i inizia con j e termina prima |
| **Finishes** | i f j | i termina con j e inizia dopo |
| **Equals** | i = j | stesso inizio e stessa fine |

Attenzione (pag. 63): **`Overlap(i,j)` non è simmetrica** per definizione (i inizia prima di j) – scelta utile per gli assiomi.

## 5. Fluents e oggetti (PDF5, pag. 65-69)

- **Oggetti fisici come eventi generalizzati** (pag. 65): "spazio-tempo chunk". Es. USA = evento dal 1776 a oggi.
- `President(USA)`: oggetto singolo costituito da persone diverse nel tempo; `President(USA, t)` è il termine temporale.
- **Identità logica** non cambia nel tempo (pag. 69): `Equals` (funzione) ≠ `=` (predicato logico).

## 6. Concetti chiave per l'esame

1. Knowledge-based agent vs problem-solving agent (PDF4, pag. 4-5).
2. KB = insieme di frasi + TELL + ASK; inferenza (PDF4, pag. 7-8).
3. Gerarchia dei linguaggi KRR; FOL = preferita (PDF4, pag. 9).
4. Ontological engineering / upper ontology (PDF5, pag. 6-8).
5. Reification, inheritance, tassonomia, disjoint/partition (PDF5, pag. 23-28).
6. PartOf transitiva e riflessiva; Bunch (PDF5, pag. 31-36).
7. Natural Kinds e Typical() (PDF5, pag. 38-42).
8. Event Calculus: Happens, Initiates, Terminates, T(f,t1,t2), frame problem (PDF5, pag. 49-57).
9. **Le 13 relazioni di Allen** con definizione formale (PDF5, pag. 61-64) – domanda tipica d'esame.
