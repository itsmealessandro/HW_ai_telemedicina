# 09 – Mappa Concetti del Corso → Progetto di Telemedicina

> Sintesi che collega le slide del corso alle scelte progettuali dell'homework "Agente intelligente per telemedicina".
> File di supporto alla discussione sulle scelte di architettura e di apprendimento.

## 1. L'agente in ottica PEAS (slide 2, pag. 21-24)

Definizione dell'agente di telemedicina secondo il framework PEAS:

| Dimensione | Contenuto per il progetto |
|---|---|
| **P**erformance | Accuratezza nella classificazione dello stato (normale/anomalo), correttezza delle notifiche al medico, tempi di risposta, sicurezza del paziente |
| **E**nvironment | Paziente, dispositivi wearable, parametri vitali, storico interazioni, medico |
| **A**ctuators | Messaggio al paziente, notifica al medico, aggiornamento database, escalation |
| **S**ensors | Input dei parametri vitali (pressione, HR, temperatura, SpO₂, glicemia), input utente |

**Tipo di ambiente** (slide 2, pag. 25-31): parzialmente osservabile (solo i dati forniti), multi-agent
(paziente + medico + sistema), non deterministico, sequenziale, dinamico, continuo, parzialmente noto.
→ È il caso più difficile (pag. 31); giustifica un agente **model-based / goal-based** con componente di apprendimento.

**Tipo di agente consigliato** (slide 2, pag. 40-47): **utility-based + learning**: l'utility function internalizza la
performance measure; la componente learning adatta il comportamento con l'esperienza (pag. 46-47).

## 2. Approcci AI possibili (slide 1, pag. 79-106)

| Approccio | Applicazione in telemedicina |
|---|---|
| **Symbolic AI / rule-based** | Regole mediche e guideline ("IF febbre AND tosse THEN sospetto..."); trasparente e interpretabile; nessun dato di training necessario |
| **Connectionist / Deep Learning** | Classificazione dello stato del paziente da pattern vitali; NLP sulle cartelle cliniche |
| **Neuro-Symbolic** | Deep learning per la percezione + ragionamento su regole cliniche; ideale per diagnosi assistita (explainability + performance) |
| **ML classico (supervised)** | Classificazione normale/anomalo/critico dei parametri; predizione di rischio; clustering di pazienti |
| **Unsupervised** | Anomaly detection sui parametri vitali (senza label) |

## 3. Supervised vs Reinforcement Learning – analisi per il progetto

### 3.1 Cosa dicono le slide
- **Supervised** (slide 9, pag. 19): impara da coppie (xᵢ, yᵢ) con label/ground truth. Vantaggi: performance
  prevedibili, ben compreso. Limiti: richiede annotazioni (costose).
- **Reinforcement** (slide 9, pag. 38): apprende tramite interazione agente-ambiente con reward/punishment; l'agente
  deve capire *quali azioni* hanno contribuito al reward.
- La definizione stessa del task è **episodica/classificatoria**: l'agente riceve parametri vitali e deve *restituire
  un messaggio* → è un mapping input → output con risposta desiderata definibile, non un problema di sequenze di
  decisioni che modificano lo stato.

### 3.2 Perché il supervised è più adatto al task
1. **Il task è una classificazione**: "dato un set di parametri vitali → livello di rischio e messaggio".
   Questo è esattamente il problema supervised (slide 9, pag. 10-15, 19).
2. **Il ground truth esiste**: le soglie mediche di normalità/anomalia sono note (linee guida); si può costruire un
   dataset etichettato con regole cliniche o con dataset pubblici.
3. **La valutazione è diretta**: accuracy/precision/recall sul test set; nessuna ambiguità di reward shaping.
4. **L'RL richiede un ambiente di interazione** (stato, azioni, reward) che per un servizio di consulenza al paziente
   è artificiale: la "ricompensa" per un messaggio corretto è post-hoc e difficile da definire in modo non arbitrario.

### 3.3 Dove l'RL avrebbe senso (scenario alternativo)
Solo se il task diventasse **decisionale e sequenziale**: es. agente che regola dosaggi/timing di terapia e riceve
feedback dai parametri successivi del paziente (slide 9, pag. 38). Nel task attuale ("restituire un messaggio") questo
non è richiesto.

### 3.4 Considerazioni pratiche per l'implementazione
- **Dati limitati**: un modello supervised puro richiede un dataset. Se non disponibile, una base **rule-based/symbolic**
  è la soluzione immediata e interpretabile; un modello supervised può essere addestrato su dati generati dalle stesse
  regole o da dataset pubblici (es. dataset di parametri vitali). **È esattamente la strada scelta dal progetto**:
  il teacher rule-based genera offline le label del dataset sintetico e l'MLP le distilla (knowledge distillation).
- **Spiegabilità** (slide 13, pag. 33-36): in ambito medico l'interpretabilità è critica. Un modello black-box
  (NN profonda) è più difficile da motivare in un contesto clinico. Il progetto risponde con un **safety gate
  deterministico a precedenza assoluta**: le regole cliniche restano la fonte di verità e l'MLP opera solo dietro
  di esse (mai declassamento dei casi critici).
- **Ibrido**: il percorso più solido per l'esame è mostrare BOTH: l'agente rule-based/symbolic come baseline
  interpretabile e un modello supervised (es. logistic regression / random forest / piccolo MLP) addestrato per
  confronto, discutendo bias-variance e cross-validation (slide 9, pag. 31-33). Il progetto realizza l'ibrido
  **neuro-symbolic**: MLP numpy (connectionist) + regole cliniche (symbolic) con precedenza delle regole.

## 4. Rilevanza degli altri argomenti del corso

- **Rappresentazione della conoscenza** (slide 4-5): la KB medica (range, soglie, correlazioni) è una knowledge base;
  l'inferenza può essere vista come ASK su regole. Le relazioni di Allen sono utili se si aggiungono *serie temporali*
  di parametri (es. "tachicardia DOPO ipotensione").
- **Algebra relazionale / ISEQL / interval reasoning** (slide 6-8): rilevanti se si passa da singole rilevazioni a
  *pattern temporali* negli storici dei parametri (es. rilevare eventi "crisi ipertensiva" come intervallo). È un
  possibile arricchimento opzionale del progetto e argomento da conoscere per l'esame.
- **VLM in healthcare** (slide 13): riferimento per la discussione su etica, bias, privacy e spiegabilità; non necessario
  per il task a parametri vitali.
- **Recommender** (slide 3): non centrale per il task, ma utile se si aggiungono *raccomandazioni personalizzate*
  (content-based sul profilo del paziente).

## 5. Etica e regolamentazione (slide 1, pag. 123-127)

- **AI Act UE** (in vigore 1/8/2024): i sistemi medici AI sono ad alto rischio → documentazione, trasparenza, controllo umano.
- **Responsible AI**: sicurezza, privacy (GDPR), equità, trasparenza, responsabilità.
- **Limiti dell'AI medica** (slide 1, pag. 23): non basta l'accuratezza; servono prove di miglioramento degli outcome
  clinici, trasparenza, assenza di bias, privacy dei dati.

## 6. Messaggi chiave per la discussione

1. Il task di telemedicina descritto è un **problema di classificazione** → paradigma naturale: **supervised**.
2. L'**RL** non è sbagliato in assoluto, ma è inadatto al task *così com'è formulato* (nessuna sequenza di decisioni
   che modifichi lo stato; ground truth definibile a priori).
3. Con **pochi dati disponibili**, la base rule-based/symbolic resta valida e interpretabile; un supervised model
   può integrarla (generando dati etichettati dalle regole stesse) per dimostrare la competenza in ML all'esame.
4. Motivare ogni scelta con i concetti del corso (PEAS, ambiente, bias-variance, train/validation/test, AI Act)
   è ciò che conta per la valutazione.

## 7. Stato finale del progetto (Fase 8, 2026-08-19)

Il progetto è completo (Fasi 0-7 implementate, Fase 8 = documentazione e review).
Decisioni finali e lettura dei risultati:

### 7.1 Decisioni architetturali finali

- **Knowledge distillation**: il teacher rule-based (`safety_rules.py` +
  `training/teacher_rules.py`) genera offline le label del dataset sintetico
  congelato; l'MLP numpy le approssima. L'accuracy misura la **fedeltà al
  teacher**, non la validità clinica.
- **Safety gate a precedenza assoluta**: un caso critico per le regole non
  passa mai dall'MLP → recall di sistema sui critici = 1.0 per costruzione.
- **Zona di incertezza**: `max(probabilità softmax) < 0.6` → fallback al
  rule-based (conservatività sui boundary).
- **Regola `max(rule-based, MLP)`**: la classe finale è la più alta tra le
  due; il modello può solo alzare, mai abbassare (falsi positivi in più =
  prezzo intenzionale della conservatività).
- **Soglia glicemia critica unificata a ≤ 60 mg/dL** (discrepanza legacy
  teacher ≤60 vs override RL ≤55 risolta a favore della più conservativa).
- **Persistenza SQLite rumorosa**: DB non scrivibile → `ValueError`, mai
  perdita silenziosa di record clinici; notifiche per casi critici su DB +
  stderr, generate dal service (non dalla CLI).
- **Vincolo rispettato**: solo numpy + standard library (sqlite3 incluso);
  nessuna dipendenza nuova; fonte unica delle soglie con guard test.

### 7.2 Risultati (report.json, seed 41)

| Metrica | Valore | Lettura |
|---|---|---|
| Accuracy test congelato | 0.9851 | fedeltà al teacher su split bufferizzato |
| Accuracy slice no-buffer | 0.9324 | stima più onesta sui near-threshold |
| Kappa vs teacher | 0.9775 | accordo quasi perfetto con le regole |
| Recall `alto` (MLP) | 0.9958 (30 mancati) | **nessun declassamento di sistema**: il gate copre i 30 mancati |
| Errori su 20000 | 298 (distanza media 0.524) | errori concentrati sui boundary, dove interviene la zona di incertezza |
| Baseline teacher | 1.0 | upper bound di imitazione (label generate dalle stesse regole) |

### 7.3 Limiti dichiarati (dettaglio in `supervised_telemedicina/docs/review_finale.md`)

Label sintetiche (niente pretesa diagnostica), test bufferizzato ottimistico,
falsi positivi introdotti da `max(rule-based, MLP)`, edge case notifica
persa se l'insert fallisce dopo quello dell'analisi, thread-safety non
garantita, nessuna webapp (CLI è il canale).
