# 05 – Machine Learning

> Fonte: `9_Machine Learning.pdf` (Prof. Fabio Persia, DT0171, A.A. 2025/2026)

## 1. Definizione e posizione nell'AI

- **ML** è un sottocampo dell'AI che studia la capacità di migliorare le proprie prestazioni basandosi sull'esperienza (pag. 3).
- **ML vs AI simbolica** (pag. 5): ML = **bottom-up** (dai particolari/esempi → generale/modello);
  AI simbolica = **top-down** (dalle ipotesi generali → conclusioni particolari).
- **Perché ML** (pag. 6): (1) i progettisti non possono prevedere tutte le situazioni future; (2) a volte non sanno
  come programmare esplicitamente la soluzione.

## 2. Formulazione del problema (pag. 8-9)

- Obiettivo: apprendere una funzione **f: X → Y** che mappa input x ∈ X (vettore n-dimensionale di feature) a output y ∈ Y.
- La funzione vera f non è apprendibile esattamente; si impara un'**ipotesi h** che approssima f.
- **Training**: processo di calcolo del modello. **Inference**: uso del modello su dati nuovi.

## 3. Classificazione vs Regressione (pag. 10-15)

| Tipo | Output Y | Esempio |
|---|---|---|
| **Classificazione binaria** | {0, 1} | Spam/non-spam |
| **Classificazione multiclasse** | {−1, 0, +1} | Sentiment analysis |
| **Classificazione multilabel** | insieme di valori | Generi musicali di una canzone |
| **Regressione singola** | ℝ | Temperatura di domani |
| **Regressione multioutput** | ℝᵏ | Coordinate k-dimensionali |

## 4. Assunzioni chiave (pag. 22)

- x è tipicamente un vettore numerico n-dimensionale (X ⊆ ℝⁿ).
- Il modello h è estratto da uno **spazio delle ipotesi ℋ** (model class): funzioni lineari, polinomiali, reti neurali, ecc.
- La **complessità del modello deve essere bilanciata** rispetto alla quantità di dati disponibili.

## 5. Paradigmi di apprendimento

### 5.1 Supervised Learning (pag. 19)
- **Input**: coppie (xᵢ, yᵢ) con yᵢ = f(xᵢ) = **label/ground truth**.
- Il training set fornisce il "segnale di supervisione".
- **Vantaggi**: performance prevedibili, ben compreso teoricamente.
- **Limiti**: richiede annotazione umana (costosa); overfitting se il dataset è piccolo o rumoroso.

### 5.2 Self-Supervised Learning (pag. 21)
- Le label sono prodotte **automaticamente** dagli esempi stessi (senza annotazione umana).
- Esempio: predire la prossima parola in un testo.
- **Vantaggio**: elimina il costo dell'annotazione.
- **Rilevanza telemedicina**: generare automaticamente task di predizione da dati clinici strutturati (EHR).

### 5.3 Unsupervised Learning (pag. 35-36)
- Solo esempi xᵢ, **senza label**. Task principali: **clustering**, anomaly detection, similarity search, ranking.
- **Clustering** (pag. 36): raggruppare oggetti simili; i metodi differiscono per similarità, forma dei cluster, numero di cluster.
- **Rilevanza telemedicina**: clustering di pazienti per profili clinici; anomaly detection sui parametri vitali.

### 5.4 Semisupervised Learning (pag. 37)
- Solo un sottoinsieme (piccolo) di esempi ha le label; si sfruttano sia dati etichettati che non.
- **Rilevanza telemedicina**: pochi dati annotati dal medico, molti non etichettati.

### 5.5 Reinforcement Learning (pag. 38)
- Modella l'**interazione agente-ambiente** tramite reward/punishment.
- L'agente decide quali azioni hanno contribuito al reward e aggiusta il comportamento.
- **Rilevanza telemedicina**: agente che regola parametri di terapia (dosaggio, timing) in base ai feedback del paziente.

## 6. Training, Validazione e Test (pag. 31-33)

- **Training set** (pag. 31): addestra il modello (impara i parametri).
- **Validation set** (pag. 32): valuta modelli con diversi **iperparametri**, sceglie il migliore.
- **Test set**: valutazione finale su dati mai visti. Splitting tipico: 80/10/10 o 80/20.
- **Cross-Validation** (pag. 33): **k-fold** → si divide il dataset in k gruppi; per ognuno si usa come validation e gli
  altri come training; si mediano le performance. Popolari k=5 o k=10. **LOOCV**: k = numero di campioni (extremum).
  Serve comunque un test set separato.

## 7. Bias-Variance Tradeoff (pag. 25-30, 44-47)

- **Bias** (informale, pag. 44): differenza tra il valore atteso secondo h e il valore vero di f. Alto bias = **underfitting**.
- **Varianza** (informale, pag. 46): varianza dei valori calcolati da h su tutti i possibili training set. Alta varianza = **overfitting**.
- Sono in **conflitto**: quando uno è alto, l'altro è basso.
- **Capacità del modello**: modelli complessi = low bias / high variance; modelli semplici = high bias / low variance.
- **Overfitting** (pag. 47): modello troppo specializzato sul training set, generalizza male.
- **Rasoio di Ockham** (pag. 30): scegliere l'ipotesi più semplice che spiega i dati.

## 8. Iperparametri (pag. 31)

Parametri **non appresi** dal training ma fissati prima: grado polinomiale, numero di layer, numero di alberi,
learning rate, funzione di attivazione, ecc. Si sintonizzano sul validation set.

## 9. Concetti chiave per l'esame

1. Definizione ML e differenza da AI simbolica (bottom-up vs top-down) (pag. 3-5).
2. Classificazione vs Regressione, con esempi concreti (pag. 10-15) – domanda tipica d'esame.
3. Paradigmi: supervised, self-supervised, unsupervised, semisupervised, RL (pag. 19-38).
4. Train/validation/test split e cross-validation (pag. 31-33).
5. Bias-variance tradeoff, overfitting/underfitting, Rasoio di Ockham (pag. 25-30, 44-47).
6. Iperparametri vs parametri appresi (pag. 31).
