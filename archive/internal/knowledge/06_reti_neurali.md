# 06 – Reti Neurali

> Fonte: `10_Neural Networks.pdf` (Prof. Fabio Persia, DT0171, A.A. 2025/2026)

## 1. Storia (pag. 6-14)

| Anno | Evento |
|---|---|
| 1943 | McCulloch & Pitts: primo neurone artificiale (on/off) |
| 1958 | Rosenblatt: **Perceptron** (Mark I), primo NN implementato |
| 1969 | Minsky & Papert: limiti del perceptron (XOR non risolvibile) → AI winter |
| 1986 | Rumelhart, Hinton, Williams: **backpropagation** (Nature) → rinascita NN |
| 2011 | Avvento del **deep learning** grazie a GPU/TPU |
| 2017 | Vaswani et al.: **Transformer** ("Attention Is All You Need") |
| 2018 | Turing Award a Bengio, Hinton, LeCun |
| 2024 | Nobel in Fisica a Hopfield e Hinton |

## 2. Il Perceptron (pag. 16-23)

- **Modello computazionale**: riceve input x₁,...,xₙ, calcola **y = Σ wⱼxⱼ + b**, applica funzione di attivazione a soglia.
- **Funzione di attivazione**: step function `g(x) = {0 se x≤0, 1 se x>0}` (o sign function).
- **Formula** (pag. 18): **y = g(Σⱼ₌₀ⁿ wⱼxⱼ)** con w₀ = b, x₀ = 1.
- **Output** ∈ {0,1} → classificazione binaria.
- **Limite**: solo dati **linearmente separabili** (con attivazione lineare = modello lineare).

## 3. Heuristica di Rosenblatt (pag. 22-24) – domanda tipica d'esame

```
Inizializza w casualmente
Per ogni epoca:
  Per ogni esempio x⁽ⁱ⁾:
    ŷ⁽ⁱ⁾ = g(Σⱼ wⱼ·xⱼ⁽ⁱ⁾)
    Per ogni peso wⱼ:
      wⱼ ← wⱼ + λ·(y⁽ⁱ⁾ − ŷ⁽ⁱ⁾)·xⱼ⁽ⁱ⁾
```

- **λ > 0** = learning rate (iperparametro).
- Si ripete per un numero fisso di epoche o finché l'errore è basso.

## 4. Multilayer Perceptron (MLP) (pag. 27-35)

- Stacking di perceptron → rete fully-connected. Ogni strato Lₚ ha dₚ neuroni; ogni neurone riceve gli output dello strato precedente.
- **Formula per neurone u nello strato p** (pag. 30):
  **yᵤ⁽ᵖ⁾ = g( Σᵥ₌₀ dp₋₁ wᵥ,ᵤ⁽ᵖ⁾ · yᵥ⁽ᵖ⁻¹⁾ )**
- **Forma matriciale** (pag. 31): **y⁽ᵖ⁾ = g( W⁽ᵖ⁾ · y⁽ᵖ⁻¹⁾ )**
- **Funzione complessiva** (pag. 33): **MLP(x) = f⁽ᵏ⁾ ∘ f⁽ᵏ⁻¹⁾ ∘ ⋯ ∘ f⁽¹⁾(x)**
- **Input layer**: riceve i dati; **Hidden layers**: L₂,...,Lₖ₋₁; **Output layer**: Lₖ (pag. 29).
- **Feedforward**: l'informazione fluisce dall'input all'output senza feedback. Se ci sono feedback → **RNN** (pag. 35).
- **Attenzione** (pag. 34): se l'attivazione è lineare, l'MLP si riduce a un singolo strato (perceptron) → servono non-linearità.

## 5. Funzioni di attivazione (pag. 37-41)

| Funzione | Formula | Range | Uso |
|---|---|---|---|
| **ReLU** | max(0, x) | [0, +∞) | Default per hidden layers |
| **Sigmoid** | 1/(1+e⁻ˣ) | (0, 1) | Output layer (binaria) |
| **Tanh** | (eˣ−e⁻ˣ)/(eˣ+e⁻ˣ) | (−1, 1) | Hidden layers |
| **Arctan** | arctan(x) | (−π/2, π/2) | Hidden layers |
| **Softmax** | eˣⁱ / Σⱼeˣʲ | (0, 1), somma = 1 | Output layer (multiclasse) |

**Requisito**: le attivazioni devono essere **differenziabili** per la backpropagation.

## 6. Loss function (pag. 44-46)

- **Loss individuale**: ℓ(fw(xi), yi) misura la "distanza" tra predizione e label.
- **Loss globale** (pag. 46): **ℒ(W, fw, D) = (1/|D|) Σ ℓ(fw(xi), yi)**
- **Obiettivo**: argmin_W ℒ(W, fw, D).

### Funzioni di loss principali (pag. 59-62)

| Loss | Formula | Uso |
|---|---|---|
| **Lₚ distance** | ‖ŷ − y‖ₚ | Regressione |
| **NLL** | −log(ŷq) | Classificazione |
| **Cross-Entropy (CE)** | −ŷq + log(Σⱼeŷⱼ) | Classificazione multiclasse (con softmax) |
| **Binary CE (BCE)** | −(y·log(ŷ) + (1−y)·log(1−ŷ)) | Classificazione binaria |

## 7. Gradient Descent (pag. 47-48)

- Direzione di discesa più ripida = **−∇zϕ** (negativo del gradiente).
- **Aggiornamento** (pag. 48): **z ← z − λ∇zϕ**, λ = learning rate.
- **Limite**: solo funzioni differenziabili; può restare in minimi locali; **gradienti vanishing/exploding** (pag. 52).

## 8. Backpropagation (pag. 50-51)

- Applica la **chain rule** per funzioni composte.
- **Forward pass**: calcola e memorizza le derivate per ogni strato.
- **Backward pass**: calcola i gradienti dall'ultimo al primo strato, combinando i risultati intermedi (**programmazione dinamica**).
- Evita di ricalcolare derivate identiche.

## 9. Minibatch SGD (pag. 54-56)

```
Per ogni epoca:
  Shuffle D e dividilo in minibatch M
  Per ogni minibatch M:
    Calcola fW(xi) per ogni (xi, yi) ∈ M
    Calcola ℒ̃ = (1/|M|) Σ ℓ(fW(xi), yi)
    Calcola ∇W ℒ̃
    W ← W − λ∇W ℒ̃
```

- Più efficiente del GD completo; la **stocasticità** aiuta a sfuggire ai minimi locali.

## 10. Concetti chiave per l'esame

1. Perceptron: formula y = g(Σ wⱼxⱼ), limiti (XOR, linear separability) (pag. 16-23).
2. **Heuristica di Rosenblatt**: wⱼ ← wⱼ + λ(y⁽ⁱ⁾ − ŷ⁽ⁱ⁾)xⱼ⁽ⁱ⁾ (pag. 22-24).
3. MLP: y⁽ᵖ⁾ = g(W⁽ᵖ⁾ · y⁽ᵖ⁻¹⁾); MLP(x) = f⁽ᵏ⁾ ∘ ⋯ ∘ f⁽¹⁾(x) (pag. 31-33).
4. Funzioni di attivazione e quando usarle (pag. 37-41).
5. Loss: Lₚ, NLL, CE, BCE (pag. 59-62).
6. Gradient descent: z ← z − λ∇zϕ (pag. 48).
7. Backpropagation = chain rule + programmazione dinamica (pag. 50-51).
8. Minibatch SGD e vantaggi (pag. 54-56).
