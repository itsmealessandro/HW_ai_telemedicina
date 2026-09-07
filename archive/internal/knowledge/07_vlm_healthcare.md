# 07 – VLM in Healthcare e VLM + ISEQL

> Fonti: `12_VLM + ISEQL.pdf`, `13_VLM in Healthcare.pdf` (Prof. Fabio Persia, DT0171, A.A. 2025/2026)

## 1. VLM (Vision-Language Models) – basi

**Contesto** (12, pag. 1-3): la videosorveglianza richiede analisi continua di flussi massivi di dati eterogenei.
I **VLM** sono capaci di riconoscimento a livello di frame e captioning semantico (zero-shot video understanding).
**Limite VLM** (12, pag. 1): deboli nel reasoning temporale, composizione di eventi, robustezza alle occlusioni.

## 2. Architettura a 3 livelli VLM + ISEQL (12, pag. 4-14)

| Livello | Componente | Funzione |
|---|---|---|
| 1 | **VLM** | Analisi per-frame: presenza persone/oggetti, relazioni semplici, griglia spaziale |
| 1.5 | **Data Adaptation Module** | Converte l'output testuale del VLM in Simple Time-Point Labeling (DB temporale relazionale) |
| 2 | **Interval Action Detector** | Aggrega time-point in intervalli; pulisce errori VLM; gestisce occlusioni |
| 3 | **High-Level Event Detector** | Rileva eventi complessi (scambi pacchi, loitering) usando pattern ISEQL |

- **VLM nel framework** (12, pag. 5-6): processa ogni frame singolarmente → (1) descrizione testuale, (2) immagine con
  griglia; rileva eventi medio-complessi ("persona seduta", "persona con bottiglia").
- **ISEQL** (12, pag. 11, 14): lavora sui 3 livelli dell'architettura **ANSI-SPARC** (esterno, concettuale, interno).
  Al livello interno: algoritmi efficienti per **cache-efficient sweeping-based interval joins** (Piatov et al., VLDB 2021).
  Usa le **relazioni di Allen** sulle intervalli temporali per il reasoning.
- **Running example** (12, pag. 15-19): 4 frame non consecutivi → VLM rileva persone + oggetti + relazioni spaziali →
  Data Adaptation → DB temporale → Interval Action Detector aggrega in intervalli ("persona A tiene pacco X dal frame
  1 al 4") → High-Level Event Detector rileva **BDPE** (Basic Direct Package Exchange) o **BIPE** (Basic Indirect
  Package Exchange) o assenza di scambio.

**Vantaggi** (12, pag. 20-22): VLM = flessibilità percettiva; ISEQL = correzione errori + modellazione scenari complessi.
Separa **perception** (VLM) da **reasoning** (logica a intervalli) → sistema interpretabile e spiegabile.
I VLM comprimono la scena in vettori latenti → difficoltà nel riconoscimento temporale su video lunghi; il framework
mantiene **struttura temporale e spaziale esplicita**.

## 3. VLM in Healthcare (13)

### 3.1 Evoluzione dell'AI medica (13, pag. 5)

| Era | Approccio | Modello | Limite |
|---|---|---|---|
| Ieri | Computer Vision (CNN) | Deep Learning | "Muto" (solo classificazione) |
| Oggi | NLP (LLM) | Transformers | "Cieco" (solo testo) |
| Domani | **VLM** | CLIP, LLaVA | "Vede e parla" (multimodale) |

### 3.2 Anatomia di un VLM (13, pag. 6-10)

- **Vision Encoder**: (Transformer-based) converte i pixel in rappresentazioni ad alta dimensionalità.
- **Text Encoder**: converte i token testuali in vettori semantici.
- **Multimodal Alignment** (pag. 9): spazio di embedding condiviso dove immagini e testo vivono nello stesso spazio
  vettoriale (**CLIP**, Radford et al. 2021).
- **Principi chiave** (pag. 10-11): **Visual-Language Alignment** (contrastive learning in CLIP) e **Instruction Tuning**
  (supervised fine-tuning per seguire istruzioni, es. "Describe this X-ray in radiology style").

### 3.3 VLM generali vs Medical VLM (13, pag. 14)

| Feature | GPT-4V (generale) | Med-PaLM M (medicale) |
|---|---|---|
| Dati | Web-scale | Letteratura biomedica, EHR, scans |
| Rischio | Descrizioni superficiali | Insight clinici specifici |
| Sicurezza | Tolleranza errori | Domain specificity essenziale |

Esempio: GPT-4V su chest X-ray → "chest image"; Med-PaLM M → "alveolar opacity in the left lower lobe, suspected pneumonia".

### 3.4 Opportunità cliniche (13, pag. 15-20)

- **Automated Reporting**: bozze strutturate di report (CXR → DICOM SR).
- **Clinical Visual QA**: conversazione con immagini mediche.
- **Doctor-Patient Communication**: traduzione del jargon medico in linguaggio comprensibile.
- **Interventional AI**: assistenza chirurgica in tempo reale (fasi, zone sicure).
- **Grounding & Detection**: localizzazione patologie con bounding box.

### 3.5 Metriche (13, pag. 16)

- **NON usare** solo metriche NLP (BLEU).
- Usare metriche di **accuratezza clinica**: **RadGraph F1** per la correttezza fattuale.

### 3.6 Case study: MMMED (13, pag. 21-28)

- **Dataset**: 582 MCQ da esami di specializzazione spagnola (MIR 2016-2023); lingue: spagnolo, inglese, italiano.

| Modello | Accuracy media |
|---|---|
| Gemini 2.0 Flash (closed) | 72.6% |
| GPT-4o (closed) | 72.0% |
| Qwen2.5-VL-7B (open) | 70.9% |
| LLaVA-v1.5-7B (open) | 36.9% |

- **Passing grade clinico**: >80% → anche i SOTA non sono affidabili per uso autonomo (pag. 28).
- **Language Tax** (pag. 26): i modelli open-source soffrono di cali in lingue non-inglese.
- **Attention Gap** (pag. 27): errori dovuti a failure nel correlare testo e regione corretta dell'immagine.

### 3.7 Case study: neuroscienze – MS Disability Progression (13, pag. 31-32)

Input multimodale: **3D MRI (volumetric) + clinical reports (textual)**. Architettura **Mid-Fusion**:
- Vision Stream: ResNet3D + **LoRA** (r=6, ultimi conv layer).
- Text Stream: RadLLaMA-7b + LoRA (r=2).
- Fusion: **Cross-Attention module** che fonde i token immagine e testo prima del classification head.

### 3.8 Sfide (13, pag. 33-36)

| Sfida | Descrizione |
|---|---|
| **Specialist vs Generalist** | VLM "jack of all trades" → peggio di una CNN specializzata; gap affidabilità/flessibilità |
| **Hallucinations mediche** | Tolleranza errore = 0%; in medicina = malpractice |
| **Bias & Fairness** | Dataset sbilanciati verso popolazioni occidentali; modelli dermatologia falliscono su pelli scure |
| **Black Box** | Attention maps meno precise dei CAM nelle CNN; serve explainability causale |
| **Privacy & Security** | Invio dati a API esterne viola GDPR/HIPAA; servono modelli open-weight (Llama-3, Mistral) per ospedali locali |

### 3.9 Direzioni future (13, pag. 37-39)

- **Grounding & Detection**: bounding box a livello di pixel per verificare che il modello guardi la patologia corretta.
- **PEFT (LoRA)**: Low-Rank Adaptation: freeze del modello principale, si addestrano solo piccoli adapter → uso su GPU consumer.
- **Visual Chain-of-Thought**: guidare il modello passo-passo (identifica organo → cerca opacità → concludi diagnosi)
  riduce le allucinazioni.

## 4. Concetti chiave per l'esame

1. Architettura a 3 livelli VLM + ISEQL e separazione perception/reasoning (12, pag. 4-14).
2. VLM generali vs medici: GPT-4V vs Med-PaLM M (13, pag. 14).
3. Metriche cliniche: RadGraph F1, non solo BLEU (13, pag. 16).
4. MMMED: risultati e language tax / attention gap (13, pag. 21-28).
5. Sfide: hallucination, bias, black box, privacy/GDPR (13, pag. 33-36).
6. Direzioni future: grounding, LoRA/PEFT, visual CoT (13, pag. 37-39).
