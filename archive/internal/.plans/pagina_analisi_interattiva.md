# Piano — Pagina "Analisi interattiva"

## Obiettivo
Aggiungere alla dashboard (`tools/genera_dashboard.py`) una 7ª tab "Analisi
interattiva" con un form per i 6 parametri vitali. L'utente invia un campione;
il backend lo valida (casi esagerati/invalidi) e, se plausibile, restituisce
una **traccia completa** della pipeline che il frontend rivela **a step**
(animazione) per spiegare cosa succede "dentro la rete" fino alla risposta.

## Architettura
- Il backend calcola tutto in una chiamata e ritorna un JSON di traccia; il
  frontend lo *anima* progressivamente (compute-once, reveal-after).
- Vincolo: runtime numpy + stdlib; server HTTP = `http.server` della stdlib.
  Nessuna nuova dipendenza.

## Componenti

### 1. `src/telemedicina_supervised/agents/trace.py` (NUOVO)
Funzione `trace_analysis(parametri, agent=None) -> dict` che specchia
`SupervisedAgent.predict` ma cattura i valori intermedi:

- `input`: {feature: valore} nell'ordine `FEATURE_ORDER`.
- `validazione`: {valido, errori, range_fisiologici} — usa `valida_parametri`
  (fonte unica in `safety_rules`). Se `!valido` -> `stopped_at="validazione"`,
  `risposta` = errore, nessun passo successivo.
- `safety_gate`: {classe_regola, anomalie, pattern} da `analizza()`.
  Se `classe_regola == "alto"` -> `stopped_at="gate"` (MLP bypassato, badge
  GATE), `risposta` = alto, nessun passo rete.
- `rete` (solo se non critico): tramite `agent._carica()` per ottenere il
  modello gia' caricato, poi:
  - `input_normalizzato` = `modello.scaler.transform(vettore)` (6 float)
  - `z1` = pre-attivazione hidden (`vettore_norm @ W1.T + b1`)
  - `a1` = ReLU(z1)
  - `z2` = logits (`a1 @ W2.T + b2`)
  - `probabilita` = softmax(z2) -> {basso, medio, alto}
  (tutti da `modello.forward(normalizzato)`)
- `soglia_incertezza`: {soglia: SOGLIA_INCERTEZZA, confidenza, superata,
  fallback} — `fallback=True` se confidenza < soglia.
- `risposta`: {classe, messaggio, allerta_medico, percorso} — il risultato
  finale autoritativo e' preso da `agent.predict(parametri)` (garantisce
  coerenza con il resto del sistema); i campi di `rete` servono solo per la
  visualizzazione. `percorso` in {"gate","mlp","fallback","regole","errore"}.

Persistenza: **NO** (usa `SupervisedAgent` diretto, non `AnalysisService`) per
non inquinare il tab "Analisi Live" / notifiche con input demo.

### 2. Endpoint `POST /api/valuta` in `tools/genera_dashboard.py`
In `crea_handler`: aggiungere `do_POST` che:
- legge il body JSON `{parametri: {...}}`;
- chiama `trace_analysis(parametri)` (istanza `SupervisedAgent` creata una
  volta e passata al Handler, per cache del modello);
- ritorna il dict come JSON (`Content-Type: application/json; charset=utf-8`);
- in caso di errore -> 500 con `{"errore": ...}`.
`GET /api/analisi` resta invariato (read-only, tab Live).

### 3. Frontend (7ª tab) in `tools/genera_dashboard.py`
- Aggiungere `("interattiva", "7. Analisi interattiva")` a `TAB`.
- CSS: riusare classi esistenti (`card`, `nodo`, `barre-prob`, `scala-confidenza`,
  `badge-*`, `metriche-*`). Aggiungere solo cio' che serve all'animazione
  (es. `.step-hidden`, `.step-visibile`, transizione opacita').
- HTML (`genera_html`): pannello con form a 6 campi (placeholder = range
  fisiologici come hint) + pulsante "Valuta" + area "racconto passo-passo".
- JS: al submit, `fetch('POST','/api/valuta')`; poi rivela i nodi **uno alla
  volta** con ritardo (async/await + setTimeout) nell'ordine:
  validazione -> safety gate -> (rete: normalizzazione -> hidden z1/a1 ->
  logits z2 -> softmax -> confidenza sulla scala -> soglia) -> risposta.
  Se `stopped_at` e' "validazione" mostra gli errori e si ferma; se "gate"
  mostra il bypass critico e si ferma prima della rete. Badge coerenti.

## Sequenza / Lane
- Fase 1 (backend, `@fixer`): `agents/trace.py` + endpoint `POST /api/valuta`.
- Fase 2 (frontend, `@designer`): 7ª tab, form, animazione step-by-step.
- Fase 3 (verifica): `python tools/genera_dashboard.py --serve`; curl di un
  campione plausibile e di uno esagerato; `python -m unittest discover -s tests`
  deve restare verde.

## Decisioni (da confermare con l'utente)
- Persistenza DB: no-persist (consigliato) vs persisti come le altre analisi.
- Nome endpoint: `/api/valuta` (POST) per distinguerlo dal GET di lettura.
- Rete: mostrare normalizzato + z1/a1/z2/probs (leggibili); nascosti
  eventualmente riassunti se `n_hidden` e' grande.
