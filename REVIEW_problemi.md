# Problemi trovati nella review delle modifiche non committate

Data: 2026-08-21
Scope: modifiche non committate (rinomine `docs/latex/ → docs/src/`, `genera_dashboard.py`, nuovo `trace.py`, capitoli `p1_*.tex`, README, Makefile, bibliografia).

Nessun problema blocca il cambiamento. I punti 1–3 sono piccole migliorie di robustezza, il punto 4 è una correzione di una riga alla documentazione.

---

## Bug

### 1. `trace.py` non rispetta il proprio contratto di input — crash su input non-dict
- **Dove:** `supervised_telemedicina/src/telemedicina_supervised/agents/trace.py`, righe 44 e 80
- **Severità:** bassa (latente; non raggiungibile con l'attuale cablaggio)
- **Problema:** la docstring promette supporto per "mapping (dict) o oggetto con i 6 attributi ufficiali", ma la riga 44 usa `parametri.get(k)` e la riga 80 usa `parametri[n]`, che funzionano solo con dict. Un oggetto con attributi (es. stile `VitalParameters`, che `safety_rules` supporta esplicitamente) solleva `AttributeError`.
- **Scenario realistico:** oggi l'unico chiamante passa un dict JSON via HTTP, quindi nulla si rompe a runtime; ma il primo chiamante che passa un oggetto attributo-style riceve un'eccezione non gestita invece di una trace.
- **Fix suggerito:** oppure restringere la docstring ai soli mapping, oppure usare lo stesso accesso `_get()`-style usato da `valida_parametri`/`analizza`.

### 2. Messaggio "affidabile" fuorviante quando la confidenza è null
- **Dove:** `supervised_telemedicina/tools/genera_dashboard.py`, righe 984–990
- **Severità:** bassa (richiede pesi del modello degradati/corrotti)
- **Problema:** `trace.py` calcola la rete di visualizzazione con la propria `forward()`, mentre `agent.predict()` rivalida indipendentemente l'inferenza. Se i controlli di predict falliscono (probabilità non finite, shape errata), l'esito ha `confidenza=None, fallback=True`, ma `trace["soglia_incertezza"]["fallback"]` resta `False` (è definito come `conf is not None and conf < soglia`). Il JS renderizza allora *"Confidenza 0.0000 ≥ soglia 0.6 → affidabile"* (`null → 0`) subito seguito dal badge FALLBACK sulla risposta finale: output contraddittorio.
- **Scenario realistico:** serve un artefatto corrotto/incompatibile che passa la validazione di `_carica()` ma fallisce i controlli numerici di predict. Raro, ma la logica di visualizzazione non dovrebbe stampare uno `0.0000` fabbricato.
- **Fix suggerito:** gestire esplicitamente `confidenza == null` nel JS (mostrare "n/d" invece di `0.0000`).

### 3. Nessuna guardia di rientro sul form interattivo
- **Dove:** `genera_dashboard.py`, sezione JS, `init()` (~riga 1009)
- **Severità:** bassa (da cosmetico a confuso)
- **Problema:** cliccando di nuovo "Valuta" mentre la sequenza animata è in corso (~650 ms × fino a 7 step ≈ 4,5 s), `#racconto` viene svuotato mentre il primo `mostra()` asincrono continua ad aggiungere gli step catturati: le due render si intrecciano producendo output illeggibile.
- **Scenario realistico:** qualsiasi doppio click o re-submit impaziente.
- **Fix suggerito:** flag booleano impostato al submit e azzerato al completamento di `mostra`.

---

## Documentazione

### 4. Il registro delle modifiche indica dove sono state aggiunte le citazioni in modo errato
- **Dove:** `docs/src/documento.tex`, voce 5 del changelog
- **Severità:** minore (accuratezza della documentazione)
- **Problema:** il changelog dichiara inserzioni `\cite{}` nei capitoli 1, 2, 3, 5, 6, 7, 8. In realtà le citazioni esistono solo in 01, 03, 05, 07, 08; `02_fondamenti_ml.tex` e `06_training.tex` sono rinomine pure senza variazioni di contenuto.
- **Fix suggerito:** correggere l'elenco dei capitoli nella voce del registro (documento esplicitamente dedicato alla tracciabilità delle modifiche).

### 5. `\label{cap:changelog}` dopo `\chapter*`
- **Dove:** `docs/src/documento.tex`
- **Severità:** banale
- **Problema:** le label sui capitoli con stella risolvono contro l'ultimo contatore numerato, quindi un eventuale `\ref{cap:changelog}` stamperebbe un numero sbagliato.
- **Stato:** oggi nessun riferimento lo usa; è solo una trappola per usi futuri.

---

## Note di struttura / convenzioni (non bug)

- **Accoppiamento a API privata:** `trace.py:77` chiama `agent._carica()`. Stesso package e specchia gli interni di `predict`, quindi accettabile, ma lega la trace a un metodo privato; un accessor pubblico sarebbe più pulito se l'agente evolvesse.
- **Lavoro duplicato per richiesta:** ogni `/api/valuta` esegue `analizza()` due volte (una diretta, una dentro `predict`) e carica l'artefatto MLP due volte (`_carica()` non fa caching se il modello non è iniettato alla costruzione). Innocuo per un tool demo su localhost; da sapere se l'endpoint venisse riutilizzato.
- **Copertura di test mancante:** `tests/test_dashboard.py` copre `GET /api/analisi` e il ciclo di vita di `_serve`, ma nulla esercita `POST /api/valuta` né `trace_analysis`. Vista la disciplina di test del resto del progetto, è un gap.
- **Nit di hardening:** `do_POST` legge tutto ciò che `Content-Length` dichiara senza limite di dimensione, e le risposte d'errore includono `str(exc)` (coerente col pattern pre-esistente di `_api_analisi`). Il binding su 127.0.0.1 rende entrambi accettabili qui.

---

## Cambiamenti di comportamento (intenzionali, segnalati per consapevolezza)

- `--serve` ora costruisce un `SupervisedAgent` all'avvio ed espone `POST /api/valuta`. La costruzione è economica (il caricamento del modello resta lazy in `_carica`): nessuna regressione di avvio; i test esistenti che chiamano `crea_handler(db_path, dir)` con due argomenti restano compatibili.
- Le build statiche (non-serve) ora renderizzano la tab 7 con un form il cui fetch fallisce gracefully verso il messaggio "serve attivo": buona degradazione, coerente col design del tool.

---

## Verificato pulito

- La ristrutturazione LaTeX è sana: classe `report` supporta `\part`/`\chapter*`; `tikz` caricato; `\codice`/`\file`/`esempio` definiti; tutte le nuove label/ref dei `p1_*` risolvono; il percorso bib `bib/bibliografia` combacia con la directory di esecuzione bibtex del Makefile; la build produce `docs/documento.pdf` come documentato. **Build eseguita e verificata: exit 0, nessun warning di citazioni/riferimenti.**
- `.gitignore` (`docs/*.pdf`, `docs/src/*.pdf`) combacia col nuovo layout di output del Makefile e con la policy di ignore precedente.
- Il nuovo JS escapezza ogni stringa server-side tramite `esc()` prima dell'interpolazione: nessun percorso XSS nel nuovo pannello; i valori `RANGE_FISIOLOGICI` interpolati negli attributi HTML sono costanti numeriche.
- I valori stringa dai campi vuoti del form (`leggiForm` invia `''` quando `parseFloat` fallisce) fluiscono in `valida_parametri`, che restituisce errori strutturati "valore non numerico" invece di sollevare eccezioni: il percorso input malformato degrada correttamente end-to-end.
- `mlp.MLP.forward()` restituisce esattamente le chiavi `z1/a1/z2/probs` consumate da `trace.py`; l'uso dello scaler combacia con `predict`.
