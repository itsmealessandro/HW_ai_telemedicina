# Documentazione tecnica — supervised_telemedicina

Knowledge distillation di un teacher rule-based in un MLP numpy, con safety
gate deterministico a precedenza assoluta, persistenza SQLite e notifiche
per casi critici. **Stato: Fasi 0-7 complete** (109/109 test, regressione
legacy 14/14).

Vincoli di progetto: solo numpy + standard library (nessuna dipendenza
nuova); tutte le soglie cliniche vivono SOLO in
`src/telemedicina_supervised/safety/safety_rules.py` (fonte unica, guard
test `tests/test_single_source.py`).

## 1. Pipeline

```
teacher rule-based (safety_rules + training/teacher_rules.py)
        │  genera le label
        ▼
dataset sintetico congelato (training/synthetic_generator.py, seed 41/42/43)
        │  split train/val/test mai ri-generati
        ▼
MLP numpy (src/telemedicina_supervised/ml/) — grid 6 config, scelta su
        │  min(loss_val), riaddestramento finale, scaler salvato nell'artifact
        ▼
SupervisedAgent — safety gate (precedenza ASSOLUTA) → MLP → soglia di
        │  incertezza → regola max(rule-based, MLP)
        ▼
AnalysisService.analizza() — validazione → agente → esito
        │
        ├──► SQLite (tabella analisi) + notifiche per casi critici
        └──► storico in memoria (compatibilità)
```

La CLI (`main.py`) è un thin wrapper su `AnalysisService.analizza()`:
percorso applicativo unico, così ogni flusso futuro (API, webapp) eredita
persistenza e notifiche per costruzione.

## 2. Schema del database (SQLite, stdlib)

File: `data/analisi.db` (default, configurabile con `--db-path` o
`config.DB_PATH`). Creato automaticamente alla prima analisi (schema lazy
init). Implementazione: `src/telemedicina_supervised/database/analisi_db.py`.

### Tabella `analisi`

| Campo | Tipo | Note |
|---|---|---|
| `id` | INTEGER PK AUTOINCREMENT | |
| `timestamp` | TEXT | ISO-8601 UTC |
| `classe` | TEXT | `basso` \| `medio` \| `alto` \| `errore` |
| `allerta_medico` | INTEGER | 0/1: richiede attenzione medica |
| `errori` | TEXT | JSON (lista di stringhe; `[]` se valido) |
| `probabilita` | TEXT | JSON (dict classe→float) oppure NULL se l'MLP non è stato usato (gate o fallback) |
| `metadati` | TEXT | JSON: `modello_usato`, `classe_mlp`, `classe_regola`, `confidenza`, `fallback`, `motivo_fallback`, `override_sicurezza` |
| `messaggio` | TEXT | messaggio per il paziente |

### Tabella `notifiche`

| Campo | Tipo | Note |
|---|---|---|
| `id` | INTEGER PK AUTOINCREMENT | |
| `timestamp` | TEXT | ISO-8601 UTC (stesso dell'analisi) |
| `classe` | TEXT | classe che ha generato l'allerta |
| `messaggio` | TEXT | testo della notifica |

Una notifica viene registrata quando `allerta_medico=True` (classe `alto` o
input invalido), su due canali: riga in `notifiche` + stampa su stderr con
timestamp (`[NOTIFICA] ...`). La notifica parte dal **service**, mai dalla
CLI.

**Fallimento rumoroso**: se il database non è scrivibile,
`AnalysisService.analizza()` solleva `ValueError` con messaggio chiaro —
un record clinico non va mai perso in silenzio. La CLI lo traduce in un
errore su stderr con exit code 1.

## 3. Design decisions chiave

1. **Soglia glicemia critica unificata a ≤ 60 mg/dL.** Il legacy aveva una
   discrepanza (teacher `<= 60`, override RL `<= 55`): scelta unificata a 60,
   la più conservativa e coerente con le label del teacher (documentata in
   `safety_rules.py`).
2. **Safety gate a precedenza assoluta.** Un caso critico per le regole non
   passa mai dall'MLP: il recall di sistema sui critici è 1.0 per costruzione.
3. **Zona di incertezza.** Se `max(probabilità softmax) < 0.6`
   (`SOGLIA_INCERTEZZA`), fallback al rule-based: il sistema è conservativo
   dove il modello è debole (boundary).
4. **Regola `max(rule-based, MLP)`.** La classe finale è la più alta tra
   quella delle regole e quella dell'MLP: il modello può solo alzare la
   classe, mai abbassarla. Prezzo intenzionale: falsi positivi in più
   rispetto al teacher (vedi review finale).
5. **Persistenza rumorosa.** DB non scrivibile → `ValueError`, mai
   degradazione silenziosa.
6. **Fonte unica delle soglie.** `safety_rules.py` è l'unico posto in cui
   vivono range, soglie e pattern; il guard test `test_single_source.py`
   impedisce duplicazioni future.

## 4. Riproducibilità

- Seed base **41** (default, configurabile); split congelati con seed
  derivati (41/42/43); `metadati.json` con hash di provenienza.
- Selezione della configurazione su `min(loss_val)` (mai sul test); test
  valutato UNA sola volta.
- Report deterministico in `data/models/report.json`.

## 5. Risultati (report.json, seed 41)

| Metrica | Valore |
|---|---|
| Accuracy test congelato | **0.9851** |
| Accuracy slice no-buffer | **0.9324** |
| Kappa vs teacher | **0.9775** |
| Recall `alto` (MLP) | 0.9958 (30 mancati su 7100) |
| Errori su 20000 | 298 (distanza media 0.524 dalle soglie) |
| Baseline teacher | 1.0 (upper bound di imitazione) |
| Config migliore | hidden 32, lr 0.05, batch 32, epoche 25 |

Interpretazione: l'accuracy misura la **fedeltà al teacher**, non la
validità clinica su pazienti reali (le label sono generate da regole su
dati sintetici). Vedi `docs/review_finale.md` per i limiti dichiarati.