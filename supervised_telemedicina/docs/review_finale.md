# Review informale finale — supervised_telemedicina

**Data:** 2026-08-19
**Stato:** Fasi 0-7 complete (Fase 8 = documentazione e review).
**Suite:** 109/109 test OK · regressione legacy 14/14 · `git diff --check` pulito.

## 1. Risultati quantitativi (da `data/models/report.json`, seed 41)

| Metrica | Valore |
|---|---|
| Accuracy test congelato | **0.9851** |
| Accuracy slice no-buffer | **0.9324** |
| Kappa vs teacher (test congelato) | **0.9775** |
| Recall `alto` (MLP, test congelato) | **0.9958** (30 mancati su 7100) |
| Precision/recall/F1 per classe | basso 0.9912/0.9852/0.9882 · medio 0.9758/0.9717/0.9737 · alto 0.9863/0.9958/0.9910 |
| Errori su 20000 | 298 (distanza media 0.524 dalle soglie, max 0.681) |
| Baseline rule-based (teacher) | 1.0 (upper bound di imitazione) |
| Config migliore | hidden 32, lr 0.05, batch 32, epoche 25 (loss_val 0.0510) |
| Distribuzione test | basso 7220 · medio 5680 · alto 7100 |

**Proprietà di sicurezza verificata nel codice**: il safety gate ha
precedenza assoluta — un caso `alto` per le regole non passa mai dall'MLP.
I 30 mancati dell'MLP non producono alcun declassamento di sistema: il
recall di sistema sui critici è **1.0 per costruzione**.

## 2. Limiti dichiarati (obbligatori — nessuna pretesa diagnostica)

1. **Label sintetiche.** Le label sono generate dal teacher rule-based su
   dati SINTETICI: l'accuracy misura la fedeltà al teacher (knowledge
   distillation), NON la validità clinica su pazienti reali. Il sistema non
   è un dispositivo medico e non va usato come tale.
2. **Test bufferizzato.** Il test congelato è generato con buffer zone: le
   metriche su di esso (0.9851) sono ottimistiche sui casi near-threshold.
   Lo slice no-buffer (0.9324) è la stima più onesta della generalizzazione
   sui boundary.
3. **Falsi positivi intenzionali.** La regola `max(rule-based, MLP)` implica
   che il sistema integrato ha falsi positivi in più rispetto al teacher
   (l'MLP può solo alzare la classe, mai abbassarla). È il prezzo scelto
   della conservatività: meglio un allarme in più che un caso critico
   declassato.
4. **Edge case noto sulla notifica.** Se l'insert della notifica fallisce
   dopo l'insert dell'analisi (es. errore SQLite a metà transazione), il
   record dell'analisi c'è ma la notifica è persa. Accettato per il
   progetto didattico (SQLite locale, operazioni singole); una transazione
   atomica su entrambe le tabelle è il miglioramento naturale.
5. **Thread-safety non garantita.** L'uso dichiarato è single-thread (CLI o
   un processo per servizio); una connessione SQLite per istanza.
6. **Nessuna webapp.** Decisione presa in Fase 6: la webapp non è nel piano;
   la CLI è il canale applicativo del progetto.

## 3. Prossimi passi possibili (opzionali, fuori piano)

- **Webapp** (come il progetto legacy) sopra `AnalysisService`: erediterebbe
  persistenza e notifiche per costruzione, senza toccare il contratto.
- **Più profili sintetici** nel generatore (es. pazienti pediatrici,
  comorbidità) per ampliare la copertura del dataset.
- **Validazione su dati reali** (es. dataset pubblici di parametri vitali)
  per misurare la distanza tra fedeltà al teacher e validità clinica.
- **Transazione atomica** analisi+notifica e **test multi-thread** se il
  sistema diventasse concorrente.

## 4. Conclusione

Il progetto è completo rispetto al piano: distillation riproducibile (seed
41), safety gate verificato nel codice, persistenza e notifiche su un unico
percorso applicativo, 109 test e regressione legacy verde. I limiti sopra
sono dichiarati e coerenti con la natura didattica del lavoro.