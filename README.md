# Sistema di Telemedicina con Agente Intelligente 🏥

Sistema completo per il monitoraggio remoto dei parametri vitali con analisi intelligente e notifiche automatiche.

## 🎯 Caratteristiche Principali

- ✅ **Monitoraggio Parametri Vitali**: Pressione, frequenza cardiaca, temperatura, saturazione ossigeno, glicemia
- 🤖 **Agente Intelligente**: Analisi automatica con valutazione rischio (BASSO/MEDIO/ALTO)
- 📊 **Database SQLite**: Storico completo di tutte le interazioni
- 🔔 **Sistema Notifiche**: Alert automatici a paziente e medico
- 📈 **Statistiche**: Dashboard con metriche aggregate
- 👨‍⚕️ **Area Medico**: Visualizzazione pazienti critici

## 🚀 Quick Start

### Requisiti
- Python 3.8+
- Nessuna libreria esterna richiesta (solo standard library)

### Installazione

```bash
# 1. Scarica/clona il progetto
cd progetto_telemedicina

# 2. Verifica struttura file
progetto_telemedicina/
├── main.py
├── DOCUMENTAZIONE.md
├── README.md
├── models/
│   ├── __init__.py
│   └── vital_parameters.py
├── agents/
│   ├── __init__.py
│   └── intelligent_agent.py
├── database/
│   ├── __init__.py
│   └── db_manager.py
└── utils/
    ├── __init__.py
    └── notifications.py

# 3. Esegui il sistema
python main.py
```

### Primo Utilizzo

1. **Seleziona opzione 1** - Inserisci nuovi parametri vitali
2. **Inserisci ID paziente**: es. `P001`
3. **Inserisci nome**: es. `Mario Rossi`
4. **Inserisci parametri vitali** (valori esempio normali):
   - Pressione Sistolica: `120`
   - Pressione Diastolica: `80`
   - Frequenza Cardiaca: `75`
   - Temperatura: `36.8`
   - Saturazione Ossigeno: `98`
   - Glicemia: `95`

5. **Ricevi analisi immediata** con:
   - Livello di rischio
   - Anomalie rilevate
   - Raccomandazioni personalizzate

## 📖 Documentazione Completa

Per una comprensione approfondita del sistema, consulta **`DOCUMENTAZIONE.md`** che include:

- Architettura dettagliata del sistema
- Spiegazione di ogni componente
- Rationale delle scelte tecniche
- Flussi di funzionamento
- Design pattern utilizzati
- Estensioni future (API REST, ML, App Mobile)
- Compliance GDPR e certificazioni mediche

## 🎨 Struttura del Codice

### main.py
Entry point del sistema con interfaccia CLI. Gestisce menu interattivo e coordina tutti i componenti.

### models/vital_parameters.py
Definisce il modello dati per i parametri vitali con:
- Range di normalità basati su linee guida mediche
- Validazione a due livelli (fisiologica + clinica)
- Identificazione automatica anomalie con gravità

### agents/intelligent_agent.py
Agente intelligente basato su regole che:
- Analizza parametri individuali
- Identifica correlazioni patologiche (shock, crisi ipertensiva, sepsi)
- Calcola rischio globale
- Genera raccomandazioni personalizzate
- Decide quando allertare il medico

### database/db_manager.py
Gestore database SQLite con:
- Schema ottimizzato con indici
- Query per storico, alert, statistiche
- Supporto per trend analysis

### utils/notifications.py
Sistema notifiche che gestisce:
- Notifiche paziente (raccomandazioni)
- Alert medico (emergenze)
- Log tracciabilità completa

## 🔍 Esempi di Utilizzo

### Parametri Normali
```
Input: 120/80 mmHg, 75 bpm, 36.8°C, 98%, 95 mg/dL
Output: 
  ✅ Rischio BASSO
  ✅ Tutti i parametri nella norma
  ✅ Continuare monitoraggio regolare
```

### Parametri Critici
```
Input: 190/115 mmHg, 125 bpm, 36.8°C, 98%, 95 mg/dL
Output:
  🚨 Rischio ALTO
  ⚠️ CRISI IPERTENSIVA rilevata
  🚨 Medico ALLERTATO automaticamente
  ⚠️ Recarsi immediatamente al pronto soccorso
```

## 🧪 Testing

Il sistema include logica di validazione robusta:
- Validazione input (range fisiologici)
- Gestione errori con messaggi chiari
- Log completo per debugging
- Pattern di test inclusi in documentazione

## 🔐 Sicurezza e Privacy

- Database locale (no cloud di default)
- Log separati per paziente/medico
- Struttura pronta per crittografia
- Compliance GDPR (vedi documentazione)

## 🛣️ Roadmap

### ✅ Fase 1 - MVP (Completata)
- Sistema CLI funzionale
- Agente intelligente con regole
- Database SQLite con storico
- Notifiche simulate

### 🚧 Fase 2 - Production (In pianificazione)
- API REST per integrazione web/mobile
- Autenticazione e autorizzazione
- Notifiche email/SMS reali
- Deploy cloud (AWS/Azure)

### 📅 Fase 3 - Enterprise
- Dashboard web React
- App mobile nativa
- Machine Learning per predizione rischio
- Integrazione wearables IoT

## 📚 Risorse

### Linee Guida Mediche
- American Heart Association - Pressione arteriosa
- European Society of Cardiology - Frequenza cardiaca
- WHO - Temperatura corporea
- American Diabetes Association - Glicemia

### Standard Tecnici
- HL7 FHIR per interoperabilità healthcare
- ISO 13485 per dispositivi medici
- GDPR per protezione dati

## ❓ FAQ

**Q: Posso usare questo sistema in produzione?**
A: Questo è un prototipo educativo. Per produzione serve: certificazione medical device, compliance regolamentare, infrastruttura cloud enterprise.

**Q: Come aggiungo nuovi parametri vitali?**
A: Modifica `VitalParameters` in `models/vital_parameters.py`, aggiungi regole in `IntelligentAgent`, aggiorna schema database.

**Q: Supporta più medici/pazienti?**
A: Sì, il database supporta N pazienti. Per multi-medico serve aggiungere autenticazione e ruoli.

**Q: Posso integrare con dispositivi IoT?**
A: Struttura modulare facilita integrazione. Vedi sezione "Integrazione Dispositivi IoT" in DOCUMENTAZIONE.md.

## 🤝 Contributi

Questo è un progetto didattico. Per miglioramenti:
1. Studia DOCUMENTAZIONE.md per comprendere architettura
2. Segui pattern esistenti per coerenza
3. Documenta ogni modifica
4. Testa approfonditamente

## 📄 Licenza

[Inserire licenza appropriata - es. MIT, Apache 2.0]

## 👥 Autori

Sistema di Telemedicina - Progetto Universitario
[Inserire nomi autori/team]

## 📞 Supporto

- 🐛 Bug/Issue: [GitHub Issues]
- 💬 Domande: [Email supporto]
- 📖 Docs: DOCUMENTAZIONE.md

---

**⚠️ DISCLAIMER MEDICO**: Questo sistema è un prototipo educativo. NON sostituisce consulto medico professionale. Per emergenze mediche chiamare il 118.
