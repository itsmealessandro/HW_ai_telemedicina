"""
main.py - Entry Point del Sistema di Telemedicina

Questo è il file principale che avvia il sistema di telemedicina.
Gestisce l'interfaccia utente a linea di comando e coordina tutti i moduli.

Funzionalità principali:
- Menu interattivo per pazienti e medici
- Inserimento parametri vitali
- Visualizzazione storico e alert
- Coordinamento tra agente intelligente, database e notifiche
"""

from database.db_manager import DatabaseManager
from agents.intelligent_agent import IntelligentAgent
from models.vital_parameters import VitalParameters
from utils.notifications import NotificationSystem
import sys
from datetime import datetime


def stampa_menu_principale():
    """Mostra il menu principale del sistema"""
    print("\n" + "="*50)
    print("SISTEMA DI TELEMEDICINA - MONITORAGGIO VITALE")
    print("="*50)
    print("1. Inserisci nuovi parametri vitali (Paziente)")
    print("2. Visualizza storico parametri")
    print("3. Visualizza alert attivi")
    print("4. Area Medico - Visualizza pazienti critici")
    print("5. Statistiche sistema")
    print("0. Esci")
    print("="*50)


def inserisci_parametri(db_manager, agente, notif_system):
    """
    Gestisce l'inserimento di nuovi parametri vitali per un paziente
    
    Args:
        db_manager: Gestore del database
        agente: Agente intelligente per l'analisi
        notif_system: Sistema di notifiche
    """
    print("\n--- INSERIMENTO PARAMETRI VITALI ---")
    
    # Raccolta dati paziente
    paziente_id = input("ID Paziente: ").strip()
    if not paziente_id:
        print("❌ ID Paziente obbligatorio!")
        return
    
    nome = input("Nome Paziente: ").strip()
    
    try:
        # Raccolta parametri vitali
        print("\nInserisci i parametri vitali:")
        pressione_sistolica = float(input("Pressione Sistolica (mmHg) [90-140]: "))
        pressione_diastolica = float(input("Pressione Diastolica (mmHg) [60-90]: "))
        frequenza_cardiaca = float(input("Frequenza Cardiaca (bpm) [60-100]: "))
        temperatura = float(input("Temperatura (°C) [36.0-37.5]: "))
        saturazione_ossigeno = float(input("Saturazione Ossigeno (%) [95-100]: "))
        glicemia = float(input("Glicemia (mg/dL) [70-140]: "))
        
        # Creazione oggetto parametri
        parametri = VitalParameters(
            pressione_sistolica=pressione_sistolica,
            pressione_diastolica=pressione_diastolica,
            frequenza_cardiaca=frequenza_cardiaca,
            temperatura=temperatura,
            saturazione_ossigeno=saturazione_ossigeno,
            glicemia=glicemia
        )
        
        # Analisi tramite agente intelligente
        print("\n🔍 Analisi parametri in corso...")
        analisi = agente.analizza_parametri(parametri)
        
        # Salvataggio nel database
        interazione_id = db_manager.salva_interazione(
            paziente_id=paziente_id,
            nome_paziente=nome,
            parametri=parametri,
            livello_rischio=analisi['livello_rischio'],
            raccomandazioni=analisi['raccomandazioni'],
            allerta_medico=analisi['allerta_medico']
        )
        
        # Mostra risultati al paziente
        print("\n" + "="*50)
        print("📊 RISULTATI ANALISI")
        print("="*50)
        print(f"Livello di Rischio: {analisi['livello_rischio'].upper()}")
        print(f"\nParametri Anomali:")
        if analisi['anomalie']:
            for anomalia in analisi['anomalie']:
                print(f"  ⚠️  {anomalia}")
        else:
            print("  ✅ Tutti i parametri sono nella norma")
        
        print(f"\n💡 Raccomandazioni:")
        print(f"  {analisi['raccomandazioni']}")
        
        # Gestione notifiche
        if analisi['allerta_medico']:
            print("\n🚨 ATTENZIONE: Il medico è stato allertato!")
            notif_system.invia_allerta_medico(
                paziente_id=paziente_id,
                nome_paziente=nome,
                parametri=parametri,
                anomalie=analisi['anomalie']
            )
        
        # Notifica paziente
        notif_system.invia_notifica_paziente(
            paziente_id=paziente_id,
            messaggio=analisi['raccomandazioni'],
            livello_rischio=analisi['livello_rischio']
        )
        
        print(f"\n✅ Interazione salvata con ID: {interazione_id}")
        
    except ValueError as e:
        print(f"❌ Errore: Inserire valori numerici validi! ({e})")
    except Exception as e:
        print(f"❌ Errore durante l'elaborazione: {e}")


def visualizza_storico(db_manager):
    """Visualizza lo storico delle interazioni di un paziente"""
    print("\n--- STORICO PARAMETRI ---")
    paziente_id = input("ID Paziente: ").strip()
    
    storico = db_manager.ottieni_storico_paziente(paziente_id)
    
    if not storico:
        print(f"❌ Nessuna interazione trovata per il paziente {paziente_id}")
        return
    
    print(f"\n📋 Storico per paziente: {storico[0][2]}")
    print("="*80)
    
    for row in storico:
        print(f"\nData: {row[8]}")
        print(f"Livello Rischio: {row[9]}")
        print(f"Pressione: {row[3]}/{row[4]} mmHg")
        print(f"Frequenza Cardiaca: {row[5]} bpm")
        print(f"Temperatura: {row[6]}°C")
        print(f"Saturazione O2: {row[7]}%")
        print(f"Glicemia: {row[11]} mg/dL")
        print(f"Raccomandazioni: {row[10]}")
        print("-"*80)


def visualizza_alert(db_manager):
    """Visualizza tutti gli alert critici attivi"""
    print("\n--- ALERT CRITICI ATTIVI ---")
    
    alert = db_manager.ottieni_alert_critici()
    
    if not alert:
        print("✅ Nessun alert critico attivo")
        return
    
    print(f"\n🚨 Trovati {len(alert)} alert critici:\n")
    
    for row in alert:
        print(f"Paziente: {row[2]} (ID: {row[1]})")
        print(f"Data: {row[8]}")
        print(f"Livello Rischio: {row[9]}")
        print(f"Allerta Medico: {'Sì' if row[11] else 'No'}")
        print(f"Glicemia: {row[11]} mg/dL")
        print(f"Raccomandazioni: {row[10]}")
        print("="*80)


def area_medico(db_manager):
    """Area riservata per i medici"""
    print("\n--- AREA MEDICO ---")
    print("1. Visualizza pazienti con rischio ALTO")
    print("2. Visualizza tutti gli alert")
    print("3. Cerca paziente specifico")
    print("0. Torna al menu principale")
    
    scelta = input("\nScelta: ").strip()
    
    if scelta == "1":
        pazienti_critici = db_manager.ottieni_pazienti_rischio_alto()
        if not pazienti_critici:
            print("✅ Nessun paziente con rischio alto")
            return
        
        print(f"\n⚠️  Pazienti con rischio ALTO: {len(pazienti_critici)}\n")
        for p in pazienti_critici:
            print(f"Paziente: {p[2]} (ID: {p[1]})")
            print(f"Ultima rilevazione: {p[8]}")
            print(f"Raccomandazioni: {p[10]}")
            print("-"*80)
    
    elif scelta == "2":
        visualizza_alert(db_manager)
    
    elif scelta == "3":
        visualizza_storico(db_manager)


def mostra_statistiche(db_manager):
    """Mostra statistiche generali del sistema"""
    print("\n--- STATISTICHE SISTEMA ---")
    stats = db_manager.ottieni_statistiche()
    
    print(f"\n📊 Statistiche Generali:")
    print(f"Totale Interazioni: {stats['totale_interazioni']}")
    print(f"Pazienti Unici: {stats['pazienti_unici']}")
    print(f"Alert Critici Attivi: {stats['alert_critici']}")
    print(f"Pazienti a Rischio Alto: {stats['rischio_alto']}")
    print(f"Pazienti a Rischio Medio: {stats['rischio_medio']}")
    print(f"Pazienti a Rischio Basso: {stats['rischio_basso']}")


def main():
    """Funzione principale del sistema"""
    print("🏥 Inizializzazione Sistema di Telemedicina...")
    
    # Inizializzazione componenti
    db_manager = DatabaseManager()
    agente = IntelligentAgent()
    notif_system = NotificationSystem()
    
    print("✅ Sistema pronto!\n")
    
    # Loop principale
    while True:
        stampa_menu_principale()
        scelta = input("\nInserisci la tua scelta: ").strip()
        
        if scelta == "1":
            inserisci_parametri(db_manager, agente, notif_system)
        elif scelta == "2":
            visualizza_storico(db_manager)
        elif scelta == "3":
            visualizza_alert(db_manager)
        elif scelta == "4":
            area_medico(db_manager)
        elif scelta == "5":
            mostra_statistiche(db_manager)
        elif scelta == "0":
            print("\n👋 Grazie per aver utilizzato il Sistema di Telemedicina!")
            db_manager.chiudi()
            sys.exit(0)
        else:
            print("❌ Scelta non valida!")
        
        input("\nPremi INVIO per continuare...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Sistema terminato dall'utente")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Errore critico: {e}")
        sys.exit(1)
