"""
telemedicina/utils/notifications.py - Sistema di Notifiche

Questo modulo gestisce l'invio di notifiche a pazienti e medici.
In questa implementazione, le notifiche sono simulate con:
- Log su file per tracciabilità
- Print su console per feedback immediato

In un sistema production, questo modulo integrerebbe:
- Email (SMTP)
- SMS (Twilio, AWS SNS)
- Push notifications (Firebase, OneSignal)
- Messaggistica app mobile
- Alert su dashboard web real-time (WebSocket)

Architettura modulare permette facile estensione senza modificare altri componenti.
"""

from datetime import datetime
from typing import Optional
from telemedicina.models.vital_parameters import VitalParameters
from telemedicina import config
import os


class NotificationSystem:
    """
    Sistema centralizzato per la gestione delle notifiche.
    
    Implementa pattern Strategy per diversi canali di notifica.
    Attualmente simula notifiche con log, ma struttura permette
    facile integrazione con servizi reali.
    """
    
    def __init__(self, log_dir: str = "logs"):
        """
        Inizializza il sistema di notifiche.
        
        Args:
            log_dir: Directory per file di log notifiche
        """
        self.log_dir = log_dir
        self._crea_directory_log()
        
        # File log separati per tipo notifica
        self.log_paziente = os.path.join(log_dir, "notifiche_pazienti.log")
        self.log_medico = os.path.join(log_dir, "alert_medici.log")
        
        print(f"✅ Sistema notifiche inizializzato (log: {log_dir})")
    
    def _crea_directory_log(self):
        """Crea directory log se non esistente"""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
    
    def invia_notifica_paziente(
        self,
        paziente_id: str,
        messaggio: str,
        livello_rischio: str
    ):
        """
        Invia notifica al paziente con raccomandazioni.
        
        In production:
        - Invierebbe email/SMS al paziente
        - Push notification su app mobile
        - Messaggio in-app quando paziente accede
        
        Args:
            paziente_id: ID paziente destinatario
            messaggio: Testo raccomandazioni
            livello_rischio: Livello rischio per priorità
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Determina priorità notifica
        priorita = self._calcola_priorita_notifica(livello_rischio)
        
        # Formatta messaggio
        notifica = self._formatta_notifica_paziente(
            paziente_id,
            messaggio,
            livello_rischio,
            priorita,
            timestamp
        )
        
        # Simula invio (in production: chiamata API email/SMS)
        self._log_notifica(self.log_paziente, notifica)
        
        # Feedback console
        print(f"\n📱 NOTIFICA PAZIENTE [{paziente_id}]:")
        print(f"   Priorità: {priorita}")
        print(f"   {messaggio[:100]}...")
    
    def invia_allerta_medico(
        self,
        paziente_id: str,
        nome_paziente: str,
        parametri: VitalParameters,
        anomalie: list
    ):
        """
        Invia allerta critica al medico responsabile.
        
        In production:
        - Email urgente al medico
        - SMS per emergenze
        - Alert su dashboard medico
        - Notifica push su app medico
        - Possibile escalation automatica
        
        Args:
            paziente_id: ID paziente
            nome_paziente: Nome paziente
            parametri: Parametri vitali misurati
            anomalie: Lista anomalie rilevate
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Formatta allerta dettagliata per medico
        allerta = self._formatta_allerta_medico(
            paziente_id,
            nome_paziente,
            parametri,
            anomalie,
            timestamp
        )
        
        # Simula invio urgente (in production: email + SMS)
        self._log_notifica(self.log_medico, allerta)
        
        # Feedback console con evidenza
        print("\n" + "="*60)
        print("🚨 ALLERTA MEDICO - INTERVENTO RICHIESTO")
        print("="*60)
        print(f"Paziente: {nome_paziente} (ID: {paziente_id})")
        print(f"Data/Ora: {timestamp}")
        print(f"\nAnomalie rilevate: {len(anomalie)}")
        for i, anomalia in enumerate(anomalie[:3], 1):  # Mostra prime 3
            print(f"  {i}. {anomalia}")
        print("="*60)
    
    def _calcola_priorita_notifica(self, livello_rischio: str) -> str:
        """
        Calcola priorità della notifica in base al rischio.
        
        Priorità determina:
        - Velocità invio
        - Canali utilizzati
        - Persistenza notifica
        
        Args:
            livello_rischio: 'basso', 'medio', 'alto'
        
        Returns:
            str: 'BASSA', 'MEDIA', 'ALTA', 'CRITICA'
        """
        mapping = {
            'basso': 'BASSA',
            'medio': 'MEDIA',
            'alto': 'ALTA',
            'critico': 'CRITICA'
        }
        return mapping.get(livello_rischio, 'MEDIA')
    
    def _formatta_notifica_paziente(
        self,
        paziente_id: str,
        messaggio: str,
        livello_rischio: str,
        priorita: str,
        timestamp: str
    ) -> str:
        """
        Formatta notifica per il paziente in formato strutturato.
        
        Args:
            paziente_id: ID paziente
            messaggio: Testo raccomandazioni
            livello_rischio: Livello rischio
            priorita: Priorità calcolata
            timestamp: Data/ora
        
        Returns:
            str: Notifica formattata
        """
        return (
            f"\n{'='*80}\n"
            f"NOTIFICA PAZIENTE\n"
            f"{'='*80}\n"
            f"Timestamp: {timestamp}\n"
            f"Paziente ID: {paziente_id}\n"
            f"Livello Rischio: {livello_rischio.upper()}\n"
            f"Priorità: {priorita}\n"
            f"{'-'*80}\n"
            f"MESSAGGIO:\n"
            f"{messaggio}\n"
            f"{'='*80}\n"
        )
    
    def _formatta_allerta_medico(
        self,
        paziente_id: str,
        nome_paziente: str,
        parametri: VitalParameters,
        anomalie: list,
        timestamp: str
    ) -> str:
        """
        Formatta allerta dettagliata per il medico.
        
        Include tutti i dettagli necessari per valutazione rapida:
        - Identificazione paziente
        - Tutti i parametri vitali
        - Anomalie specifiche
        - Timestamp preciso
        
        Args:
            paziente_id: ID paziente
            nome_paziente: Nome paziente
            parametri: Parametri vitali completi
            anomalie: Lista anomalie
            timestamp: Data/ora
        
        Returns:
            str: Allerta formattata per medico
        """
        anomalie_str = "\n".join([f"   - {a}" for a in anomalie])
        
        return (
            f"\n{'='*80}\n"
            f"🚨 ALLERTA MEDICO - PARAMETRI CRITICI\n"
            f"{'='*80}\n"
            f"Timestamp: {timestamp}\n"
            f"Paziente: {nome_paziente}\n"
            f"ID Paziente: {paziente_id}\n"
            f"{'-'*80}\n"
            f"PARAMETRI VITALI RILEVATI:\n"
            f"   Pressione: {parametri.pressione_sistolica}/{parametri.pressione_diastolica} mmHg\n"
            f"   Frequenza Cardiaca: {parametri.frequenza_cardiaca} bpm\n"
            f"   Temperatura: {parametri.temperatura}°C\n"
            f"   Saturazione O2: {parametri.saturazione_ossigeno}%\n"
            f"   Glicemia: {parametri.glicemia} mg/dL\n"
            f"{'-'*80}\n"
            f"ANOMALIE RILEVATE:\n"
            f"{anomalie_str}\n"
            f"{'-'*80}\n"
            f"AZIONE RICHIESTA: Valutazione clinica urgente\n"
            f"{'='*80}\n"
        )
    
    def _log_notifica(self, log_file: str, contenuto: str):
        """
        Salva notifica su file log per tracciabilità.
        
        Importante per:
        - Audit trail
        - Conformità normative (GDPR, HIPAA)
        - Debug e troubleshooting
        - Analisi storiche
        
        Args:
            log_file: Percorso file log
            contenuto: Contenuto da loggare
        """
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(contenuto)
                f.write("\n")
        except IOError as e:
            print(f"⚠️  Errore scrittura log notifiche: {e}")
    
    def invia_promemoria_monitoraggio(self, paziente_id: str, nome_paziente: str):
        """
        Invia promemoria per monitoraggio regolare.
        
        Usato per:
        - Recall pazienti con monitoraggio programmato
        - Compliance terapeutica
        - Follow-up automatico
        
        Args:
            paziente_id: ID paziente
            nome_paziente: Nome paziente
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        messaggio = (
            f"Gentile {nome_paziente},\n\n"
            f"È il momento di effettuare la tua misurazione dei parametri vitali "
            f"come da programma di monitoraggio.\n\n"
            f"Ti ricordiamo di:\n"
            f"- Essere a riposo da almeno 5 minuti\n"
            f"- Effettuare la misurazione nello stesso momento della giornata\n"
            f"- Registrare tutti i parametri richiesti\n\n"
            f"Il tuo team sanitario"
        )
        
        notifica = (
            f"\n{'='*80}\n"
            f"PROMEMORIA MONITORAGGIO\n"
            f"{'='*80}\n"
            f"Timestamp: {timestamp}\n"
            f"Paziente: {nome_paziente} (ID: {paziente_id})\n"
            f"{'-'*80}\n"
            f"{messaggio}\n"
            f"{'='*80}\n"
        )
        
        self._log_notifica(self.log_paziente, notifica)
        print(f"\n⏰ Promemoria monitoraggio inviato a {nome_paziente}")
    
    def invia_report_periodico(self, medico_id: str, statistiche: dict):
        """
        Invia report periodico al medico con statistiche aggregate.
        
        Report include:
        - Numero pazienti monitorati
        - Alert generati
        - Trend generali
        - Pazienti che richiedono attenzione
        
        Frequenza tipica: giornaliero o settimanale
        
        Args:
            medico_id: ID medico destinatario
            statistiche: Dict con metriche aggregate
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = (
            f"\n{'='*80}\n"
            f"REPORT PERIODICO SISTEMA TELEMEDICINA\n"
            f"{'='*80}\n"
            f"Data/Ora: {timestamp}\n"
            f"Medico ID: {medico_id}\n"
            f"{'-'*80}\n"
            f"STATISTICHE PERIODO:\n"
            f"   Pazienti Monitorati: {statistiche.get('pazienti_unici', 0)}\n"
            f"   Interazioni Totali: {statistiche.get('totale_interazioni', 0)}\n"
            f"   Alert Critici: {statistiche.get('alert_critici', 0)}\n"
            f"   Pazienti Rischio Alto: {statistiche.get('rischio_alto', 0)}\n"
            f"   Pazienti Rischio Medio: {statistiche.get('rischio_medio', 0)}\n"
            f"   Pazienti Rischio Basso: {statistiche.get('rischio_basso', 0)}\n"
            f"{'='*80}\n"
        )
        
        self._log_notifica(self.log_medico, report)
        print(f"\n📊 Report periodico generato per medico {medico_id}")


# Funzioni utility per integrazione futura con servizi esterni

def configura_smtp_email(smtp_server: str, porta: int, username: str, password: str):
    """
    Configura connessione SMTP per invio email.
    
    Da implementare quando si integra servizio email reale.
    
    Args:
        smtp_server: Indirizzo server SMTP
        porta: Porta SMTP (tipicamente 587 per TLS)
        username: Username autenticazione
        password: Password autenticazione
    """
    # TODO: Implementare connessione SMTP
    pass


def configura_twilio_sms(account_sid: str, auth_token: str, numero_mittente: str):
    """
    Configura servizio Twilio per invio SMS.
    
    Da implementare quando si integra servizio SMS.
    
    Args:
        account_sid: Account SID Twilio
        auth_token: Token autenticazione
        numero_mittente: Numero telefono mittente
    """
    # TODO: Implementare integrazione Twilio
    pass


def configura_firebase_push(credentials_path: str):
    """
    Configura Firebase per push notifications.
    
    Da implementare per notifiche push su app mobile.
    
    Args:
        credentials_path: Percorso file credenziali Firebase
    """
    # TODO: Implementare integrazione Firebase
    pass
