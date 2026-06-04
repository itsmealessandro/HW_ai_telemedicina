"""
analysis_service.py - Servizio di analisi dei parametri vitali.

Coordina agente intelligente, database e notifiche per eseguire
il flusso completo di analisi, persistenza e comunicazione.
"""

from telemedicina.agents.intelligent_agent import IntelligentAgent
from telemedicina.database.db_manager import DatabaseManager
from telemedicina.utils.notifications import NotificationSystem
from telemedicina.models.vital_parameters import VitalParameters
from telemedicina import config


class AnalysisService:
    """Servizio che orchestra l'analisi completa dei parametri vitali."""

    def __init__(self, db_manager=None, agent=None, notif_system=None, agente_tipo="rule"):
        self.agente_tipo = agente_tipo
        self.db = db_manager or DatabaseManager(config.DB_PATH)
        self.notif = notif_system or NotificationSystem(log_dir=config.LOG_DIR)

        if agente_tipo == "rl":
            self._carica_agente_rl()
        else:
            self.agent = agent or IntelligentAgent()

    def _carica_agente_rl(self):
        from telemedicina.agents.rl_agent import QLearningAgent
        self.agent = QLearningAgent()
        if not self.agent.carica_q_table():
            raise RuntimeError(
                "Q-table non trovata. Esegui il training con --build."
            )
        print("Agente RL caricato.")

    def analizza_e_salva(self, paziente_id, nome_paziente, parametri):
        """Esegue analisi, salvataggio su DB e notifiche."""
        analisi = self.agent.analizza_parametri(parametri)

        interazione_id = self.db.salva_interazione(
            paziente_id=paziente_id,
            nome_paziente=nome_paziente,
            parametri=parametri,
            livello_rischio=analisi['livello_rischio'],
            raccomandazioni=analisi['raccomandazioni'],
            allerta_medico=analisi['allerta_medico']
        )

        if analisi['allerta_medico']:
            self.notif.invia_allerta_medico(
                paziente_id=paziente_id,
                nome_paziente=nome_paziente,
                parametri=parametri,
                anomalie=analisi['anomalie']
            )

        self.notif.invia_notifica_paziente(
            paziente_id=paziente_id,
            messaggio=analisi['raccomandazioni'],
            livello_rischio=analisi['livello_rischio']
        )

        return analisi, interazione_id

    def ottieni_storico(self, paziente_id, limite=10):
        return self.db.ottieni_storico_paziente(paziente_id, limite)

    def ottieni_alert(self):
        return self.db.ottieni_alert_critici()

    def ottieni_pazienti_critici(self):
        return self.db.ottieni_pazienti_rischio_alto()

    def ottieni_statistiche(self):
        return self.db.ottieni_statistiche()

    def chiudi(self):
        self.db.chiudi()
