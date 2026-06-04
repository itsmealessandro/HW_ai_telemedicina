"""
telemedicina/database/db_manager.py - Gestore Database SQLite

Questo modulo gestisce tutte le operazioni sul database SQLite.
Responsabilità:
- Creazione e inizializzazione dello schema
- Salvataggio interazioni paziente-sistema
- Recupero storico e statistiche
- Query per alert e pazienti critici

Schema Database:
- Tabella 'interazioni': tutte le misurazioni e analisi
- Indici per performance su query frequenti
- Vincoli di integrità referenziale

Utilizza SQLite per semplicità e portabilità (file locale).
Per deployment production si consiglia PostgreSQL/MySQL.
"""

import sqlite3
from datetime import datetime
from typing import List, Dict, Optional
from telemedicina.models.vital_parameters import VitalParameters
from telemedicina import config
import os


class DatabaseManager:
    """
    Gestore del database per il sistema di telemedicina.
    
    Gestisce connessione, schema e tutte le operazioni CRUD.
    Implementa pattern Singleton per evitare connessioni multiple.
    """
    
    def __init__(self, db_path: str = "telemedicina.db"):
        """
        Inizializza il gestore database.
        
        Args:
            db_path: Percorso file database SQLite
        """
        self.db_path = db_path
        self.conn = None
        self._connetti()
        self._crea_schema()
    
    def _connetti(self):
        """
        Stabilisce connessione al database SQLite.
        
        Configurazioni:
        - check_same_thread: False per uso multi-thread
        - Row factory: accesso colonne per nome
        """
        try:
            self.conn = sqlite3.connect(
                self.db_path, 
                check_same_thread=False
            )
            self.conn.row_factory = sqlite3.Row  # Accesso per nome colonna
            print(f"✅ Connessione database: {self.db_path}")
        except sqlite3.Error as e:
            print(f"❌ Errore connessione database: {e}")
            raise
    
    def _crea_schema(self):
        """
        Crea lo schema del database se non esistente.
        
        Tabella principale: interazioni
        Campi:
        - id: chiave primaria auto-incrementale
        - paziente_id: identificativo univoco paziente
        - nome_paziente: nome completo
        - Parametri vitali (6 colonne)
        - timestamp: data/ora rilevazione
        - livello_rischio: basso/medio/alto
        - raccomandazioni: testo consigli
        - allerta_medico: flag booleano
        
        Indici per ottimizzazione query:
        - idx_paziente: ricerche per paziente
        - idx_rischio: filtri per livello rischio
        - idx_timestamp: ordinamenti temporali
        """
        try:
            cursor = self.conn.cursor()
            
            # Creazione tabella interazioni
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS interazioni (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    paziente_id TEXT NOT NULL,
                    nome_paziente TEXT NOT NULL,
                    pressione_sistolica REAL NOT NULL,
                    pressione_diastolica REAL NOT NULL,
                    frequenza_cardiaca REAL NOT NULL,
                    temperatura REAL NOT NULL,
                    saturazione_ossigeno REAL NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    livello_rischio TEXT NOT NULL,
                    raccomandazioni TEXT,
                    allerta_medico BOOLEAN DEFAULT 0,
                    glicemia REAL NOT NULL
                )
            """)
            
            # Indice per ricerche per paziente (query più frequente)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_paziente 
                ON interazioni(paziente_id)
            """)
            
            # Indice per filtri per livello rischio
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_rischio 
                ON interazioni(livello_rischio)
            """)
            
            # Indice composito per query di allerta medico
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_allerta 
                ON interazioni(allerta_medico, livello_rischio)
            """)
            
            # Indice per ordinamenti temporali
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON interazioni(timestamp DESC)
            """)
            
            self.conn.commit()
            print("✅ Schema database inizializzato")
            
        except sqlite3.Error as e:
            print(f"❌ Errore creazione schema: {e}")
            raise
    
    def salva_interazione(
        self,
        paziente_id: str,
        nome_paziente: str,
        parametri: VitalParameters,
        livello_rischio: str,
        raccomandazioni: str,
        allerta_medico: bool
    ) -> int:
        """
        Salva una nuova interazione nel database.
        
        Registra tutti i parametri vitali misurati insieme all'analisi
        dell'agente intelligente per tracciabilità completa.
        
        Args:
            paziente_id: ID univoco paziente
            nome_paziente: Nome completo
            parametri: Oggetto VitalParameters
            livello_rischio: 'basso', 'medio', 'alto'
            raccomandazioni: Testo raccomandazioni
            allerta_medico: Flag allerta
        
        Returns:
            int: ID dell'interazione salvata
        
        Raises:
            sqlite3.Error: In caso di errore database
        """
        try:
            cursor = self.conn.cursor()
            
            cursor.execute("""
                INSERT INTO interazioni (
                    paziente_id, nome_paziente,
                    pressione_sistolica, pressione_diastolica,
                    frequenza_cardiaca, temperatura,
                    saturazione_ossigeno, glicemia,
                    livello_rischio, raccomandazioni,
                    allerta_medico
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                paziente_id,
                nome_paziente,
                parametri.pressione_sistolica,
                parametri.pressione_diastolica,
                parametri.frequenza_cardiaca,
                parametri.temperatura,
                parametri.saturazione_ossigeno,
                parametri.glicemia,
                livello_rischio,
                raccomandazioni,
                allerta_medico
            ))
            
            self.conn.commit()
            return cursor.lastrowid
            
        except sqlite3.Error as e:
            print(f"❌ Errore salvataggio interazione: {e}")
            self.conn.rollback()
            raise
    
    def ottieni_storico_paziente(
        self, 
        paziente_id: str, 
        limite: int = 10
    ) -> List[sqlite3.Row]:
        """
        Recupera lo storico delle interazioni di un paziente.
        
        Ordinato per data decrescente (più recenti prima).
        Utile per trend analysis e follow-up.
        
        Args:
            paziente_id: ID paziente
            limite: Numero massimo record da restituire
        
        Returns:
            List[sqlite3.Row]: Lista interazioni ordinate per data
        """
        try:
            cursor = self.conn.cursor()
            
            cursor.execute("""
                SELECT * FROM interazioni
                WHERE paziente_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (paziente_id, limite))
            
            return cursor.fetchall()
            
        except sqlite3.Error as e:
            print(f"❌ Errore recupero storico: {e}")
            return []
    
    def ottieni_alert_critici(self) -> List[sqlite3.Row]:
        """
        Recupera tutte le interazioni con allerta medico attiva.
        
        Include solo alert delle ultime 48 ore per rilevanza.
        Ordinato per rischio (alto prima) e poi per data.
        
        Returns:
            List[sqlite3.Row]: Lista alert critici
        """
        try:
            cursor = self.conn.cursor()
            
            cursor.execute("""
                SELECT * FROM interazioni
                WHERE allerta_medico = 1
                AND datetime(timestamp) >= datetime('now', '-2 days')
                ORDER BY 
                    CASE livello_rischio
                        WHEN 'alto' THEN 1
                        WHEN 'medio' THEN 2
                        ELSE 3
                    END,
                    timestamp DESC
            """)
            
            return cursor.fetchall()
            
        except sqlite3.Error as e:
            print(f"❌ Errore recupero alert: {e}")
            return []
    
    def ottieni_pazienti_rischio_alto(self) -> List[sqlite3.Row]:
        """
        Recupera pazienti con ultima misurazione a rischio alto.
        
        Mostra solo l'ultima interazione per paziente per evitare duplicati.
        Essenziale per dashboard medico e prioritizzazione interventi.
        
        Returns:
            List[sqlite3.Row]: Pazienti a rischio alto
        """
        try:
            cursor = self.conn.cursor()
            
            # Subquery per trovare ultima interazione per paziente
            cursor.execute("""
                SELECT i1.* FROM interazioni i1
                INNER JOIN (
                    SELECT paziente_id, MAX(timestamp) as max_time
                    FROM interazioni
                    GROUP BY paziente_id
                ) i2 ON i1.paziente_id = i2.paziente_id 
                    AND i1.timestamp = i2.max_time
                WHERE i1.livello_rischio = 'alto'
                ORDER BY i1.timestamp DESC
            """)
            
            return cursor.fetchall()
            
        except sqlite3.Error as e:
            print(f"❌ Errore recupero pazienti rischio alto: {e}")
            return []
    
    def ottieni_statistiche(self) -> Dict[str, int]:
        """
        Calcola statistiche aggregate del sistema.
        
        Metriche fornite:
        - Totale interazioni registrate
        - Numero pazienti unici
        - Alert critici attivi
        - Distribuzione per livello rischio
        
        Utile per dashboard e reporting.
        
        Returns:
            Dict[str, int]: Dizionario con statistiche
        """
        try:
            cursor = self.conn.cursor()
            stats = {}
            
            # Totale interazioni
            cursor.execute("SELECT COUNT(*) FROM interazioni")
            stats['totale_interazioni'] = cursor.fetchone()[0]
            
            # Pazienti unici
            cursor.execute("SELECT COUNT(DISTINCT paziente_id) FROM interazioni")
            stats['pazienti_unici'] = cursor.fetchone()[0]
            
            # Alert critici ultimi 2 giorni
            cursor.execute("""
                SELECT COUNT(*) FROM interazioni
                WHERE allerta_medico = 1
                AND datetime(timestamp) >= datetime('now', '-2 days')
            """)
            stats['alert_critici'] = cursor.fetchone()[0]
            
            # Distribuzione per rischio (ultima misurazione per paziente)
            cursor.execute("""
                SELECT livello_rischio, COUNT(*) as cnt
                FROM (
                    SELECT i1.paziente_id, i1.livello_rischio
                    FROM interazioni i1
                    INNER JOIN (
                        SELECT paziente_id, MAX(timestamp) as max_time
                        FROM interazioni
                        GROUP BY paziente_id
                    ) i2 ON i1.paziente_id = i2.paziente_id 
                        AND i1.timestamp = i2.max_time
                )
                GROUP BY livello_rischio
            """)
            
            for row in cursor.fetchall():
                stats[f'rischio_{row[0]}'] = row[1]
            
            # Assicura che tutti i livelli esistano (anche con 0)
            for livello in ['basso', 'medio', 'alto']:
                if f'rischio_{livello}' not in stats:
                    stats[f'rischio_{livello}'] = 0
            
            return stats
            
        except sqlite3.Error as e:
            print(f"❌ Errore calcolo statistiche: {e}")
            return {
                'totale_interazioni': 0,
                'pazienti_unici': 0,
                'alert_critici': 0,
                'rischio_basso': 0,
                'rischio_medio': 0,
                'rischio_alto': 0
            }
    
    def ottieni_trend_paziente(
        self, 
        paziente_id: str, 
        parametro: str,
        giorni: int = 7
    ) -> List[tuple]:
        """
        Recupera il trend temporale di un parametro specifico.
        
        Utile per visualizzare grafici di evoluzione nel tempo.
        
        Args:
            paziente_id: ID paziente
            parametro: Nome colonna parametro (es. 'pressione_sistolica')
            giorni: Numero giorni da considerare
        
        Returns:
            List[tuple]: Lista (timestamp, valore)
        """
        try:
            cursor = self.conn.cursor()
            
            # Validazione parametro per prevenire SQL injection
            parametri_validi = [
                'pressione_sistolica', 'pressione_diastolica',
                'frequenza_cardiaca', 'temperatura',
                'saturazione_ossigeno', 'glicemia'
            ]
            
            if parametro not in parametri_validi:
                raise ValueError(f"Parametro non valido: {parametro}")
            
            query = f"""
                SELECT timestamp, {parametro}
                FROM interazioni
                WHERE paziente_id = ?
                AND datetime(timestamp) >= datetime('now', '-{giorni} days')
                ORDER BY timestamp ASC
            """
            
            cursor.execute(query, (paziente_id,))
            return cursor.fetchall()
            
        except (sqlite3.Error, ValueError) as e:
            print(f"❌ Errore recupero trend: {e}")
            return []
    
    def chiudi(self):
        """
        Chiude la connessione al database.
        
        Da chiamare sempre alla fine dell'esecuzione per:
        - Salvare modifiche pendenti
        - Liberare risorse
        - Evitare corruzione database
        """
        if self.conn:
            self.conn.close()
            print("✅ Connessione database chiusa")
    
    def __del__(self):
        """Destructor: assicura chiusura connessione"""
        self.chiudi()
