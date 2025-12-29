"""
test_examples.py - Script per Testing Automatico Sistema

Questo file contiene test automatici per validare il corretto funzionamento
di tutti i componenti del sistema di telemedicina.

Eseguire con: python test_examples.py

Test coperti:
1. Validazione parametri vitali
2. Identificazione anomalie
3. Analisi agente intelligente
4. Calcolo livello rischio
5. Salvataggio/recupero database
6. Sistema notifiche

Non richiede librerie esterne (no pytest necessario).
"""

from models.vital_parameters import VitalParameters
from agents.intelligent_agent import IntelligentAgent
from database.db_manager import DatabaseManager
from utils.notifications import NotificationSystem
import os
import sys


class TestRunner:
    """Runner semplice per eseguire test senza dipendenze esterne"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.total = 0
    
    def run_test(self, test_name: str, test_func):
        """Esegue un singolo test e traccia il risultato"""
        self.total += 1
        try:
            test_func()
            self.passed += 1
            print(f"✅ PASS: {test_name}")
        except AssertionError as e:
            self.failed += 1
            print(f"❌ FAIL: {test_name}")
            print(f"   Errore: {e}")
        except Exception as e:
            self.failed += 1
            print(f"❌ ERROR: {test_name}")
            print(f"   Eccezione: {e}")
    
    def print_summary(self):
        """Stampa riepilogo risultati"""
        print("\n" + "="*60)
        print("RIEPILOGO TEST")
        print("="*60)
        print(f"Totale test: {self.total}")
        print(f"Passati: {self.passed} ✅")
        print(f"Falliti: {self.failed} ❌")
        print(f"Success rate: {(self.passed/self.total)*100:.1f}%")
        print("="*60)


# ============================================================================
# TEST PARAMETRI VITALI
# ============================================================================

def test_parametri_normali():
    """Test con tutti i parametri nel range normale"""
    params = VitalParameters(
        pressione_sistolica=120,
        pressione_diastolica=80,
        frequenza_cardiaca=75,
        temperatura=36.8,
        saturazione_ossigeno=98,
        glicemia=95
    )
    
    valido, errori = params.valida_parametri()
    assert valido, f"Parametri normali dovrebbero essere validi: {errori}"
    
    anomalie = params.identifica_anomalie()
    assert len(anomalie) == 0, f"Parametri normali non dovrebbero avere anomalie: {anomalie}"


def test_validazione_pressione_invalida():
    """Test che la validazione rilevi pressioni impossibili"""
    params = VitalParameters(
        pressione_sistolica=80,  # Minore della diastolica!
        pressione_diastolica=120,
        frequenza_cardiaca=75,
        temperatura=36.8,
        saturazione_ossigeno=98,
        glicemia=95
    )
    
    valido, errori = params.valida_parametri()
    assert not valido, "Pressione sistolica < diastolica dovrebbe essere invalida"
    assert any("maggiore" in err.lower() for err in errori), "Dovrebbe segnalare errore pressione"


def test_identificazione_ipertensione():
    """Test identificazione ipertensione"""
    params = VitalParameters(
        pressione_sistolica=160,  # Alta
        pressione_diastolica=100,  # Alta
        frequenza_cardiaca=75,
        temperatura=36.8,
        saturazione_ossigeno=98,
        glicemia=95
    )
    
    anomalie = params.identifica_anomalie()
    assert len(anomalie) >= 2, "Dovrebbe rilevare anomalie pressione sistolica e diastolica"
    
    # Verifica che le anomalie siano di pressione
    anomalie_pressione = [a for a in anomalie if 'Pressione' in a['parametro']]
    assert len(anomalie_pressione) == 2, "Dovrebbero esserci 2 anomalie di pressione"


def test_classificazione_gravita():
    """Test che la gravità delle anomalie sia classificata correttamente"""
    params = VitalParameters(
        pressione_sistolica=195,  # Critica (>180)
        pressione_diastolica=80,
        frequenza_cardiaca=75,
        temperatura=36.8,
        saturazione_ossigeno=98,
        glicemia=95
    )
    
    anomalie = params.identifica_anomalie()
    anomalia_sistolica = [a for a in anomalie if 'Sistolica' in a['parametro']][0]
    
    assert anomalia_sistolica['gravita'] == 'critica', \
        f"Pressione 195 dovrebbe essere critica, invece: {anomalia_sistolica['gravita']}"


# ============================================================================
# TEST AGENTE INTELLIGENTE
# ============================================================================

def test_agente_parametri_normali():
    """Test che l'agente classifichi correttamente parametri normali"""
    agent = IntelligentAgent()
    params = VitalParameters(
        pressione_sistolica=120,
        pressione_diastolica=80,
        frequenza_cardiaca=75,
        temperatura=36.8,
        saturazione_ossigeno=98,
        glicemia=95
    )
    
    analisi = agent.analizza_parametri(params)
    
    assert analisi['livello_rischio'] == 'basso', \
        f"Parametri normali dovrebbero dare rischio basso: {analisi['livello_rischio']}"
    assert not analisi['allerta_medico'], \
        "Parametri normali non dovrebbero allertare medico"
    assert len(analisi['anomalie']) == 0, \
        "Parametri normali non dovrebbero avere anomalie"


def test_agente_crisi_ipertensiva():
    """Test che l'agente rilevi crisi ipertensiva"""
    agent = IntelligentAgent()
    params = VitalParameters(
        pressione_sistolica=195,  # Critica
        pressione_diastolica=115,  # Critica
        frequenza_cardiaca=75,
        temperatura=36.8,
        saturazione_ossigeno=98,
        glicemia=95
    )
    
    analisi = agent.analizza_parametri(params)
    
    assert analisi['livello_rischio'] == 'alto', \
        f"Crisi ipertensiva dovrebbe dare rischio alto: {analisi['livello_rischio']}"
    assert analisi['allerta_medico'], \
        "Crisi ipertensiva dovrebbe allertare medico"
    
    # Verifica che sia rilevato il pattern
    dettagli = analisi['dettagli_analisi']
    pattern_critici = dettagli['pattern_critici']
    assert any('IPERTENSIVA' in p for p in pattern_critici), \
        "Dovrebbe rilevare pattern crisi ipertensiva"


def test_agente_shock_pattern():
    """Test che l'agente rilevi pattern di possibile shock"""
    agent = IntelligentAgent()
    params = VitalParameters(
        pressione_sistolica=85,  # Bassa
        pressione_diastolica=55,  # Bassa
        frequenza_cardiaca=125,  # Alta (compenso)
        temperatura=36.8,
        saturazione_ossigeno=98,
        glicemia=95
    )
    
    analisi = agent.analizza_parametri(params)
    
    assert analisi['livello_rischio'] == 'alto', \
        "Pattern shock dovrebbe dare rischio alto"
    assert analisi['allerta_medico'], \
        "Pattern shock dovrebbe allertare medico"
    
    pattern_critici = analisi['dettagli_analisi']['pattern_critici']
    assert any('SHOCK' in p for p in pattern_critici), \
        "Dovrebbe rilevare pattern shock"


def test_agente_rischio_medio():
    """Test che l'agente classifichi correttamente rischio medio"""
    agent = IntelligentAgent()
    params = VitalParameters(
        pressione_sistolica=155,  # Elevata moderata
        pressione_diastolica=95,  # Elevata moderata
        frequenza_cardiaca=75,
        temperatura=36.8,
        saturazione_ossigeno=98,
        glicemia=95
    )
    
    analisi = agent.analizza_parametri(params)
    
    assert analisi['livello_rischio'] == 'medio', \
        f"Ipertensione moderata dovrebbe dare rischio medio: {analisi['livello_rischio']}"
    assert not analisi['allerta_medico'], \
        "Rischio medio non dovrebbe allertare medico automaticamente"


# ============================================================================
# TEST DATABASE
# ============================================================================

def test_database_salvataggio_recupero():
    """Test salvataggio e recupero dati dal database"""
    # Usa database temporaneo
    test_db = "test_temp.db"
    
    try:
        db = DatabaseManager(test_db)
        
        # Crea parametri test
        params = VitalParameters(
            pressione_sistolica=120,
            pressione_diastolica=80,
            frequenza_cardiaca=75,
            temperatura=36.8,
            saturazione_ossigeno=98,
            glicemia=95
        )
        
        # Salva interazione
        interaction_id = db.salva_interazione(
            paziente_id="TEST_001",
            nome_paziente="Test Patient",
            parametri=params,
            livello_rischio="basso",
            raccomandazioni="Test recommendations",
            allerta_medico=False
        )
        
        assert interaction_id > 0, "Dovrebbe restituire ID valido"
        
        # Recupera storico
        storico = db.ottieni_storico_paziente("TEST_001")
        assert len(storico) == 1, "Dovrebbe esserci 1 interazione"
        
        # Verifica dati salvati
        record = storico[0]
        assert record[1] == "TEST_001", "ID paziente non corrisponde"
        assert record[2] == "Test Patient", "Nome paziente non corrisponde"
        assert record[3] == 120, "Pressione sistolica non corrisponde"
        
        db.chiudi()
    
    finally:
        # Cleanup: rimuovi database test
        if os.path.exists(test_db):
            os.remove(test_db)


def test_database_alert_critici():
    """Test recupero alert critici"""
    test_db = "test_temp.db"
    
    try:
        db = DatabaseManager(test_db)
        
        # Salva interazione critica
        params_critici = VitalParameters(
            pressione_sistolica=195,
            pressione_diastolica=115,
            frequenza_cardiaca=125,
            temperatura=36.8,
            saturazione_ossigeno=98,
            glicemia=95
        )
        
        db.salva_interazione(
            paziente_id="CRITICAL_001",
            nome_paziente="Critical Patient",
            parametri=params_critici,
            livello_rischio="alto",
            raccomandazioni="Immediate medical attention",
            allerta_medico=True  # Alert medico attivo
        )
        
        # Recupera alert
        alert = db.ottieni_alert_critici()
        assert len(alert) >= 1, "Dovrebbe esserci almeno 1 alert"
        
        # Verifica che l'alert sia quello salvato
        alert_trovato = any(a[1] == "CRITICAL_001" for a in alert)
        assert alert_trovato, "L'alert salvato dovrebbe essere recuperato"
        
        db.chiudi()
    
    finally:
        if os.path.exists(test_db):
            os.remove(test_db)


def test_database_statistiche():
    """Test calcolo statistiche"""
    test_db = "test_temp.db"
    
    try:
        db = DatabaseManager(test_db)
        
        # Aggiungi multiple interazioni
        for i in range(5):
            params = VitalParameters(
                pressione_sistolica=120 + i*10,
                pressione_diastolica=80,
                frequenza_cardiaca=75,
                temperatura=36.8,
                saturazione_ossigeno=98,
                glicemia=95
            )
            
            livello = 'basso' if i < 2 else 'medio' if i < 4 else 'alto'
            
            db.salva_interazione(
                paziente_id=f"PAZ_{i}",
                nome_paziente=f"Paziente {i}",
                parametri=params,
                livello_rischio=livello,
                raccomandazioni="Test",
                allerta_medico=(livello == 'alto')
            )
        
        # Calcola statistiche
        stats = db.ottieni_statistiche()
        
        assert stats['totale_interazioni'] == 5, "Dovrebbero esserci 5 interazioni"
        assert stats['pazienti_unici'] == 5, "Dovrebbero esserci 5 pazienti unici"
        assert stats['rischio_basso'] == 2, "Dovrebbero esserci 2 pazienti a rischio basso"
        assert stats['rischio_medio'] == 2, "Dovrebbero esserci 2 pazienti a rischio medio"
        assert stats['rischio_alto'] == 1, "Dovrebbe esserci 1 paziente a rischio alto"
        
        db.chiudi()
    
    finally:
        if os.path.exists(test_db):
            os.remove(test_db)


# ============================================================================
# TEST NOTIFICHE
# ============================================================================

def test_notifiche_creazione():
    """Test che il sistema notifiche si inizializzi correttamente"""
    test_log_dir = "test_logs"
    
    try:
        notif = NotificationSystem(log_dir=test_log_dir)
        
        # Verifica che la directory sia stata creata
        assert os.path.exists(test_log_dir), "Directory log dovrebbe esistere"
        
        # Verifica attributi
        assert hasattr(notif, 'log_paziente'), "Dovrebbe avere attributo log_paziente"
        assert hasattr(notif, 'log_medico'), "Dovrebbe avere attributo log_medico"
    
    finally:
        # Cleanup
        import shutil
        if os.path.exists(test_log_dir):
            shutil.rmtree(test_log_dir)


def test_notifiche_log():
    """Test che le notifiche vengano loggate correttamente"""
    test_log_dir = "test_logs"
    
    try:
        notif = NotificationSystem(log_dir=test_log_dir)
        
        # Invia notifica paziente
        notif.invia_notifica_paziente(
            paziente_id="TEST_001",
            messaggio="Test message",
            livello_rischio="basso"
        )
        
        # Verifica che il file log esista e contenga dati
        log_file = os.path.join(test_log_dir, "notifiche_pazienti.log")
        assert os.path.exists(log_file), "File log dovrebbe esistere"
        
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "TEST_001" in content, "Log dovrebbe contenere ID paziente"
            assert "Test message" in content, "Log dovrebbe contenere messaggio"
    
    finally:
        import shutil
        if os.path.exists(test_log_dir):
            shutil.rmtree(test_log_dir)


# ============================================================================
# TEST INTEGRAZIONE
# ============================================================================

def test_integrazione_completa():
    """Test del flusso completo: parametri → agente → database → notifiche"""
    test_db = "test_integration.db"
    test_logs = "test_integration_logs"
    
    try:
        # Setup componenti
        db = DatabaseManager(test_db)
        agent = IntelligentAgent()
        notif = NotificationSystem(log_dir=test_logs)
        
        # 1. Crea parametri
        params = VitalParameters(
            pressione_sistolica=195,
            pressione_diastolica=115,
            frequenza_cardiaca=125,
            temperatura=38.5,
            saturazione_ossigeno=90,
            glicemia=200
        )
        
        # 2. Analizza con agente
        analisi = agent.analizza_parametri(params)
        assert analisi['livello_rischio'] == 'alto', "Dovrebbe essere rischio alto"
        
        # 3. Salva in database
        interaction_id = db.salva_interazione(
            paziente_id="INTEGRATION_001",
            nome_paziente="Integration Test",
            parametri=params,
            livello_rischio=analisi['livello_rischio'],
            raccomandazioni=analisi['raccomandazioni'],
            allerta_medico=analisi['allerta_medico']
        )
        assert interaction_id > 0, "Interazione dovrebbe essere salvata"
        
        # 4. Invia notifiche
        notif.invia_notifica_paziente(
            paziente_id="INTEGRATION_001",
            messaggio=analisi['raccomandazioni'],
            livello_rischio=analisi['livello_rischio']
        )
        
        if analisi['allerta_medico']:
            notif.invia_allerta_medico(
                paziente_id="INTEGRATION_001",
                nome_paziente="Integration Test",
                parametri=params,
                anomalie=analisi['anomalie']
            )
        
        # 5. Verifica recupero dati
        storico = db.ottieni_storico_paziente("INTEGRATION_001")
        assert len(storico) == 1, "Dovrebbe esserci 1 record"
        
        alert = db.ottieni_alert_critici()
        assert len(alert) >= 1, "Dovrebbe esserci almeno 1 alert"
        
        db.chiudi()
        
    finally:
        # Cleanup
        if os.path.exists(test_db):
            os.remove(test_db)
        
        import shutil
        if os.path.exists(test_logs):
            shutil.rmtree(test_logs)


# ============================================================================
# MAIN - Esecuzione Test
# ============================================================================

def main():
    """Funzione principale che esegue tutti i test"""
    print("\n" + "="*60)
    print("SISTEMA DI TELEMEDICINA - TEST SUITE")
    print("="*60 + "\n")
    
    runner = TestRunner()
    
    # Test Parametri Vitali
    print("📊 TEST PARAMETRI VITALI")
    print("-" * 60)
    runner.run_test("Parametri normali", test_parametri_normali)
    runner.run_test("Validazione pressione invalida", test_validazione_pressione_invalida)
    runner.run_test("Identificazione ipertensione", test_identificazione_ipertensione)
    runner.run_test("Classificazione gravità", test_classificazione_gravita)
    
    # Test Agente Intelligente
    print("\n🤖 TEST AGENTE INTELLIGENTE")
    print("-" * 60)
    runner.run_test("Agente con parametri normali", test_agente_parametri_normali)
    runner.run_test("Agente rileva crisi ipertensiva", test_agente_crisi_ipertensiva)
    runner.run_test("Agente rileva pattern shock", test_agente_shock_pattern)
    runner.run_test("Agente classifica rischio medio", test_agente_rischio_medio)
    
    # Test Database
    print("\n💾 TEST DATABASE")
    print("-" * 60)
    runner.run_test("Salvataggio e recupero", test_database_salvataggio_recupero)
    runner.run_test("Recupero alert critici", test_database_alert_critici)
    runner.run_test("Calcolo statistiche", test_database_statistiche)
    
    # Test Notifiche
    print("\n🔔 TEST SISTEMA NOTIFICHE")
    print("-" * 60)
    runner.run_test("Creazione sistema notifiche", test_notifiche_creazione)
    runner.run_test("Logging notifiche", test_notifiche_log)
    
    # Test Integrazione
    print("\n🔗 TEST INTEGRAZIONE")
    print("-" * 60)
    runner.run_test("Flusso completo integrazione", test_integrazione_completa)
    
    # Riepilogo
    runner.print_summary()
    
    # Exit code
    sys.exit(0 if runner.failed == 0 else 1)


if __name__ == "__main__":
    main()
