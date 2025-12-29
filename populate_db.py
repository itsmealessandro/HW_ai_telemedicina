# populate_db.py
"""
Popola il database di telemedicina con 20 record sintetici.
Serve per testare il training del modello supervisionato.
"""

import random
from database.db_manager import DatabaseManager
from models.vital_parameters import VitalParameters

def genera_parametri_sintetici():
    """Genera un oggetto VitalParameters con valori casuali realistici"""
    return VitalParameters(
        pressione_sistolica=random.uniform(90, 180),
        pressione_diastolica=random.uniform(60, 110),
        frequenza_cardiaca=random.uniform(50, 130),
        temperatura=random.uniform(36.0, 40.0),
        saturazione_ossigeno=random.uniform(85, 100),
        glicemia=random.uniform(50, 200)
    )

def main():
    db = DatabaseManager(db_path="telemedicina.db")
    
    for i in range(20):
        paziente_id = f"P{i:03d}"
        nome_paziente = f"Paziente {i}"
        parametri = genera_parametri_sintetici()
        livello_rischio = random.choice(["basso", "medio", "alto"])
        raccomandazioni = "Monitoraggio standard"
        allerta_medico = livello_rischio == "alto"
        
        db.salva_interazione(
            paziente_id=paziente_id,
            nome_paziente=nome_paziente,
            parametri=parametri,
            livello_rischio=livello_rischio,
            raccomandazioni=raccomandazioni,
            allerta_medico=allerta_medico
        )
        print(f"✅ Record {i+1}/20 inserito: {paziente_id} - {livello_rischio}")
    
    db.chiudi()
    print("🎉 Popolamento database completato!")

if __name__ == "__main__":
    main()

