"""
models/vital_parameters.py - Modello Dati per Parametri Vitali

Questo modulo definisce la struttura dati per i parametri vitali di un paziente.
Incapsula tutti i valori misurati e fornisce metodi per la validazione.

Parametri monitorati:
- Pressione arteriosa (sistolica/diastolica)
- Frequenza cardiaca
- Temperatura corporea
- Saturazione ossigeno
- Glicemia

Ogni parametro ha range di normalità definiti per l'analisi.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class VitalParameters:
    """
    Classe che rappresenta i parametri vitali di un paziente.
    
    Utilizza dataclass per una gestione automatica di __init__, __repr__, ecc.
    Tutti i valori sono float per garantire precisione nelle misurazioni.
    """
    
    pressione_sistolica: float  # mmHg - Pressione durante contrazione cardiaca
    pressione_diastolica: float  # mmHg - Pressione durante rilassamento
    frequenza_cardiaca: float  # bpm - Battiti per minuto
    temperatura: float  # °C - Temperatura corporea
    saturazione_ossigeno: float  # % - SpO2
    glicemia: float  # mg/dL - Glucosio nel sangue
    
    # Range di normalità (min, max) per ogni parametro
    # Basati su linee guida mediche standard
    RANGE_NORMALI = {
        'pressione_sistolica': (90, 140),
        'pressione_diastolica': (60, 90),
        'frequenza_cardiaca': (60, 100),
        'temperatura': (36.0, 37.5),
        'saturazione_ossigeno': (95, 100),
        'glicemia': (70, 140)
    }
    
    # Soglie critiche che richiedono intervento immediato
    SOGLIE_CRITICHE = {
        'pressione_sistolica': (180, 'alta'),
        'pressione_diastolica': (110, 'alta'),
        'frequenza_cardiaca_alta': (120, 'alta'),
        'frequenza_cardiaca_bassa': (50, 'bassa'),
        'temperatura_alta': (38.5, 'alta'),
        'saturazione_ossigeno': (90, 'bassa'),
        'glicemia_alta': (200, 'alta'),
        'glicemia_bassa': (60, 'bassa')
    }
    
    def valida_parametri(self) -> Tuple[bool, List[str]]:
        """
        Valida che tutti i parametri siano in range fisiologicamente possibili.
        Non valida se sono "normali", ma solo se sono valori plausibili.
        
        Returns:
            tuple[bool, List[str]]: (True se validi, Lista errori)
        """
        errori = []
        
        # Validazione pressione sistolica
        if not (50 <= self.pressione_sistolica <= 250):
            errori.append("Pressione sistolica fuori range fisiologico (50-250 mmHg)")
        
        # Validazione pressione diastolica
        if not (30 <= self.pressione_diastolica <= 150):
            errori.append("Pressione diastolica fuori range fisiologico (30-150 mmHg)")
        
        # La sistolica deve essere maggiore della diastolica
        if self.pressione_sistolica <= self.pressione_diastolica:
            errori.append("Pressione sistolica deve essere maggiore della diastolica")
        
        # Validazione frequenza cardiaca
        if not (30 <= self.frequenza_cardiaca <= 200):
            errori.append("Frequenza cardiaca fuori range fisiologico (30-200 bpm)")
        
        # Validazione temperatura
        if not (34.0 <= self.temperatura <= 42.0):
            errori.append("Temperatura fuori range fisiologico (34-42 °C)")
        
        # Validazione saturazione ossigeno
        if not (70 <= self.saturazione_ossigeno <= 100):
            errori.append("Saturazione ossigeno fuori range fisiologico (70-100 %)")
        
        # Validazione glicemia
        if not (20 <= self.glicemia <= 500):
            errori.append("Glicemia fuori range fisiologico (20-500 mg/dL)")
        
        return (len(errori) == 0, errori)
    
    def identifica_anomalie(self) -> List[Dict[str, any]]:
        """
        Identifica quali parametri sono fuori dal range normale.
        
        Returns:
            List[Dict]: Lista di anomalie con dettagli:
                - parametro: nome del parametro
                - valore: valore misurato
                - range_normale: tupla (min, max)
                - gravita: 'lieve', 'moderata', 'critica'
        """
        anomalie = []
        
        # Controllo pressione sistolica
        if not self._in_range('pressione_sistolica', self.pressione_sistolica):
            gravita = self._valuta_gravita_pressione_sistolica(self.pressione_sistolica)
            anomalie.append({
                'parametro': 'Pressione Sistolica',
                'valore': self.pressione_sistolica,
                'unita': 'mmHg',
                'range_normale': self.RANGE_NORMALI['pressione_sistolica'],
                'gravita': gravita
            })
        
        # Controllo pressione diastolica
        if not self._in_range('pressione_diastolica', self.pressione_diastolica):
            gravita = self._valuta_gravita_pressione_diastolica(self.pressione_diastolica)
            anomalie.append({
                'parametro': 'Pressione Diastolica',
                'valore': self.pressione_diastolica,
                'unita': 'mmHg',
                'range_normale': self.RANGE_NORMALI['pressione_diastolica'],
                'gravita': gravita
            })
        
        # Controllo frequenza cardiaca
        if not self._in_range('frequenza_cardiaca', self.frequenza_cardiaca):
            gravita = self._valuta_gravita_frequenza(self.frequenza_cardiaca)
            anomalie.append({
                'parametro': 'Frequenza Cardiaca',
                'valore': self.frequenza_cardiaca,
                'unita': 'bpm',
                'range_normale': self.RANGE_NORMALI['frequenza_cardiaca'],
                'gravita': gravita
            })
        
        # Controllo temperatura
        if not self._in_range('temperatura', self.temperatura):
            gravita = self._valuta_gravita_temperatura(self.temperatura)
            anomalie.append({
                'parametro': 'Temperatura',
                'valore': self.temperatura,
                'unita': '°C',
                'range_normale': self.RANGE_NORMALI['temperatura'],
                'gravita': gravita
            })
        
        # Controllo saturazione ossigeno
        if not self._in_range('saturazione_ossigeno', self.saturazione_ossigeno):
            gravita = self._valuta_gravita_saturazione(self.saturazione_ossigeno)
            anomalie.append({
                'parametro': 'Saturazione Ossigeno',
                'valore': self.saturazione_ossigeno,
                'unita': '%',
                'range_normale': self.RANGE_NORMALI['saturazione_ossigeno'],
                'gravita': gravita
            })
        
        # Controllo glicemia
        if not self._in_range('glicemia', self.glicemia):
            gravita = self._valuta_gravita_glicemia(self.glicemia)
            anomalie.append({
                'parametro': 'Glicemia',
                'valore': self.glicemia,
                'unita': 'mg/dL',
                'range_normale': self.RANGE_NORMALI['glicemia'],
                'gravita': gravita
            })
        
        return anomalie
    
    def _in_range(self, parametro: str, valore: float) -> bool:
        """Verifica se un valore è nel range normale"""
        min_val, max_val = self.RANGE_NORMALI[parametro]
        return min_val <= valore <= max_val
    
    def _valuta_gravita_pressione_sistolica(self, valore: float) -> str:
        """Valuta la gravità di un'anomalia nella pressione sistolica"""
        if valore >= 180:
            return 'critica'
        elif valore >= 160 or valore < 80:
            return 'moderata'
        else:
            return 'lieve'
    
    def _valuta_gravita_pressione_diastolica(self, valore: float) -> str:
        """Valuta la gravità di un'anomalia nella pressione diastolica"""
        if valore >= 110:
            return 'critica'
        elif valore >= 100 or valore < 50:
            return 'moderata'
        else:
            return 'lieve'
    
    def _valuta_gravita_frequenza(self, valore: float) -> str:
        """Valuta la gravità di un'anomalia nella frequenza cardiaca"""
        if valore >= 120 or valore <= 50:
            return 'critica'
        elif valore >= 110 or valore <= 55:
            return 'moderata'
        else:
            return 'lieve'
    
    def _valuta_gravita_temperatura(self, valore: float) -> str:
        """Valuta la gravità di un'anomalia nella temperatura"""
        if valore >= 38.5 or valore <= 35.0:
            return 'critica'
        elif valore >= 38.0 or valore <= 35.5:
            return 'moderata'
        else:
            return 'lieve'
    
    def _valuta_gravita_saturazione(self, valore: float) -> str:
        """Valuta la gravità di un'anomalia nella saturazione ossigeno"""
        if valore < 90:
            return 'critica'
        elif valore < 93:
            return 'moderata'
        else:
            return 'lieve'
    
    def _valuta_gravita_glicemia(self, valore: float) -> str:
        """Valuta la gravità di un'anomalia nella glicemia"""
        if valore >= 200 or valore <= 60:
            return 'critica'
        elif valore >= 180 or valore <= 65:
            return 'moderata'
        else:
            return 'lieve'
    
    def to_dict(self) -> Dict[str, float]:
        """
        Converte l'oggetto in dizionario per serializzazione.
        
        Returns:
            Dict[str, float]: Dizionario con tutti i parametri
        """
        return {
            'pressione_sistolica': self.pressione_sistolica,
            'pressione_diastolica': self.pressione_diastolica,
            'frequenza_cardiaca': self.frequenza_cardiaca,
            'temperatura': self.temperatura,
            'saturazione_ossigeno': self.saturazione_ossigeno,
            'glicemia': self.glicemia
        }
    
    def __str__(self) -> str:
        """Rappresentazione stringa leggibile dei parametri"""
        return (
            f"Parametri Vitali:\n"
            f"  Pressione: {self.pressione_sistolica}/{self.pressione_diastolica} mmHg\n"
            f"  Frequenza Cardiaca: {self.frequenza_cardiaca} bpm\n"
            f"  Temperatura: {self.temperatura}°C\n"
            f"  Saturazione O2: {self.saturazione_ossigeno}%\n"
            f"  Glicemia: {self.glicemia} mg/dL"
        )
