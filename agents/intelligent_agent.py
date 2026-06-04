"""
agents/intelligent_agent.py - Agente Intelligente per Analisi Parametri

Questo modulo implementa l'agente intelligente che analizza i parametri vitali
e determina il livello di rischio del paziente.

L'agente utilizza un sistema a regole basato su:
1. Analisi individuale di ogni parametro
2. Valutazione della gravita delle anomalie
3. Analisi delle correlazioni tra parametri
4. Determinazione del rischio globale

Output dell'agente:
- Livello di rischio: basso, medio, alto
- Lista anomalie rilevate
- Raccomandazioni personalizzate
- Flag per allerta medico
"""

from models.vital_parameters import VitalParameters
from typing import Dict, List


class IntelligentAgent:
    """
    Agente intelligente per l'analisi dei parametri vitali.

    Implementa un sistema esperto basato su regole mediche che:
    - Analizza ogni parametro vitale
    - Identifica anomalie e loro gravita
    - Correla parametri per identificare pattern patologici
    - Genera raccomandazioni personalizzate
    - Decide quando allertare il medico
    """

    def __init__(self):
        """Inizializza l'agente con le regole di analisi"""
        self.regole_correlazione = self._inizializza_regole_correlazione()

    def analizza_parametri(self, parametri: VitalParameters) -> Dict:
        """
        Analisi completa dei parametri vitali.

        Processo di analisi:
        1. Validazione formale dei parametri
        2. Identificazione anomalie individuali
        3. Analisi correlazioni tra parametri
        4. Calcolo livello rischio globale
        5. Generazione raccomandazioni

        Args:
            parametri: Oggetto VitalParameters con i valori misurati

        Returns:
            Dict con:
                - livello_rischio: str ('basso', 'medio', 'alto')
                - anomalie: List[str] descrizioni anomalie
                - raccomandazioni: str consigli per il paziente
                - allerta_medico: bool flag per notifica medico
                - dettagli_analisi: Dict con informazioni dettagliate
        """

        # Step 1: Validazione parametri
        valido, errori = parametri.valida_parametri()
        if not valido:
            return {
                'livello_rischio': 'errore',
                'anomalie': errori,
                'raccomandazioni': 'Parametri non validi. Controllare le misurazioni.',
                'allerta_medico': True,
                'dettagli_analisi': {'errori_validazione': errori}
            }

        # Step 2: Identificazione anomalie
        anomalie = parametri.identifica_anomalie()

        # Step 3: Analisi correlazioni
        pattern_critici = self._analizza_correlazioni(parametri)

        # Step 4: Calcolo livello rischio
        livello_rischio = self._calcola_livello_rischio(anomalie, pattern_critici)

        # Step 5: Generazione raccomandazioni
        raccomandazioni = self._genera_raccomandazioni(
            anomalie,
            pattern_critici,
            livello_rischio
        )

        # Step 6: Decisione allerta medico
        allerta_medico = self._richiede_allerta_medico(livello_rischio, anomalie, pattern_critici)

        # Formattazione anomalie per output
        anomalie_descrizione = self._formatta_anomalie(anomalie)

        return {
            'livello_rischio': livello_rischio,
            'anomalie': anomalie_descrizione,
            'raccomandazioni': raccomandazioni,
            'allerta_medico': allerta_medico,
            'dettagli_analisi': {
                'numero_anomalie': len(anomalie),
                'pattern_critici': pattern_critici,
                'anomalie_dettagliate': anomalie
            }
        }

    def _calcola_livello_rischio(self, anomalie: List[Dict], pattern_critici: List[str]) -> str:
        """
        Calcola il livello di rischio globale del paziente.

        Logica di calcolo:
        - ALTO: presenza di anomalie critiche o pattern pericolosi
        - MEDIO: presenza di anomalie moderate o multiple lievi
        - BASSO: nessuna anomalia o solo anomalie lievi isolate

        Args:
            anomalie: Lista anomalie identificate
            pattern_critici: Lista pattern patologici rilevati

        Returns:
            str: 'basso', 'medio', 'alto'
        """

        # Nessuna anomalia -> rischio basso
        if len(anomalie) == 0:
            return 'basso'

        # Pattern critici identificati -> rischio alto
        if len(pattern_critici) > 0:
            return 'alto'

        # Conta anomalie per gravita
        critiche = sum(1 for a in anomalie if a['gravita'] == 'critica')
        moderate = sum(1 for a in anomalie if a['gravita'] == 'moderata')
        lievi = sum(1 for a in anomalie if a['gravita'] == 'lieve')

        # Almeno un'anomalia critica -> rischio alto
        if critiche > 0:
            return 'alto'

        # Piu di una moderata O piu di due lievi -> rischio medio
        if moderate > 1 or lievi > 2:
            return 'medio'

        # Una moderata O 1-2 lievi -> rischio medio
        if moderate > 0 or lievi > 0:
            return 'medio'

        return 'basso'

    def _analizza_correlazioni(self, parametri: VitalParameters) -> List[str]:
        """
        Analizza correlazioni tra parametri per identificare pattern patologici.

        Pattern monitorati:
        - Shock ipovolemico: ipotensione + tachicardia
        - Insufficienza respiratoria: ipossia + tachicardia
        - Crisi ipertensiva: ipertensione grave + tachicardia
        - Ipoglicemia severa: glicemia bassa + tachicardia
        - Infezione sistemica: febbre + tachicardia

        Args:
            parametri: Parametri vitali misurati

        Returns:
            List[str]: Pattern critici identificati
        """
        pattern = []

        # Pattern 1: Possibile shock (pressione bassa + frequenza alta)
        if (
            parametri.pressione_sistolica < 90
            and parametri.frequenza_cardiaca > 100
        ):
            pattern.append("SHOCK POSSIBILE: Ipotensione con tachicardia compensatoria")

        # Pattern 2: Crisi ipertensiva (pressione molto alta)
        if (
            parametri.pressione_sistolica >= 180
            or parametri.pressione_diastolica >= 110
        ):
            pattern.append("CRISI IPERTENSIVA: Pressione arteriosa pericolosamente elevata")

        # Pattern 3: Insufficienza respiratoria (ipossia + compenso cardiaco)
        if (
            parametri.saturazione_ossigeno < 92
            and parametri.frequenza_cardiaca > 100
        ):
            pattern.append("INSUFFICIENZA RESPIRATORIA: Ipossia con tachicardia compensatoria")

        # Pattern 4: Sepsi/infezione grave (febbre alta + tachicardia)
        if (
            parametri.temperatura >= 38.5
            and parametri.frequenza_cardiaca > 100
        ):
            pattern.append("INFEZIONE SISTEMICA: Febbre elevata con risposta cardiaca")

        # Pattern 5: Ipoglicemia severa con compenso
        if (
            parametri.glicemia < 60
            and parametri.frequenza_cardiaca > 100
        ):
            pattern.append("IPOGLICEMIA SEVERA: Glicemia critica con tachicardia")

        # Pattern 6: Ipertermia critica
        if parametri.temperatura >= 39.5:
            pattern.append("IPERTERMIA CRITICA: Temperatura pericolosamente elevata")

        # Pattern 7: Ipossia severa
        if parametri.saturazione_ossigeno < 88:
            pattern.append("IPOSSIA SEVERA: Saturazione ossigeno critica")

        return pattern

    def _genera_raccomandazioni(
        self,
        anomalie: List[Dict],
        pattern_critici: List[str],
        livello_rischio: str
    ) -> str:
        """
        Genera raccomandazioni personalizzate basate sull'analisi.

        Le raccomandazioni sono graduali in base al rischio:
        - ALTO: richiedere assistenza medica immediata
        - MEDIO: contattare il medico, monitoraggio ravvicinato
        - BASSO: consigli di lifestyle e monitoraggio standard

        Args:
            anomalie: Anomalie identificate
            pattern_critici: Pattern patologici
            livello_rischio: Livello rischio calcolato

        Returns:
            str: Raccomandazioni testuali per il paziente
        """

        # Rischio ALTO: intervento urgente
        if livello_rischio == 'alto':
            if len(pattern_critici) > 0:
                return (
                    "ATTENZIONE: Rilevati parametri critici che richiedono valutazione medica IMMEDIATA. "
                    "Si raccomanda di recarsi al pronto soccorso o chiamare il 118. "
                    f"Pattern identificati: {', '.join(pattern_critici[:2])}."
                )
            return (
                "ATTENZIONE: Parametri vitali significativamente alterati. "
                "Contattare IMMEDIATAMENTE il proprio medico o recarsi al pronto soccorso. "
                "Non attendere il miglioramento spontaneo."
            )

        # Rischio MEDIO: monitoraggio e contatto medico
        if livello_rischio == 'medio':
            raccomandazioni = [
                "Contattare il proprio medico entro 24 ore per valutazione.",
                "Ripetere la misurazione dei parametri vitali tra 4-6 ore."
            ]

            # Raccomandazioni specifiche per parametro
            for anomalia in anomalie:
                param = anomalia['parametro']

                if 'Pressione' in param:
                    if anomalia['valore'] > anomalia['range_normale'][1]:
                        raccomandazioni.append("Ridurre il consumo di sale e riposare.")
                    else:
                        raccomandazioni.append("Mantenersi idratati e evitare alzate brusche.")

                elif 'Frequenza Cardiaca' in param:
                    if anomalia['valore'] > anomalia['range_normale'][1]:
                        raccomandazioni.append("Evitare sforzi fisici e caffeina.")
                    else:
                        raccomandazioni.append("Evitare attivita che richiedono prontezza.")

                elif 'Temperatura' in param:
                    if anomalia['valore'] > anomalia['range_normale'][1]:
                        raccomandazioni.append("Assumere antipiretici se prescritti e mantenersi idratati.")

                elif 'Saturazione' in param:
                    raccomandazioni.append("Respirare profondamente e stare in ambiente ventilato.")

                elif 'Glicemia' in param:
                    if anomalia['valore'] > anomalia['range_normale'][1]:
                        raccomandazioni.append("Evitare cibi zuccherati e verificare terapia diabetica.")
                    else:
                        raccomandazioni.append("Assumere zuccheri semplici (succo, miele) e riposare.")

            return " ".join(raccomandazioni[:3])

        # Rischio BASSO: monitoraggio standard
        return (
            "Parametri vitali nella norma o con lievi variazioni. "
            "Continuare il monitoraggio regolare. "
            "Mantenere uno stile di vita sano con alimentazione equilibrata e attivita fisica regolare. "
            "Ripetere la misurazione come da programma di follow-up."
        )

    def _richiede_allerta_medico(
        self,
        livello_rischio: str,
        anomalie: List[Dict],
        pattern_critici: List[str]
    ) -> bool:
        """
        Determina se e necessario allertare il medico.

        Criteri per allerta:
        - Livello rischio ALTO
        - Presenza di pattern critici
        - Anomalie critiche multiple

        Args:
            livello_rischio: Livello rischio calcolato
            anomalie: Anomalie identificate
            pattern_critici: Pattern patologici

        Returns:
            bool: True se serve allerta medico
        """

        # Rischio alto -> sempre allertare
        if livello_rischio == 'alto':
            return True

        # Pattern critici -> sempre allertare
        if len(pattern_critici) > 0:
            return True

        # Anomalie critiche multiple -> allertare
        critiche = sum(1 for a in anomalie if a['gravita'] == 'critica')
        if critiche >= 2:
            return True

        return False

    def _formatta_anomalie(self, anomalie: List[Dict]) -> List[str]:
        """
        Formatta le anomalie in descrizioni testuali leggibili.

        Args:
            anomalie: Lista anomalie con dettagli

        Returns:
            List[str]: Descrizioni testuali delle anomalie
        """
        if len(anomalie) == 0:
            return []

        descrizioni = []
        for a in anomalie:
            min_norm, max_norm = a['range_normale']

            if a['valore'] > max_norm:
                direzione = "elevata"
            else:
                direzione = "bassa"

            desc = (
                f"{a['parametro']}: {a['valore']} {a['unita']} "
                f"({direzione}, normale: {min_norm}-{max_norm} {a['unita']}) "
                f"[Gravita: {a['gravita'].upper()}]"
            )
            descrizioni.append(desc)

        return descrizioni

    def _inizializza_regole_correlazione(self) -> Dict:
        """
        Inizializza le regole per l'analisi delle correlazioni.

        Queste regole potrebbero essere espanse in futuro con:
        - Machine learning per pattern recognition
        - Regole personalizzate per patologie specifiche
        - Integrazione con storia clinica del paziente

        Returns:
            Dict: Dizionario con regole di correlazione
        """
        return {
            'shock_pattern': {
                'condizioni': ['pressione_bassa', 'tachicardia'],
                'gravita': 'critica',
                'azione': 'intervento_immediato'
            },
            'respiratory_failure': {
                'condizioni': ['ipossia', 'tachicardia'],
                'gravita': 'critica',
                'azione': 'intervento_immediato'
            },
            'hypertensive_crisis': {
                'condizioni': ['ipertensione_severa'],
                'gravita': 'critica',
                'azione': 'intervento_urgente'
            }
        }
