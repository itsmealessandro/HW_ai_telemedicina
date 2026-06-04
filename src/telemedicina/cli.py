"""
cli.py - Interfaccia a riga di comando del sistema di telemedicina.
"""

import sys
import os
import numpy as np
from telemedicina.services.analysis_service import AnalysisService
from telemedicina.models.vital_parameters import VitalParameters
from telemedicina.agents.rl_agent import QLearningAgent, AZIONI, N_BINS, SOGLIE
from telemedicina import config


def stampa_menu_principale():
    print("\n" + "=" * 50)
    print("SISTEMA DI TELEMEDICINA - MONITORAGGIO VITALE")
    print("=" * 50)
    print("1. Inserisci nuovi parametri vitali (Paziente)")
    print("2. Visualizza storico parametri")
    print("3. Visualizza alert attivi")
    print("4. Area Medico - Visualizza pazienti critici")
    print("5. Statistiche sistema")
    print("0. Esci")
    print("=" * 50)


def inserisci_parametri(service):
    print("\n--- INSERIMENTO PARAMETRI VITALI ---")
    paziente_id = input("ID Paziente: ").strip()
    if not paziente_id:
        print("ID Paziente obbligatorio!")
        return
    nome = input("Nome Paziente: ").strip()

    try:
        print("\nInserisci i parametri vitali:")
        pressione_sistolica = float(input("Pressione Sistolica (mmHg) [90-140]: "))
        pressione_diastolica = float(input("Pressione Diastolica (mmHg) [60-90]: "))
        frequenza_cardiaca = float(input("Frequenza Cardiaca (bpm) [60-100]: "))
        temperatura = float(input("Temperatura (C) [36.0-37.5]: "))
        saturazione_ossigeno = float(input("Saturazione Ossigeno (%) [95-100]: "))
        glicemia = float(input("Glicemia (mg/dL) [70-140]: "))

        parametri = VitalParameters(
            pressione_sistolica=pressione_sistolica,
            pressione_diastolica=pressione_diastolica,
            frequenza_cardiaca=frequenza_cardiaca,
            temperatura=temperatura,
            saturazione_ossigeno=saturazione_ossigeno,
            glicemia=glicemia,
        )

        print("\nAnalisi parametri in corso...")
        analisi, interazione_id = service.analizza_e_salva(
            paziente_id=paziente_id,
            nome_paziente=nome,
            parametri=parametri,
        )

        print("\n" + "=" * 50)
        print("RISULTATI ANALISI")
        print("=" * 50)
        print(f"Livello di Rischio: {analisi['livello_rischio'].upper()}")
        print("\nParametri Anomali:")
        if analisi["anomalie"]:
            for anomalia in analisi["anomalie"]:
                print(f"  {anomalia}")
        else:
            print("  Tutti i parametri sono nella norma")
        print("\nRaccomandazioni:")
        print(f"  {analisi['raccomandazioni']}")
        if analisi["allerta_medico"]:
            print("\nATTENZIONE: Il medico e stato allertato!")
        print(f"\nInterazione salvata con ID: {interazione_id}")

    except ValueError as e:
        print(f"Errore: Inserire valori numerici validi! ({e})")
    except Exception as e:
        print(f"Errore durante l'elaborazione: {e}")


def visualizza_storico(service):
    print("\n--- STORICO PARAMETRI ---")
    paziente_id = input("ID Paziente: ").strip()
    storico = service.ottieni_storico(paziente_id)

    if not storico:
        print(f"Nessuna interazione trovata per il paziente {paziente_id}")
        return

    print(f"\nStorico per paziente: {storico[0][2]}")
    print("=" * 80)
    for row in storico:
        print(f"\nData: {row[8]}")
        print(f"Livello Rischio: {row[9]}")
        print(f"Pressione: {row[3]}/{row[4]} mmHg")
        print(f"Frequenza Cardiaca: {row[5]} bpm")
        print(f"Temperatura: {row[6]} C")
        print(f"Saturazione O2: {row[7]}%")
        print(f"Glicemia: {row[11]} mg/dL")
        print(f"Raccomandazioni: {row[10]}")
        print("-" * 80)


def visualizza_alert(service):
    print("\n--- ALERT CRITICI ATTIVI ---")
    alert = service.ottieni_alert()

    if not alert:
        print("Nessun alert critico attivo")
        return

    print(f"\nTrovati {len(alert)} alert critici:\n")
    for row in alert:
        print(f"Paziente: {row[2]} (ID: {row[1]})")
        print(f"Data: {row[8]}")
        print(f"Livello Rischio: {row[9]}")
        print(f"Allerta Medico: {'Si' if row[11] else 'No'}")
        print(f"Glicemia: {row[11]} mg/dL")
        print(f"Raccomandazioni: {row[10]}")
        print("=" * 80)


def area_medico(service):
    print("\n--- AREA MEDICO ---")
    print("1. Visualizza pazienti con rischio ALTO")
    print("2. Visualizza tutti gli alert")
    print("3. Cerca paziente specifico")
    print("0. Torna al menu principale")

    scelta = input("\nScelta: ").strip()

    if scelta == "1":
        pazienti_critici = service.ottieni_pazienti_critici()
        if not pazienti_critici:
            print("Nessun paziente con rischio alto")
            return
        print(f"\nPazienti con rischio ALTO: {len(pazienti_critici)}\n")
        for p in pazienti_critici:
            print(f"Paziente: {p[2]} (ID: {p[1]})")
            print(f"Ultima rilevazione: {p[8]}")
            print(f"Raccomandazioni: {p[10]}")
            print("-" * 80)
    elif scelta == "2":
        visualizza_alert(service)
    elif scelta == "3":
        visualizza_storico(service)


def mostra_statistiche(service):
    print("\n--- STATISTICHE SISTEMA ---")
    stats = service.ottieni_statistiche()
    print("\nStatistiche Generali:")
    print(f"Totale Interazioni: {stats['totale_interazioni']}")
    print(f"Pazienti Unici: {stats['pazienti_unici']}")
    print(f"Alert Critici Attivi: {stats['alert_critici']}")
    print(f"Pazienti a Rischio Alto: {stats['rischio_alto']}")
    print(f"Pazienti a Rischio Medio: {stats['rischio_medio']}")
    print(f"Pazienti a Rischio Basso: {stats['rischio_basso']}")


def _decodifica_stato(idx):
    bins = []
    rest = idx
    for i in range(5, -1, -1):
        div = 1
        for j in range(i):
            div *= N_BINS[j]
        b = rest // div
        rest %= div
        bins.append(b)
    bins.reverse()
    return bins


def _label_bin(i, j):
    s = SOGLIE[i]
    if j == 0:
        return f"<{s[0]}"
    elif j == N_BINS[i] - 1:
        return f">={s[N_BINS[i]-2]}"
    else:
        return f"{s[j-1]}-{s[j]-1}"


def mostra_riepilogo_rl(agent):
    NOMI_PARAM = ["PA sis", "PA dia", "FC", "Temp", "SpO2", "Glic"]

    print("\n" + "=" * 60)
    print("=== RIEPILOGO AGENTE RL ===")
    print("=" * 60)

    print("\nSafety override (soglie critiche):")
    print("  PA >= 180 o PA dia >= 110     -> alto (crisi ipertensiva)")
    print("  PA sis < 90 e FC > 100         -> alto (shock)")
    print("  FC >= 130 o FC <= 45           -> alto")
    print("  SpO2 < 88                      -> alto (ipossia severa)")
    print("  Temperatura >= 39.5            -> alto (ipertermia critica)")
    print("  Glicemia <= 55 o >= 250        -> alto")

    non_zero = np.argwhere(agent.q_table != 0)
    stati_visti = set(s for s, _ in non_zero)
    totale_stati = N_BINS[0] * N_BINS[1] * N_BINS[2] * N_BINS[3] * N_BINS[4] * N_BINS[5]

    print(
        f"\nPolicy appresa (Q-table -- {len(stati_visti)} stati con valore non-zero):"
    )
    print(
        f"  {'Stato':>6} | {'PA sis':>6} | {'PA dia':>6} | {'FC':>5} | {'Temp':>6} | {'SpO2':>4} | {'Glic':>7} | Azione"
    )
    print(
        f"  {'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*5}-+-{'-'*6}-+-{'-'*4}-+-{'-'*7}-+----------------"
    )

    for idx in sorted(stati_visti):
        bins = _decodifica_stato(idx)
        azione = AZIONI[int(np.argmax(agent.q_table[idx]))]
        desc = " | ".join(_label_bin(i, bins[i]) for i in range(6))
        print(f"  {idx:>6d} | {desc} | {azione}")

    non_appresi = totale_stati - len(stati_visti)
    print(f"\nDefault per i {non_appresi} stati non appresi: monitoring")
    print("=" * 60 + "\n")


def addestra_rl():
    from telemedicina.agents.rl_environment import SimPatientEnv

    print("=" * 60)
    print("ADDESTRAMENTO AGENTE RL (Q-LEARNING)")
    print("=" * 60)

    agent = QLearningAgent()
    env = SimPatientEnv()
    finestra = 500

    print(f"\nParametri:")
    print(f"  Episodi: {config.RL_EPISODI}")
    print(f"  Alpha: {agent.alpha}  |  Gamma: {agent.gamma}")
    print(f"  Epsilon init: {agent.epsilon}  |  Decay: {agent.epsilon_decay}")
    print(
        f"\n{'Episodio':>8} | {'Reward medio':>12} | {'Epsilon':>7} | {'Q-table size':>11}"
    )
    print("-" * 50)

    rewards = []
    for ep in range(1, config.RL_EPISODI + 1):
        severita = env.reset()
        parametri = env.parametri
        reward_ep = 0
        for _ in range(3):
            stato = agent.discretizza(parametri)
            azione = agent.scegli_azione(stato, training=True)
            reward, nuova_severita, done = env.step(azione)
            parametri = env.parametri
            stato_next = agent.discretizza(parametri)
            agent.impara(stato, azione, reward, stato_next, done)
            reward_ep += reward
            if done:
                break
        agent.decadi_epsilon()
        rewards.append(reward_ep)
        if ep % finestra == 0:
            media = np.mean(rewards[-finestra:])
            non_zero = int(np.count_nonzero(agent.q_table))
            print(f"{ep:>8} | {media:>+12.2f} | {agent.epsilon:>6.3f} | {non_zero:>8}")

    media_finale = np.mean(rewards[-finestra:])
    print("-" * 50)
    print(f"\nTraining completato! Reward media finale: {media_finale:+.2f}")
    agent.salva_q_table()
    print(f"Q-table salvata in: {config.RL_QTABLE_PATH}")


def main(modo_rl=False, force_retrain=False):
    print("Inizializzazione Sistema di Telemedicina...")

    if modo_rl:
        print("Modalita: RL (Q-learning)")
        qtable_path = config.RL_QTABLE_PATH
        if force_retrain or not os.path.exists(qtable_path):
            if force_retrain:
                print("Forza retrain richiesto. Avvio training...")
            else:
                print("Q-table non trovata. Avvio training automatico...")
            addestra_rl()
        agent = QLearningAgent()
        agent.carica_q_table()
        mostra_riepilogo_rl(agent)
        service = AnalysisService(agente_tipo="rl")
    else:
        print("Modalita: Rule-based")
        service = AnalysisService(agente_tipo="rule")

    print("Sistema pronto!\n")

    while True:
        stampa_menu_principale()
        scelta = input("\nInserisci la tua scelta: ").strip()

        if scelta == "1":
            inserisci_parametri(service)
        elif scelta == "2":
            visualizza_storico(service)
        elif scelta == "3":
            visualizza_alert(service)
        elif scelta == "4":
            area_medico(service)
        elif scelta == "5":
            mostra_statistiche(service)
        elif scelta == "0":
            print("\nGrazie per aver utilizzato il Sistema di Telemedicina!")
            service.chiudi()
            sys.exit(0)
        else:
            print("Scelta non valida!")

        input("\nPremi INVIO per continuare...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSistema terminato dall'utente")
        sys.exit(0)
    except Exception as e:
        print(f"\nErrore critico: {e}")
        sys.exit(1)
