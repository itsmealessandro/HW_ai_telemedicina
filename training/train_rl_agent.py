"""
train_rl_agent.py - Addestra l'agente RL (Q-learning) sull'ambiente simulato.

Mostra l'evoluzione della reward media per episodio,
dimostrando l'apprendimento dell'agente.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from telemedicina.agents.rl_agent import QLearningAgent
from telemedicina.agents.rl_environment import SimPatientEnv
from telemedicina import config
import numpy as np


def main():
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
    print(f"\n{'Episodio':>8} | {'Reward medio':>12} | {'Epsilon':>7} | {'Q-table size':>11}")
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
    print("\nOra puoi selezionare l'agente RL dal menu principale.")


if __name__ == "__main__":
    main()
