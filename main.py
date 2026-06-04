#!/usr/bin/env python3
"""
main.py - Entry point del Sistema di Telemedicina.

Usage:
  python main.py              # Rule-based
  python main.py -RL          # RL, auto-train se necessario
  python main.py -RL --build  # RL, forza retrain
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from telemedicina.cli import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sistema di Telemedicina")
    parser.add_argument("-RL", action="store_true", help="Usa agente RL (Q-learning)")
    parser.add_argument("--build", action="store_true", help="Forza retrain della Q-table")
    args = parser.parse_args()

    main(modo_rl=args.RL, force_retrain=args.build)
