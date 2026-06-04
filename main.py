#!/usr/bin/env python3
"""
main.py - Entry point del Sistema di Telemedicina.

Usage:
  python main.py               # Rule-based
  python main.py -RL           # RL, auto-training se Q-table assente
  python main.py --build-qt    # Solo training Q-table, mostra tempo e riepilogo
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from telemedicina.cli import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sistema di Telemedicina")
    parser.add_argument("-RL", action="store_true", help="Usa agente RL (Q-learning)")
    parser.add_argument(
        "--build-qt",
        action="store_true",
        help="Solo training Q-table, mostra tempo e riepilogo",
    )
    args = parser.parse_args()

    if args.build_qt:
        main(build_qt=True)
    else:
        main(modo_rl=args.RL)
