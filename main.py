#!/usr/bin/env python3
"""
main.py - Entry point del Sistema di Telemedicina.

Avvia l'interfaccia a riga di comando delegando a telemedicina.cli.main().
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from telemedicina.cli import main

if __name__ == "__main__":
    main()
