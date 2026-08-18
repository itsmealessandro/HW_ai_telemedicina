"""
test_single_source.py — Guard test per la FONTE UNICA delle soglie.

Scandisce i file .py di `src/` e `training/` (ESCLUSO `safety_rules.py`,
che è la fonte unica, ed esclusi i test) e fallisce se trova i literal
numerici delle soglie usati come costanti nel codice. Obiettivo: impedire
la duplicazione di soglie quando la Fase 2 aggiungerà il generatore del
dataset.

Il set dei valori controllati è auto-manutenuto: i literal noti della
review oracle + tutti i valori estratti dalle costanti di `safety_rules`
(RANGE_FISIOLOGICI, RANGE_NORMALI, SOGLIE_CRITICHE, SOGLIE_MODERATE,
pattern). Se in futuro una soglia cambia o ne viene aggiunta una, il test
la controlla automaticamente.

Escape hatch: una riga che contiene il marcatore `# noqa: soglia derivata`
viene ignorata. Da usare SOLO con una motivazione documentata (es. un
default di dataclass che deriva da safety_rules via import).

Nessun import dal legacy (archive/).
"""

import re
import unittest
from pathlib import Path

from telemedicina_supervised.safety.safety_rules import (
    PATTERN_IPERTERMIA_CRITICA_MIN,
    PATTERN_IPOSSIA_SEVERA_MAX,
    PATTERN_IPOSSIA_SPO2_MAX,
    RANGE_FISIOLOGICI,
    RANGE_NORMALI,
    SOGLIE_CRITICHE,
    SOGLIE_MODERATE,
)

RADICE = Path(__file__).resolve().parents[1]
SRC = RADICE / "src"
TRAINING = RADICE / "training"
FONTE_UNICA = "safety_rules.py"
NOQA = "# noqa: soglia derivata"

# Literal noti delle soglie (dalla review oracle del gate 2).
LITERAL_NOTI = [90, 60, 200, 38.5, 37.5, 95, 100, 140, 80, 50, 120, 93, 40, 300]


def _valori_soglie() -> set:
    """Set dei valori di soglia da controllare (noti + da safety_rules)."""
    valori = set(LITERAL_NOTI)
    for tabella in (RANGE_FISIOLOGICI, RANGE_NORMALI):
        for minimo, massimo in tabella.values():
            valori.add(minimo)
            valori.add(massimo)
    for tabella in (SOGLIE_CRITICHE, SOGLIE_MODERATE):
        for soglie in tabella.values():
            for v in soglie.values():
                if v is not None:
                    valori.add(v)
    valori.update(
        [
            PATTERN_IPOSSIA_SPO2_MAX,
            PATTERN_IPERTERMIA_CRITICA_MIN,
            PATTERN_IPOSSIA_SEVERA_MAX,
        ]
    )
    return valori


def _token(valore: float) -> str:
    """Regex per il literal: 90.0 -> \b90(?:\.0+)?\b ; 38.5 -> \b38\.5\b."""
    if float(valore).is_integer():
        return rf"\b{int(valore)}(?:\.0+)?\b"
    return rf"\b{valore}\b"


class TestFonteUnicaSoglie(unittest.TestCase):
    """Nessuna soglia duplicata fuori da safety_rules.py."""

    def test_nessuna_soglia_duplicata_in_src_e_training(self):
        pattern = re.compile("|".join(_token(v) for v in _valori_soglie()))
        violazioni = []

        for directory in (SRC, TRAINING):
            for file_py in sorted(directory.rglob("*.py")):
                if "__pycache__" in file_py.parts:
                    continue
                if file_py.name == FONTE_UNICA:
                    continue
                for numero, riga in enumerate(
                    file_py.read_text(encoding="utf-8").splitlines(), start=1
                ):
                    if NOQA in riga:
                        continue
                    match = pattern.search(riga)
                    if match:
                        violazioni.append(
                            f"{file_py.relative_to(RADICE)}:{numero}: "
                            f"literal soglia {match.group(0)!r}"
                        )

        self.assertEqual(
            violazioni,
            [],
            "Soglie duplicate fuori da safety_rules.py:\n" + "\n".join(violazioni),
        )


if __name__ == "__main__":
    unittest.main()