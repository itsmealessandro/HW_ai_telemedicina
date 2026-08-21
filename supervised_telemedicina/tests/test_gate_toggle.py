"""tests/test_gate_toggle.py — Toggle del safety gate (modalità diagnostica).

Verifica che:
- ``SupervisedAgent.predict(..., gate=False)`` bypassi il gate: l'MLP decide
  sempre (classe finale = classe MLP, nessun merge né override, nessun
  fallback sulla soglia) e la classe delle regole resta in ``classe_regola``;
  con ``gate=True`` (default) il comportamento storico è invariato;
- ``trace_analysis(..., gate=False)`` non si fermi mai al gate: ``rete``
  calcolata anche per input critici, ``safety_gate.attivo=False``, percorso
  "mlp"; con ``gate=True`` la traccia storica è invariata (``stopped_at``);
- l'endpoint ``POST /api/valuta`` accetti il flag opzionale ``"gate"`` nel
  corpo JSON e usi ``gate_default`` quando assente.

Nessuna dipendenza da artifact su disco: l'MLP minimo è iniettato via
``modello=`` (softmax quasi one-hot pilotata dal bias b2).
"""

import json
import sys
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = PROJECT_ROOT / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import genera_dashboard  # noqa: E402
from telemedicina_supervised.agents.supervised_agent import (  # noqa: E402
    _MESSAGGI_CLASSE,
    SupervisedAgent,
)
from telemedicina_supervised.agents.trace import trace_analysis  # noqa: E402
from telemedicina_supervised.ml.mlp import MLP  # noqa: E402
from telemedicina_supervised.ml.scaler import StandardScaler  # noqa: E402


BASE = {
    "pressione_sistolica": 120.0,
    "pressione_diastolica": 80.0,
    "frequenza_cardiaca": 70.0,
    "temperatura": 36.8,
    "saturazione_ossigeno": 98.0,
    "glicemia": 95.0,
}

# Caso critico per le regole (sistolica in zona critica): il gate storico
# lo blocca con classe "alto" senza consultare l'MLP.
CRITICO = dict(BASE, pressione_sistolica=190.0, frequenza_cardiaca=60.0)


def _modello(indice_classe: int = 0) -> MLP:
    """MLP minimo valido: softmax quasi one-hot sulla classe ``indice_classe``.

    W1 e W2 nulli + bias b2 dominante su una sola classe: la rete ignora
    l'input e risponde sempre la stessa classe, con confidenza sopra la
    soglia di incertezza (~0.99). Lo scaler costante è valido (dev=1).
    """
    modello = MLP(6, 2, ("basso", "medio", "alto"), seed=41)
    modello.W1.fill(0.0)
    modello.W2.fill(0.0)
    modello.b1.fill(0.0)
    modello.b2[:] = 0.0
    modello.b2[indice_classe] = 5.0
    modello.scaler = StandardScaler().fit(np.ones((3, 6)))
    return modello


class TestPredictSenzaGate(unittest.TestCase):
    def test_gate_false_decide_solo_mlp(self):
        # Regole dicono "alto", MLP dice "basso": in modalità diagnostica
        # vince l'MLP, le regole restano registrate per il confronto.
        agente = SupervisedAgent(modello=_modello(0))
        esito = agente.predict(CRITICO, gate=False)
        self.assertEqual(esito.classe, "basso")
        self.assertEqual(esito.classe_mlp, "basso")
        self.assertEqual(esito.classe_regola, "alto")
        self.assertTrue(esito.modello_usato)
        self.assertFalse(esito.fallback)
        self.assertIsNone(esito.motivo_fallback)
        self.assertFalse(esito.override_sicurezza)
        self.assertIsNotNone(esito.probabilita)
        self.assertIsNotNone(esito.confidenza)
        # Messaggio pulito: nessun pattern delle regole in modalità diagnostica.
        self.assertEqual(esito.messaggio, _MESSAGGI_CLASSE["basso"])

    def test_gate_false_mlp_critico_resta_alto(self):
        # Anche l'MLP che dice "alto" produce classe "alto" (coerenza).
        agente = SupervisedAgent(modello=_modello(2))
        esito = agente.predict(CRITICO, gate=False)
        self.assertEqual(esito.classe, "alto")
        self.assertEqual(esito.classe_mlp, "alto")
        self.assertEqual(esito.classe_regola, "alto")

    def test_gate_false_salta_soglia_incertezza(self):
        # Bias tutti nulli -> confidenza uniforme (~0.33) sotto soglia:
        # con gate=False NON scatta il fallback sulla soglia.
        agente = SupervisedAgent(modello=_modello_uniforme())
        esito = agente.predict(BASE, gate=False)
        self.assertTrue(esito.modello_usato)
        self.assertFalse(esito.fallback)
        self.assertIsNone(esito.motivo_fallback)

    def test_gate_default_blocca_critico(self):
        # Comportamento storico: caso critico -> classe "alto", MLP bypassato.
        agente = SupervisedAgent(modello=_modello(0))
        esito = agente.predict(CRITICO)
        self.assertEqual(esito.classe, "alto")
        self.assertIsNone(esito.classe_mlp)
        self.assertFalse(esito.modello_usato)
        self.assertIsNone(esito.probabilita)
        self.assertTrue(esito.fallback)
        self.assertIn("safety gate", esito.motivo_fallback or "")

    def test_gate_false_modello_mancante_degrada_su_regole(self):
        # Stesso degrado del percorso non critico: fallback sulle regole.
        with tempfile.TemporaryDirectory() as tmp:
            agente = SupervisedAgent(modello_path=Path(tmp) / "mancante.npz")
            esito = agente.predict(CRITICO, gate=False)
            self.assertEqual(esito.classe, "alto")
            self.assertFalse(esito.modello_usato)
            self.assertTrue(esito.fallback)
            self.assertIn("non trovato", esito.motivo_fallback or "")


def _modello_uniforme() -> MLP:
    """MLP con logits nulli: distribuzione uniforme (confidenza ~0.33)."""
    modello = MLP(6, 2, ("basso", "medio", "alto"), seed=41)
    modello.W1.fill(0.0)
    modello.W2.fill(0.0)
    modello.b1.fill(0.0)
    modello.b2[:] = 0.0
    modello.scaler = StandardScaler().fit(np.ones((3, 6)))
    return modello


class TestTraceSenzaGate(unittest.TestCase):
    def test_gate_false_non_si_ferma_al_gate(self):
        traccia = trace_analysis(
            CRITICO, agent=SupervisedAgent(modello=_modello(0)), gate=False
        )
        self.assertNotIn("stopped_at", traccia)
        self.assertIs(traccia["safety_gate"]["attivo"], False)
        self.assertEqual(traccia["safety_gate"]["classe_regola"], "alto")
        # La rete viene calcolata e mostrata anche per input critici.
        self.assertIsNotNone(traccia.get("rete"))
        self.assertIn("probabilita", traccia["rete"])
        self.assertEqual(traccia["risposta"]["classe"], "basso")
        self.assertEqual(traccia["risposta"]["percorso"], "mlp")
        # La soglia di incertezza resta visibile (info dal outcome).
        self.assertIn("soglia_incertezza", traccia)

    def test_gate_default_traccia_invariata(self):
        traccia = trace_analysis(CRITICO, agent=SupervisedAgent(modello=_modello(0)))
        self.assertEqual(traccia["stopped_at"], "gate")
        self.assertIs(traccia["safety_gate"]["attivo"], True)
        self.assertEqual(traccia["risposta"]["classe"], "alto")
        self.assertEqual(traccia["risposta"]["percorso"], "gate")


class TestApiValutaGate(unittest.TestCase):
    """POST /api/valuta: flag "gate" nel corpo + gate_default dell'handler."""

    def _server(self, handler):
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, thread

    def _post(self, porta, corpo):
        richiesta = urllib.request.Request(
            f"http://127.0.0.1:{porta}/api/valuta",
            data=json.dumps(corpo).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(richiesta, timeout=5) as risposta:
            self.assertEqual(risposta.status, 200)
            return json.loads(risposta.read().decode("utf-8"))

    def test_gate_false_nel_corpo(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            handler = genera_dashboard.crea_handler(
                tmp_path / "analisi.db",
                tmp_path,
                agent=SupervisedAgent(modello=_modello(0)),
            )
            server, thread = self._server(handler)
            try:
                porta = server.server_address[1]
                traccia = self._post(porta, {"parametri": CRITICO, "gate": False})
                self.assertIs(traccia["safety_gate"]["attivo"], False)
                self.assertNotIn("stopped_at", traccia)
                self.assertEqual(traccia["risposta"]["percorso"], "mlp")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_senza_gate_nel_corpo_usa_default_true(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            handler = genera_dashboard.crea_handler(
                tmp_path / "analisi.db",
                tmp_path,
                agent=SupervisedAgent(modello=_modello(0)),
            )
            server, thread = self._server(handler)
            try:
                porta = server.server_address[1]
                traccia = self._post(porta, {"parametri": CRITICO})
                self.assertIs(traccia["safety_gate"]["attivo"], True)
                self.assertEqual(traccia["stopped_at"], "gate")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_gate_default_disattivato_da_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            handler = genera_dashboard.crea_handler(
                tmp_path / "analisi.db",
                tmp_path,
                agent=SupervisedAgent(modello=_modello(0)),
                gate_default=False,
            )
            server, thread = self._server(handler)
            try:
                porta = server.server_address[1]
                traccia = self._post(porta, {"parametri": CRITICO})
                self.assertIs(traccia["safety_gate"]["attivo"], False)
                self.assertNotIn("stopped_at", traccia)
                # Il corpo può comunque riattivare il gate per richiesta.
                traccia_on = self._post(porta, {"parametri": CRITICO, "gate": True})
                self.assertIs(traccia_on["safety_gate"]["attivo"], True)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
