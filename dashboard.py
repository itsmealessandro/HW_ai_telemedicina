"""
dashboard.py - Dashboard web per il sistema di telemedicina.

Usage:
  streamlit run dashboard.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import numpy as np
import pandas as pd

from telemedicina.agents.rl_agent import QLearningAgent, AZIONI, N_BINS, SOGLIE
from telemedicina.database.db_manager import DatabaseManager
from telemedicina import config

st.set_page_config(page_title="Telemedicina – Agente RL", page_icon="🏥", layout="wide")

# ── Helpers ──


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
    return f"{s[j-1]}-{s[j]-1}"


NOMI_PARAM = ["PA sis", "PA dia", "FC", "Temp", "SpO2", "Glic"]


def _colore_qtext(v):
    try:
        val = float(v.split()[0])
    except (ValueError, IndexError):
        return ""
    if val >= 15:
        return "background-color: #1a9850; color: white"
    if val >= 10:
        return "background-color: #66bd63"
    if val >= 5:
        return "background-color: #a6d96a"
    if val >= 0:
        return "background-color: #fee08b"
    if val >= -5:
        return "background-color: #f46d43"
    return "background-color: #a50026; color: white"


def _colore_rischio(v):
    c = {
        "basso": "background-color: #4CAF50",
        "medio": "background-color: #FF9800",
        "alto": "background-color: #F44336; color: white",
    }
    return c.get(v, "")


# ── Cache ──


@st.cache_data
def _load_history():
    p = config.RL_QTABLES_HISTORY_PATH
    if os.path.exists(p):
        return np.load(p)
    return None


@st.cache_data
def _load_patients():
    try:
        db = DatabaseManager(config.DB_PATH)
        cur = db.conn.cursor()
        cur.execute("SELECT * FROM interazioni ORDER BY timestamp DESC")
        rows = [tuple(r) for r in cur.fetchall()]
        db.chiudi()
        return rows
    except Exception:
        return []


@st.cache_data
def _build_final_policy(qtable_bytes):
    non_zero = np.argwhere(qtable_bytes != 0)
    stati = sorted(set(int(s) for s, _ in non_zero))
    rows = []
    for idx in stati:
        bins = _decodifica_stato(idx)
        row = {"Stato": idx}
        for i, nome in enumerate(NOMI_PARAM):
            row[nome] = _label_bin(i, bins[i])
        az = AZIONI[int(np.argmax(qtable_bytes[idx]))]
        row["Azione"] = az
        row["Q-value"] = round(float(np.max(qtable_bytes[idx])), 2)
        rows.append(row)
    return pd.DataFrame(rows).set_index("Stato")


# ── Load once ──

history = _load_history()
patients = _load_patients()

agent = QLearningAgent()
qtable_ok = agent.carica_q_table()
qtable_bytes = agent.q_table.copy() if qtable_ok else None

# ===========================================================================
# PAGE
# ===========================================================================

st.title("Telemedicina – Agente RL")
st.caption("Dati statici: aggiornati solo dopo un nuovo `python main.py --build-qt`.")

st.divider()

# ===========================================================================
# 1. STORICO Q-TABLE
# ===========================================================================

st.header("1. Storico Q-Table")

if history is not None and history.ndim == 3:
    n_snap = history.shape[0]
    non_zero = set()
    for i in range(n_snap):
        for s, _ in np.argwhere(history[i] != 0):
            non_zero.add(int(s))
    stati = sorted(non_zero)

    if not stati:
        st.info("Nessuno stato con valori non-zero.")
    else:
        stato_info = {}
        for s in stati:
            bins = _decodifica_stato(s)
            stato_info[s] = [_label_bin(i, bins[i]) for i in range(6)]

        milestones = [
            f"{int((i + 1) * 100 // n_snap)}%" for i in range(n_snap)
        ]

        tabs = st.tabs(AZIONI)
        for a_idx, tab in enumerate(tabs):
            with tab:
                cols = NOMI_PARAM + milestones
                data = {}
                for pi, nome in enumerate(NOMI_PARAM):
                    data[nome] = [stato_info[s][pi] for s in stati]
                for i, m in enumerate(milestones):
                    vals = [history[i][s][a_idx] for s in stati]
                    if i == 0:
                        data[m] = [f"{v:.2f}" for v in vals]
                    else:
                        prev = [history[i - 1][s][a_idx] for s in stati]
                        data[m] = [
                            f"{v:.2f} \u2191" if v - p > 0.001
                            else f"{v:.2f} \u2193" if p - v > 0.001
                            else f"{v:.2f} ="
                            for v, p in zip(vals, prev)
                        ]
                df = pd.DataFrame(data, index=stati)
                df.index.name = "Stato"
                styled = df.style.map(_colore_qtext, subset=milestones)
                st.dataframe(styled, width="stretch")
else:
    st.info("Nessuna history. Esegui `python main.py --build-qt`.")

st.divider()

# ===========================================================================
# 2. PAZIENTI
# ===========================================================================

st.header("2. Pazienti")

if patients:
    df_pat = pd.DataFrame(
        patients,
        columns=[
            "id", "paziente_id", "nome_paziente",
            "pressione_sistolica", "pressione_diastolica",
            "frequenza_cardiaca", "temperatura", "saturazione_ossigeno",
            "glicemia", "timestamp", "livello_rischio",
            "raccomandazioni", "allerta_medico",
        ],
    ).drop(columns=["id"])
    df_pat["allerta_medico"] = df_pat["allerta_medico"].astype(bool)
    st.dataframe(
        df_pat.style.map(_colore_rischio, subset=["livello_rischio"]),
        width="stretch",
    )
else:
    st.info("Nessun paziente registrato.")

st.divider()

# ===========================================================================
# 3. POLICY FINALE
# ===========================================================================

st.header("3. Policy Finale")

if qtable_ok and qtable_bytes is not None:
    st.dataframe(_build_final_policy(qtable_bytes), width="stretch")

    st.subheader("Safety Override")
    st.markdown(
        """
- PA >= 180 / PA dia >= 110  ->  alto
- PA < 90 e FC > 100  ->  alto (shock)
- FC >= 130 o <= 45  ->  alto
- SpO2 < 88  ->  alto
- T >= 39.5  ->  alto
- Glicemia <= 55 o >= 250  ->  alto
"""
    )
else:
    st.warning("Nessuna policy. Esegui `python main.py --build-qt`.")
