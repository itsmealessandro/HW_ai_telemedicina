"""
dashboard.py - Dashboard di monitoraggio实时 pazienti con agente RL.

Usage:
  streamlit run dashboard.py
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd

from telemedicina.monitoring import PatientMonitor
from telemedicina.agents.rl_agent import AZIONI
from telemedicina import config

st.set_page_config(page_title="Telemedicina – Monitoraggio", page_icon="🏥", layout="wide")

# ── Avvio monitor ──

monitor = PatientMonitor(num_pazienti=15)
if not monitor.agente_pronto:
    st.error("Q-table non trovata. Esegui `python main.py --build-qt`.")
    st.stop()

monitor.avvia()

# ── Auto-refresh ──

try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=3000, key="monitor_refresh")
except ImportError:
    st.caption("ℹ️ Installa `streamlit-autorefresh` per aggiornamento automatico.")

# ── Stato corrente ──

stato = monitor.get_stato()
stats = monitor.get_statistiche()

# ── Sidebar / Navigazione ──

st.sidebar.title("🏥 Telemedicina")
pagina = st.sidebar.radio("Vista", ["Panoramica", "Dettaglio Paziente"])

paziente_selezionato = st.sidebar.selectbox(
    "Paziente",
    options=[p["paziente_id"] for p in stato],
    format_func=lambda pid: next(
        (p["nome"] for p in stato if p["paziente_id"] == pid), pid
    ),
    key="paziente_sel",
) if stato else None

st.sidebar.divider()
st.sidebar.metric("Cicli di aggiornamento", stats["cicli"])

# ── Helper colore ──

COLORE_RISCHIO = {
    "basso": "#4CAF50",
    "medio": "#FF9800",
    "alto": "#F44336",
}

COLORE_AZIONE = {
    "monitoring": "#4CAF50",
    "contatta_medico": "#FF9800",
    "pronto_soccorso": "#E65100",
    "emergenza": "#D32F2F",
}

ETICHETTA_AZIONE = {
    "monitoring": "🔍 Monitoraggio",
    "contatta_medico": "📞 Contatta Medico",
    "pronto_soccorso": "🚑 Pronto Soccorso",
    "emergenza": "🚨 Emergenza",
}


def _badge(testo, colore):
    return f'<span style="background:{colore};color:white;padding:2px 10px;border-radius:12px;font-weight:600;font-size:0.85em">{testo}</span>'


# ===========================================================================
# 1. PANORAMICA
# ===========================================================================

if pagina == "Panoramica":
    st.title("📊 Panoramica Pazienti")
    st.caption("Aggiornato ogni 3 secondi — I parametri variano gradualmente nel tempo.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Totale Pazienti", stats["totale"])
    col2.metric("🟢 Rischio Basso", stats["rischio_basso"])
    col3.metric("🟠 Rischio Medio", stats["rischio_medio"])
    col4.metric("🔴 Rischio Alto", stats["rischio_alto"])

    st.divider()

    stato_sorted = sorted(
        stato,
        key=lambda p: (
            0 if p["livello_rischio"] == "alto" else
            1 if p["livello_rischio"] == "medio" else 2
        ),
    )

    cols_per_row = 3
    for i in range(0, len(stato_sorted), cols_per_row):
        riga = stato_sorted[i : i + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, p in zip(cols, riga):
            rischio = p["livello_rischio"]
            azione = p["azione"]
            colore_r = COLORE_RISCHIO.get(rischio, "#999")
            colore_a = COLORE_AZIONE.get(azione, "#999")

            with col:
                st.markdown(
                    f"""
                    <div style="border:1px solid #ddd;border-radius:12px;padding:16px;margin-bottom:12px;
                                border-left:6px solid {colore_r};">
                        <div style="display:flex;justify-content:space-between;align-items:center;">
                            <strong style="font-size:1.1em">{p['nome']}</strong>
                            {_badge(rischio.upper(), colore_r)}
                        </div>
                        <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px 16px;margin:10px 0;font-size:0.9em;">
                            <span>PA <b>{p['pressione_sistolica']:.0f}/{p['pressione_diastolica']:.0f}</b></span>
                            <span>FC <b>{p['frequenza_cardiaca']:.0f}</b> bpm</span>
                            <span>T <b>{p['temperatura']:.1f}</b> °C</span>
                            <span>SpO2 <b>{p['saturazione_ossigeno']:.0f}</b>%</span>
                            <span>Glicemia <b>{p['glicemia']:.0f}</b></span>
                        </div>
                        <div style="margin-top:6px;">
                            {_badge(ETICHETTA_AZIONE.get(azione, azione), colore_a)}
                            {' 🔔 Allerta Medico' if p['allerta_medico'] else ''}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if st.button("Dettaglio →", key=f"go_{p['paziente_id']}", use_container_width=True):
                    st.session_state["pagina"] = "Dettaglio Paziente"
                    st.session_state["paziente_sel"] = p["paziente_id"]
                    st.rerun()

# ===========================================================================
# 2. DETTAGLIO PAZIENTE
# ===========================================================================

elif pagina == "Dettaglio Paziente" and paziente_selezionato:
    dettaglio = monitor.get_paziente(paziente_selezionato)

    if dettaglio is None:
        st.warning("Paziente non trovato.")
        st.stop()

    rischio = dettaglio["livello_rischio"]
    azione = dettaglio["azione"]
    colore_r = COLORE_RISCHIO.get(rischio, "#999")

    st.title(f"👤 {dettaglio['nome']}")
    st.caption(f"ID: {dettaglio['paziente_id']} · Severità nascosta: {dettaglio['severita_label']}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Livello Rischio", rischio.upper(), delta_color="inverse")
    col2.metric("Azione Consigliata", ETICHETTA_AZIONE.get(azione, azione))
    col3.metric("Allerta Medico", "🔔 Attiva" if dettaglio["allerta_medico"] else "✅ Nessuna")

    st.divider()

    # ── Parametri vitali ──

    st.subheader("📋 Parametri Vitali (ultima rilevazione)")

    PA_NORMALI = {
        "pressione_sistolica": (90, 140),
        "pressione_diastolica": (60, 90),
        "frequenza_cardiaca": (60, 100),
        "temperatura": (36.0, 37.5),
        "saturazione_ossigeno": (95, 100),
        "glicemia": (70, 140),
    }

    parametri_visuali = [
        ("Pressione Sistolica", f"{dettaglio['pressione_sistolica']:.1f} mmHg", dettaglio["pressione_sistolica"], 50, 250, 90, 140),
        ("Pressione Diastolica", f"{dettaglio['pressione_diastolica']:.1f} mmHg", dettaglio["pressione_diastolica"], 30, 150, 60, 90),
        ("Frequenza Cardiaca", f"{dettaglio['frequenza_cardiaca']:.1f} bpm", dettaglio["frequenza_cardiaca"], 30, 200, 60, 100),
        ("Temperatura", f"{dettaglio['temperatura']:.1f} °C", dettaglio["temperatura"], 34, 42, 36.0, 37.5),
        ("Saturazione O2", f"{dettaglio['saturazione_ossigeno']:.1f}%", dettaglio["saturazione_ossigeno"], 70, 100, 95, 100),
        ("Glicemia", f"{dettaglio['glicemia']:.1f} mg/dL", dettaglio["glicemia"], 20, 500, 70, 140),
    ]

    row1 = st.columns(3)
    row2 = st.columns(3)
    for i, (nome, label, valore, vmin, vmax, rmin, rmax) in enumerate(parametri_visuali):
        col = row1[i] if i < 3 else row2[i - 3]
        with col:
            progresso = (valore - vmin) / (vmax - vmin)
            progresso = max(0.0, min(1.0, progresso))
            pos_norm = (rmin - vmin) / (vmax - vmin)
            pos_norm_end = (rmax - vmin) / (vmax - vmin)

            in_range = rmin <= valore <= rmax
            colore_bar = "#4CAF50" if in_range else "#F44336"
            bg_color = "#e0e0e0"

            bar_html = f"""
            <div style="margin:12px 0;">
                <div style="display:flex;justify-content:space-between;font-size:0.9em;">
                    <strong>{nome}</strong>
                    <span style="font-weight:600;color:{colore_bar};">{label}</span>
                </div>
                <div style="position:relative;height:24px;background:{bg_color};border-radius:12px;margin:4px 0;overflow:hidden;">
                    <div style="position:absolute;left:{pos_norm*100:.1f}%;right:{100-pos_norm_end*100:.1f}%;height:100%;
                                background:rgba(76,175,80,0.25);border-radius:2px;"></div>
                    <div style="width:{progresso*100:.1f}%;height:100%;background:{colore_bar};border-radius:12px;
                                transition:width 0.5s;min-width:4px;"></div>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:0.75em;color:#666;">
                    <span>{vmin}</span>
                    <span style="color:#4CAF50;">norma {rmin}-{rmax}</span>
                    <span>{vmax}</span>
                </div>
            </div>
            """
            st.markdown(bar_html, unsafe_allow_html=True)

    st.divider()

    # ── Raccomandazione ──

    st.subheader("💡 Raccomandazione")
    rec_col1, rec_col2 = st.columns([3, 1])
    with rec_col1:
        st.info(dettaglio["raccomandazioni"])
    with rec_col2:
        azione_bt = ETICHETTA_AZIONE.get(azione, azione)
        st.markdown(
            f"<div style='text-align:center;padding:12px;background:{COLORE_AZIONE.get(azione,'#999')};"
            f"color:white;border-radius:12px;font-weight:600;'>{azione_bt}</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Storico azioni ──

    st.subheader("📜 Cronologia Azioni")
    storico = dettaglio.get("storico", [])
    if storico:
        df_storico = pd.DataFrame([
            {
                "Ora": s["timestamp"],
                "PA Sist": f"{s['parametri'].pressione_sistolica:.1f}",
                "PA Dia": f"{s['parametri'].pressione_diastolica:.1f}",
                "FC": f"{s['parametri'].frequenza_cardiaca:.1f}",
                "Temp": f"{s['parametri'].temperatura:.1f}",
                "SpO2": f"{s['parametri'].saturazione_ossigeno:.1f}",
                "Glicemia": f"{s['parametri'].glicemia:.1f}",
                "Rischio": s["livello_rischio"].capitalize(),
                "Azione": s["azione"],
            }
            for s in reversed(storico)
        ])
        st.dataframe(df_storico, use_container_width=True, hide_index=True)
    else:
        st.info("Nessuna cronologia disponibile.")

    st.divider()

    # ── Grafico trend ──

    st.subheader("📈 Trend Parametri")
    if storico:
        df_trend = pd.DataFrame([
            {
                "Ora": s["timestamp"],
                "Pressione Sistolica": s["parametri"].pressione_sistolica,
                "Pressione Diastolica": s["parametri"].pressione_diastolica,
                "Frequenza Cardiaca": s["parametri"].frequenza_cardiaca,
                "Temperatura": s["parametri"].temperatura,
                "SpO2": s["parametri"].saturazione_ossigeno,
                "Glicemia": s["parametri"].glicemia,
            }
            for s in storico
        ])
        df_trend["Ora"] = pd.to_datetime(df_trend["Ora"])
        df_trend = df_trend.set_index("Ora")

        trend_sel = st.multiselect(
            "Seleziona parametri da mostrare",
            options=df_trend.columns.tolist(),
            default=["Pressione Sistolica", "Frequenza Cardiaca", "SpO2"],
        )
        if trend_sel:
            st.line_chart(df_trend[trend_sel])
        else:
            st.info("Seleziona almeno un parametro per visualizzare il trend.")
    else:
        st.info("Dati insufficienti per il grafico.")
