"""tools/allena_live.py — Allenamento dal vivo dell'MLP (sessione a passi + pagina).

Fornisce:

  - ``TrainingSession``: un allenamento minibatch SGD eseguito A PASSI,
    pensato per essere osservato dal browser in slow-motion (approccio
    "passo-passo via API"): il frontend chiede N passi alla volta e mostra
    come cambiano loss, accuracy, pesi, softmax su un paziente fisso e
    confine di decisione;
  - funzioni di gestione della sessione corrente (una sola, mono-utente,
    protetta da lock per il server threading);
  - ``genera_html_allenamento()``: pagina autonoma ``allenamento.html``
    (CSS e JS inline, nessuna CDN) con controlli Play/Pausa/passo e grafici.

Il runtime resta numpy + stdlib: nessuna nuova dipendenza. Nessuna soglia
clinica è duplicata: le feature/classi arrivano da ``safety_rules``.
"""

import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Percorso assoluto a src/ (il tool vive in tools/, un livello sotto la root).
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from telemedicina_supervised.config import DATA_PROCESSED_DIR  # noqa: E402
from telemedicina_supervised.ml.mlp import MLP  # noqa: E402
from telemedicina_supervised.ml.scaler import StandardScaler  # noqa: E402
from telemedicina_supervised.safety.safety_rules import CLASSI, FEATURE_ORDER  # noqa: E402


class TrainingSession:
    """Un allenamento minibatch SGD eseguibile un passo alla volta.

    Usa un sottoinsieme piccolo del dataset congelato (per rendere ogni
    passo veloce e osservabile), fitta lo scaler SOLO sul train subset
    (come il training reale: nessun leakage) e parte da una rete con pesi
    casuali (inizializzazione He, seed riproducibile).
    """

    def __init__(
        self,
        n_hidden: int = 16,
        lr: float = 0.05,
        batch_size: int = 32,
        n_campioni: int = 2000,
        max_epoche: int = 20,
        seed: int = 41,
    ) -> None:
        self.n_hidden = int(n_hidden)
        self.lr = float(lr)
        self.batch_size = max(1, int(batch_size))
        self.max_epoche = max(1, int(max_epoche))
        self.seed = int(seed)

        # Dataset congelato (generato da training/generate_dataset.py).
        X_train = np.load(DATA_PROCESSED_DIR / "X_train.npy")
        y_train = np.load(DATA_PROCESSED_DIR / "y_train.npy")
        X_val = np.load(DATA_PROCESSED_DIR / "X_val.npy")
        y_val = np.load(DATA_PROCESSED_DIR / "y_val.npy")

        rng = np.random.default_rng(self.seed)
        n_tr = min(max(100, int(n_campioni)), X_train.shape[0])
        idx_tr = rng.choice(X_train.shape[0], size=n_tr, replace=False)
        n_va = min(max(200, n_tr // 4), X_val.shape[0])
        idx_va = rng.choice(X_val.shape[0], size=n_va, replace=False)

        # Scaler fittato SOLO sul sottoinsieme di train.
        self.scaler = StandardScaler().fit(X_train[idx_tr])
        self.X = self.scaler.transform(X_train[idx_tr])
        self.Xv = self.scaler.transform(X_val[idx_va])
        self.y = np.array([CLASSI.index(v) for v in y_train[idx_tr]], dtype=np.int64)
        self.yv = np.array([CLASSI.index(v) for v in y_val[idx_va]], dtype=np.int64)

        # Rete da ZERO: pesi casuali (inizializzazione He dentro MLP).
        self.modello = MLP(
            n_input=self.X.shape[1],
            n_hidden=self.n_hidden,
            classi=list(CLASSI),
            seed=self.seed,
        )
        self.modello.scaler = self.scaler

        self.rng_ordine = np.random.default_rng(self.seed + 7)
        self.ordine = self.rng_ordine.permutation(self.X.shape[0])
        self.pos = 0
        self.epoca = 0  # epoche COMPLETATE
        self.passo_tot = 0
        self.completato = False

        # Paziente fisso per osservare la softmax muoversi passo dopo passo.
        self.esempio = np.array([[120.0, 80.0, 75.0, 36.5, 98.0, 95.0]])

        # Griglia del confine di decisione: sistolica (x) x glicemia (y),
        # altre feature fissate a valori normali.
        self.griglia_n = 28
        gx, gy = np.meshgrid(
            np.linspace(50.0, 250.0, self.griglia_n),
            np.linspace(20.0, 500.0, self.griglia_n),
        )
        Z = np.zeros((self.griglia_n * self.griglia_n, len(FEATURE_ORDER)))
        Z[:, 0] = gx.ravel()
        Z[:, 5] = gy.ravel()
        Z[:, 1] = 80.0   # pressione diastolica
        Z[:, 2] = 75.0   # frequenza cardiaca
        Z[:, 3] = 36.5   # temperatura
        Z[:, 4] = 96.0   # saturazione
        self._griglia_X = Z

        # Serie storica per le curve (appesa a ogni passo).
        self.serie: Dict[str, List[float]] = {
            "passi": [],
            "loss_train": [],
            "loss_val": [],
            "acc_train": [],
            "acc_val": [],
        }
        lt, at, lv, av = self._metriche()
        self._registra(lt, at, lv, av)

    # ------------------------------------------------------------- metriche

    def _metriche(self):
        lt = float(self.modello.loss(self.X, self.y))
        at = float((self.modello.predici_indici(self.X) == self.y).mean())
        lv = float(self.modello.loss(self.Xv, self.yv))
        av = float((self.modello.predici_indici(self.Xv) == self.yv).mean())
        return lt, at, lv, av

    def _registra(self, lt: float, at: float, lv: float, av: float) -> None:
        self.serie["passi"].append(self.passo_tot)
        self.serie["loss_train"].append(round(lt, 5))
        self.serie["loss_val"].append(round(lv, 5))
        self.serie["acc_train"].append(round(at, 5))
        self.serie["acc_val"].append(round(av, 5))

    # ---------------------------------------------------------------- passi

    def passi(self, n: int = 1) -> Dict[str, Any]:
        """Esegue fino a ``n`` mini-batch e restituisce lo stato aggiornato."""
        for _ in range(max(1, int(n))):
            if self.completato:
                break
            if self.pos >= self.ordine.shape[0]:
                self.epoca += 1
                if self.epoca >= self.max_epoche:
                    self.completato = True
                    break
                self.ordine = self.rng_ordine.permutation(self.X.shape[0])
                self.pos = 0
            idxb = self.ordine[self.pos:self.pos + self.batch_size]
            self.pos += self.batch_size
            self.modello.passo(self.X[idxb], self.y[idxb], self.lr)
            self.passo_tot += 1
            lt, at, lv, av = self._metriche()
            self._registra(lt, at, lv, av)
        return self.stato()

    # ---------------------------------------------------------------- stato

    def stato(self) -> Dict[str, Any]:
        """Snapshot completo e serializzabile dello stato corrente."""
        lt, at, lv, av = self._metriche()
        probs = self.modello.probabilità(self.scaler.transform(self.esempio))[0]
        W1 = np.asarray(self.modello.W1, dtype=float)
        W2 = np.asarray(self.modello.W2, dtype=float)
        pred_griglia = self.modello.predici_indici(self.scaler.transform(self._griglia_X))
        epoca_mostrata = (
            self.max_epoche if self.completato
            else min(self.epoca + 1, self.max_epoche)
        )
        return {
            "attiva": True,
            "completato": self.completato,
            "passo": self.passo_tot,
            "epoca": epoca_mostrata,
            "max_epoche": self.max_epoche,
            "passi_per_epoca": int(np.ceil(self.X.shape[0] / self.batch_size)),
            "iperparametri": {
                "n_hidden": self.n_hidden,
                "lr": self.lr,
                "batch_size": self.batch_size,
                "n_campioni": int(self.X.shape[0]),
                "seed": self.seed,
            },
            "loss_train": round(lt, 4),
            "acc_train": round(at, 4),
            "loss_val": round(lv, 4),
            "acc_val": round(av, 4),
            "pesi": {
                "W1": {"media": round(float(W1.mean()), 4), "std": round(float(W1.std()), 4)},
                "W2": {"media": round(float(W2.mean()), 4), "std": round(float(W2.std()), 4)},
            },
            "probabilita_esempio": {c: float(p) for c, p in zip(CLASSI, probs)},
            "confine": {
                "w": self.griglia_n,
                "h": self.griglia_n,
                "x": [50.0, 250.0],
                "y": [20.0, 500.0],
                "classi": pred_griglia.reshape(self.griglia_n, self.griglia_n).tolist(),
            },
            "serie": self.serie,
        }


# ---------------------------------------------------------------------------
# Sessione corrente (mono-utente, protetta per il server threading)
# ---------------------------------------------------------------------------

_LOCK = threading.Lock()
_SESSIONE: Optional[TrainingSession] = None


def avvia_sessione(**kwargs: Any) -> Dict[str, Any]:
    """Crea una nuova sessione (sostituendo l'eventuale precedente)."""
    global _SESSIONE
    with _LOCK:
        _SESSIONE = TrainingSession(**kwargs)
        return _SESSIONE.stato()


def step_sessione(n: int = 1) -> Dict[str, Any]:
    with _LOCK:
        if _SESSIONE is None:
            return {"attiva": False}
        return _SESSIONE.passi(n)


def stato_sessione() -> Dict[str, Any]:
    with _LOCK:
        if _SESSIONE is None:
            return {"attiva": False}
        return _SESSIONE.stato()


def reset_sessione() -> Dict[str, Any]:
    global _SESSIONE
    with _LOCK:
        _SESSIONE = None
    return {"attiva": False}


# ---------------------------------------------------------------------------
# Pagina autonoma allenamento.html (CSS e JS inline, nessuna CDN)
# ---------------------------------------------------------------------------

def genera_html_allenamento() -> str:
    """Genera il documento HTML completo della pagina di training dal vivo."""
    return """<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Allenamento dal vivo — MLP da zero</title>
<style>
* { box-sizing: border-box; }
body {
  font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  margin: 0; background: #f5f6f8; color: #1c1e21;
}
header { background: #0d1b2a; color: #e0e1dd; padding: 24px 32px; }
header h1 { margin: 0 0 6px; font-size: 24px; }
.sottotitolo { margin: 0 0 12px; font-size: 14px; color: #9fb3c8; }
.banner {
  background: #fff3cd; border: 1px solid #ffe08a; color: #664d03;
  border-radius: 6px; padding: 10px 14px; font-size: 14px; font-weight: 600;
}
main { padding: 24px 32px; display: flex; flex-direction: column; gap: 18px; }
.card {
  background: #ffffff; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.12);
  padding: 20px 24px;
}
.card h2 { margin-top: 0; font-size: 18px; color: #0d1b2a; }
.riga-controlli { display: flex; flex-wrap: wrap; gap: 14px; align-items: end; }
.riga-controlli label { display: flex; flex-direction: column; font-size: 12px; color: #1b3a5b; font-weight: 600; gap: 4px; }
.riga-controlli select { padding: 6px 8px; border: 1px solid #cdd6e0; border-radius: 6px; font-size: 13px; }
button {
  background: #1b3a5b; color: #ffffff; border: none; border-radius: 4px;
  padding: 8px 16px; font-size: 13px; cursor: pointer;
}
button:hover { background: #2a4d75; }
button.secondario { background: #6c757d; }
button.secondario:hover { background: #495057; }
button:disabled { opacity: 0.5; cursor: default; }
.nota { font-size: 13px; color: #6c757d; font-style: italic; }
.spiega { font-size: 13px; color: #37474f; margin: 8px 0 0; line-height: 1.5; }
.spiega em { color: #1b3a5b; font-style: italic; }
.contatori { margin-top: 10px; font-size: 15px; font-weight: 700; color: #0d1b2a; }
.vel input { width: 180px; vertical-align: middle; }
.metriche-grid { display: flex; flex-wrap: wrap; gap: 12px; margin: 10px 0; }
.metriche-card {
  flex: 1 1 160px; background: #f0f4f8; border: 1px solid #dde5ee;
  border-radius: 8px; padding: 12px 14px; text-align: center;
}
.metriche-valore { display: block; font-size: 22px; font-weight: 700; color: #0d1b2a; }
.metriche-etichetta { display: block; font-size: 12px; color: #6c757d; margin-top: 4px; }
svg.curva { width: 100%; max-width: 720px; height: auto; background: #fbfcfe; border: 1px solid #e3e6ea; border-radius: 6px; }
.legenda { display: flex; flex-wrap: wrap; gap: 12px; font-size: 11px; color: #495057; margin: 8px 0; }
.legenda .chip { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 4px; vertical-align: -1px; }
.barre-prob { max-width: 420px; }
.barra-riga { display: flex; align-items: center; margin-bottom: 4px; font-size: 12px; }
.barra-etichetta { width: 52px; }
.barra-sfondo { flex: 1; background: #eef1f5; border-radius: 3px; height: 14px; margin: 0 8px; overflow: hidden; }
.barra-piena { height: 100%; background: #1b3a5b; border-radius: 3px; transition: width .3s ease; }
.barra-valore { width: 64px; text-align: right; color: #6c757d; }
.griglia-wrap { display: flex; gap: 18px; align-items: center; flex-wrap: wrap; }
#confine {
  width: 280px; height: 280px; image-rendering: pixelated;
  border: 1px solid #e3e6ea; border-radius: 6px; background: #ffffff;
}
.asse-x { font-size: 11px; color: #6c757d; text-align: center; width: 280px; }
.asse-y { font-size: 11px; color: #6c757d; writing-mode: vertical-rl; transform: rotate(180deg); }
table.pesi { border-collapse: collapse; }
table.pesi th, table.pesi td { border: 1px solid #e3e6ea; padding: 6px 12px; font-size: 13px; text-align: left; }
table.pesi th { background: #eef1f5; }
footer { padding: 16px 32px; font-size: 12px; color: #6c757d; }
footer a { color: #1b3a5b; }
</style>
</head>
<body>
<header>
  <h1>Allenamento dal vivo — l’MLP impara da zero</h1>
  <p class="sottotitolo">Minibatch SGD passo-passo: forward → loss → backpropagation → update dei pesi</p>
  <div class="banner">Dati sintetici: dimostrazione didattica su un sottoinsieme del dataset, nessuna pretesa diagnostica.</div>
</header>
<main>

<section class="card">
  <h2>1. Sessione</h2>
  <div class="riga-controlli">
    <label>Neuroni nascosti
      <select id="n-hidden"><option>8</option><option selected>16</option><option>32</option><option>64</option></select>
    </label>
    <label>Learning rate
      <select id="lr"><option>0.01</option><option selected>0.05</option><option>0.10</option></select>
    </label>
    <label>Campioni
      <select id="n-campioni"><option>1000</option><option selected>2000</option><option>5000</option></select>
    </label>
    <label>Epoche max
      <select id="max-epoche"><option>10</option><option selected>20</option><option>30</option></select>
    </label>
    <button id="btn-avvia">Avvia da zero</button>
    <button id="btn-reset" class="secondario">Reset</button>
  </div>
  <p class="spiega">“Avvia da zero” crea una rete nuova con <em>pesi casuali</em> (inizializzazione
  He): a quel punto la rete non sa nulla e classifica quasi a caso (~33% su 3 classi).
  Cambia iperparametri e riavvia per confrontare comportamenti diversi.</p>
  <span id="stato-sessione" class="nota"></span>
</section>

<section class="card">
  <h2>2. Trasporto</h2>
  <div class="riga-controlli">
    <button id="btn-play">▶ Play</button>
    <button id="btn-passo">1 passo</button>
    <button id="btn-dieci">+10 passi</button>
    <label class="vel">Velocità
      <input type="range" id="vel" min="50" max="1500" step="50" value="400">
      <span id="vel-label">400 ms tra passi</span>
    </label>
  </div>
  <div class="contatori"><span id="cont-passo">Passo 0</span> · <span id="cont-epoca">Epoca —/—</span></div>
  <p class="spiega">Ogni “passo” è un mini-batch: la rete vede 32 esempi, calcola le predizioni
  (forward), misura l’errore (loss), calcola i gradienti (backpropagation) e muove i pesi
  nella direzione che riduce l’errore (update: peso ← peso − lr × gradiente).</p>
</section>

<section class="card">
  <h2>3. Metriche</h2>
  <div class="metriche-grid">
    <div class="metriche-card"><span class="metriche-valore" id="m-loss-train">—</span><span class="metriche-etichetta">Loss train</span></div>
    <div class="metriche-card"><span class="metriche-valore" id="m-acc-train">—</span><span class="metriche-etichetta">Accuracy train</span></div>
    <div class="metriche-card"><span class="metriche-valore" id="m-loss-val">—</span><span class="metriche-etichetta">Loss val</span></div>
    <div class="metriche-card"><span class="metriche-valore" id="m-acc-val">—</span><span class="metriche-etichetta">Accuracy val</span></div>
  </div>
  <p class="spiega">La <em>loss</em> (cross-entropy) misura quanto le probabilità predette distano
  dalle etichette corrette: più è bassa, meglio va. L’<em>accuracy</em> è la percentuale di esempi
  classificati correttamente. Le colonne “val” sono misurate su dati mai usati per aggiornare i
  pesi: se la loss di val smette di scendere mentre quella di train continua, la rete sta
  memorizzando invece di imparare (overfitting).</p>
</section>

<section class="card">
  <h2>4. Curve di apprendimento</h2>
  <svg id="svg-loss" class="curva" viewBox="0 0 640 150" role="img" aria-label="Curve di loss"></svg>
  <div class="legenda">
    <span><span class="chip" style="background:#1565c0"></span>loss train</span>
    <span><span class="chip" style="background:#ef6c00"></span>loss val</span>
  </div>
  <svg id="svg-acc" class="curva" viewBox="0 0 640 150" role="img" aria-label="Curve di accuracy"></svg>
  <div class="legenda">
    <span><span class="chip" style="background:#2e7d32"></span>accuracy train</span>
    <span><span class="chip" style="background:#6a1b9a"></span>accuracy val</span>
  </div>
</section>

<section class="card">
  <h2>5. Paziente fisso — la softmax si concentra</h2>
  <div id="barre-esempio"><p class="nota">Avvia una sessione per vedere le probabilità.</p></div>
  <p class="spiega">Probabilità delle tre classi per un paziente fisso (120/80/75/36.5/98/95).
  All’inizio sono quasi uniformi (~33% ciascuna): la rete non sa nulla. Guardale
  <em>concentrarsi</em> sulla classe giusta man mano che l’allenamento avanza.</p>
</section>

<section class="card">
  <h2>6. Confine di decisione in evoluzione</h2>
  <div class="griglia-wrap">
    <span class="asse-y">glicemia 20 → 500 mg/dL</span>
    <div>
      <canvas id="confine" width="28" height="28"></canvas>
      <div class="asse-x">pressione sistolica 50 → 250 mmHg</div>
    </div>
    <div class="legenda" style="flex-direction:column; align-items:flex-start;">
      <span><span class="chip" style="background:#2e7d32"></span>basso</span>
      <span><span class="chip" style="background:#ef6c00"></span>medio</span>
      <span><span class="chip" style="background:#c62828"></span>alto</span>
    </div>
  </div>
  <p class="spiega">Ogni pixel è la classe predetta dalla rete corrente per una combinazione di
  pressione sistolica (orizzontale) e glicemia (verticale); le altre feature sono fissate a valori
  normali. All’inizio il quadro è rumore caotico: guardalo organizzarsi in regioni nette.</p>
</section>

<section class="card">
  <h2>7. Pesi in movimento</h2>
  <table class="pesi">
    <thead><tr><th>Tensore</th><th>Media</th><th>Deviazione standard</th></tr></thead>
    <tbody>
      <tr><td>W1 (ingresso → nascosto)</td><td id="w1-media">—</td><td id="w1-std">—</td></tr>
      <tr><td>W2 (nascosto → uscita)</td><td id="w2-media">—</td><td id="w2-std">—</td></tr>
    </tbody>
  </table>
  <p class="spiega">All’avvio i pesi sono rumore gaussiano; col training la loro distribuzione si
  struttura. Media e deviazione standard sono un termometro di quanto i parametri si stanno
  muovendo a ogni passo.</p>
</section>

</main>
<footer>Pagina generata localmente — nessuna richiesta di rete esterna. <a href="/dashboard.html">← Torna alla dashboard</a></footer>
<script>
(function () {
  var COLORI_CLASSE = [[46,125,50],[239,108,0],[198,40,40]];
  var serie = null;
  var inGioco = false, timer = null, occupato = false;

  function $(id) { return document.getElementById(id); }
  function statoTxt(t) { $('stato-sessione').textContent = t; }

  async function chiama(percorso, corpo) {
    var r = await fetch('/api/allena/' + percorso, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(corpo || {})
    });
    return r.json();
  }

  function barreProb(prob) {
    if (!prob) return '';
    var ordine = ['basso', 'medio', 'alto'];
    var h = '<div class="barre-prob">';
    ordine.forEach(function (c) {
      var v = prob[c]; if (v === undefined) return;
      var w = Math.max(0, Math.min(100, v * 100));
      h += '<div class="barra-riga"><span class="barra-etichetta">' + c + '</span>'
        + '<div class="barra-sfondo"><div class="barra-piena" style="width:' + w.toFixed(1) + '%"></div></div>'
        + '<span class="barra-valore">' + Number(v).toFixed(4) + '</span></div>';
    });
    return h + '</div>';
  }

  function dueCurve(idSvg, a, b, colA, colB, yMax) {
    var svg = $(idSvg), w = 640, h = 150, pad = 30;
    if (!a || a.length < 2) {
      svg.innerHTML = '<text x="' + (w / 2) + '" y="' + (h / 2) + '" text-anchor="middle" font-size="12" fill="#6c757d">Servono almeno 2 passi</text>';
      return;
    }
    function pts(arr) {
      var s = '';
      for (var i = 0; i < arr.length; i++) {
        var x = pad + (w - 2 * pad) * (i / (arr.length - 1));
        var v = Math.min(arr[i], yMax);
        var y = h - pad - (h - 2 * pad) * (v / yMax);
        s += (i ? ' ' : '') + x.toFixed(1) + ',' + y.toFixed(1);
      }
      return s;
    }
    svg.innerHTML = '<line x1="' + pad + '" y1="' + (h - pad) + '" x2="' + (w - pad) + '" y2="' + (h - pad) + '" stroke="#e3e6ea"/>'
      + '<polyline fill="none" stroke="' + colA + '" stroke-width="2" points="' + pts(a) + '"/>'
      + '<polyline fill="none" stroke="' + colB + '" stroke-width="2" stroke-dasharray="5,4" points="' + pts(b) + '"/>'
      + '<text x="' + pad + '" y="' + (pad - 8) + '" font-size="10" fill="#6c757d">' + yMax + '</text>'
      + '<text x="' + pad + '" y="' + (h - pad + 14) + '" font-size="10" fill="#6c757d">0</text>';
  }

  function disegnaConfine(conf) {
    var cv = $('confine'), ctx = cv.getContext('2d');
    var img = ctx.createImageData(conf.w, conf.h);
    for (var r = 0; r < conf.h; r++) {
      for (var c = 0; c < conf.w; c++) {
        var rgb = COLORI_CLASSE[conf.classi[r][c]] || [200, 200, 200];
        var i = (r * conf.w + c) * 4;
        img.data[i] = rgb[0]; img.data[i + 1] = rgb[1]; img.data[i + 2] = rgb[2]; img.data[i + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }

  function aggiorna(s) {
    serie = s.serie || serie;
    $('m-loss-train').textContent = s.loss_train;
    $('m-acc-train').textContent = s.acc_train;
    $('m-loss-val').textContent = s.loss_val;
    $('m-acc-val').textContent = s.acc_val;
    $('cont-passo').textContent = 'Passo ' + s.passo;
    $('cont-epoca').textContent = 'Epoca ' + s.epoca + '/' + s.max_epoche;
    $('w1-media').textContent = s.pesi.W1.media;
    $('w1-std').textContent = s.pesi.W1.std;
    $('w2-media').textContent = s.pesi.W2.media;
    $('w2-std').textContent = s.pesi.W2.std;
    $('barre-esempio').innerHTML = barreProb(s.probabilita_esempio);
    var mx = 1.2;
    (serie.loss_train || []).concat(serie.loss_val || []).forEach(function (v) { mx = Math.max(mx, v * 1.05); });
    dueCurve('svg-loss', serie.loss_train, serie.loss_val, '#1565c0', '#ef6c00', mx);
    dueCurve('svg-acc', serie.acc_train, serie.acc_val, '#2e7d32', '#6a1b9a', 1);
    disegnaConfine(s.confine);
  }

  function pausa() {
    inGioco = false;
    if (timer) { clearTimeout(timer); timer = null; }
    $('btn-play').textContent = '▶ Play';
  }

  async function ciclo() {
    if (!inGioco) return;
    if (occupato) { timer = setTimeout(ciclo, 30); return; }
    occupato = true;
    try {
      var s = await chiama('step', { n_passi: 1 });
      if (s.errore) { pausa(); statoTxt('Errore: ' + s.errore); return; }
      aggiorna(s);
      if (s.completato) {
        pausa();
        statoTxt('Allenamento completato (' + s.max_epoche + ' epoche). Cambia iperparametri e riavvia per confrontare.');
        return;
      }
    } finally {
      occupato = false;
    }
    timer = setTimeout(ciclo, parseInt($('vel').value, 10));
  }

  function leggiIpereparametri() {
    return {
      n_hidden: parseInt($('n-hidden').value, 10),
      lr: parseFloat($('lr').value),
      batch_size: 32,
      n_campioni: parseInt($('n-campioni').value, 10),
      max_epoche: parseInt($('max-epoche').value, 10)
    };
  }

  function init() {
    $('vel').addEventListener('input', function () {
      $('vel-label').textContent = $('vel').value + ' ms tra passi';
    });

    $('btn-avvia').addEventListener('click', async function () {
      pausa();
      var s = await chiama('avvia', leggiIpereparametri());
      if (s.errore) { statoTxt('Errore: ' + s.errore); return; }
      aggiorna(s);
      statoTxt('Rete inizializzata con pesi casuali: guarda l’accuracy partire da ~0.33. Premi Play.');
    });

    $('btn-reset').addEventListener('click', async function () {
      pausa();
      await chiama('reset', {});
      ['m-loss-train','m-acc-train','m-loss-val','m-acc-val'].forEach(function (id) { $(id).textContent = '—'; });
      ['w1-media','w1-std','w2-media','w2-std'].forEach(function (id) { $(id).textContent = '—'; });
      $('cont-passo').textContent = 'Passo 0';
      $('cont-epoca').textContent = 'Epoca —/—';
      $('barre-esempio').innerHTML = '<p class="nota">Sessione azzerata.</p>';
      $('svg-loss').innerHTML = ''; $('svg-acc').innerHTML = '';
      var ctx = $('confine').getContext('2d');
      ctx.clearRect(0, 0, 28, 28);
      statoTxt('Sessione azzerata.');
    });

    $('btn-play').addEventListener('click', function () {
      if (inGioco) { pausa(); return; }
      inGioco = true;
      $('btn-play').textContent = '⏸ Pausa';
      ciclo();
    });

    $('btn-passo').addEventListener('click', async function () {
      pausa();
      var s = await chiama('step', { n_passi: 1 });
      if (s.attiva === false) { statoTxt('Nessuna sessione attiva: premi “Avvia da zero”.'); return; }
      aggiorna(s);
    });

    $('btn-dieci').addEventListener('click', async function () {
      pausa();
      var s = await chiama('step', { n_passi: 10 });
      if (s.attiva === false) { statoTxt('Nessuna sessione attiva: premi “Avvia da zero”.'); return; }
      aggiorna(s);
    });

    fetch('/api/allena/stato').then(function (r) { return r.json(); }).then(function (s) {
      if (s.attiva) { aggiorna(s); statoTxt('Sessione esistente ripristinata: premi Play per continuare.'); }
      else { statoTxt('Nessuna sessione attiva: configura i parametri e premi “Avvia da zero”.'); }
    }).catch(function () {
      statoTxt('Server non raggiungibile: avvia tutto con ./progetto.sh serve');
    });
  }

  if (document.readyState !== 'loading') init();
  else document.addEventListener('DOMContentLoaded', init);
})();
</script>
</body>
</html>
"""
