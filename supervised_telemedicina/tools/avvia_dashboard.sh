#!/usr/bin/env bash
# tools/avvia_dashboard.sh — avvio e chiusura della dashboard.
#
# Fa il build, avvia il server (--serve), apre il browser sull'URL giusto e
# gestisce la chiusura: Ctrl+C o SIGTERM fermano il server con un messaggio
# pulito (nessun traceback). Porta configurabile con PORT=<porta>.
#
# Uso:
#     tools/avvia_dashboard.sh            # build + server + browser
#     PORT=8001 tools/avvia_dashboard.sh  # porta diversa
set -euo pipefail

cd "$(dirname "$0")/.."
PORTA="${PORT:-8000}"

echo "Build della dashboard..."
python tools/genera_dashboard.py

python tools/genera_dashboard.py --serve &
PID=$!
trap 'kill -TERM "$PID" 2>/dev/null' INT TERM HUP EXIT

# Attende che il server risponda (max ~10s) prima di aprire il browser.
READY=0
for _i in $(seq 1 50); do
    if curl -sf "http://127.0.0.1:$PORTA/api/analisi" >/dev/null 2>&1; then
        READY=1
        break
    fi
    if ! kill -0 "$PID" 2>/dev/null; then
        break
    fi
    sleep 0.2
done

if [ "$READY" = 1 ]; then
    echo "Apertura del browser su http://127.0.0.1:$PORTA/dashboard.html"
    xdg-open "http://127.0.0.1:$PORTA/dashboard.html" >/dev/null 2>&1 || true
else
    echo "Avviso: server non pronto entro il timeout (porta $PORTA?)." >&2
fi

# NOTA: niente `wait "$PID"` qui: bash esegue i trap solo a comando finito,
# quindi con wait il SIGINT resterebbe in coda per sempre. Il polling fa
# scattare il trap (che manda TERM al server) entro un ciclo.
while kill -0 "$PID" 2>/dev/null; do
    sleep 0.2
done
wait "$PID" 2>/dev/null || true