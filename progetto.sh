#!/usr/bin/env bash
# progetto.sh — punto d'ingresso unico del progetto.
#
# Gestisce tutte le operazioni comuni dal un solo posto, dalla root del
# repository (funziona da qualsiasi directory corrente):
#
#   ./progetto.sh test              suite principale (129 test) + regressione legacy
#   ./progetto.sh train             training: rigenera artifact MLP + report
#   ./progetto.sh analisi '<json>'  runtime: analizza un paziente via CLI
#   ./progetto.sh dashboard         genera la dashboard statica (data/dashboard.html)
#   ./progetto.sh serve [porta]     dashboard + API su http://127.0.0.1:<porta> (default 8000)
#   ./progetto.sh help              questo aiuto
#
# Nessuna dipendenza oltre a bash e python3.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$ROOT_DIR/supervised_telemedicina"
LEGACY_DIR="$ROOT_DIR/archive/legacy_qtable_rule_based"

# Colori solo su terminale interattivo.
if [ -t 1 ]; then
  BOLD=$'\033[1m'; DIM=$'\033[2m'; RESET=$'\033[0m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'
else
  BOLD=""; DIM=""; RESET=""; GREEN=""; YELLOW=""
fi

info() { printf "%s\n" "${GREEN}==>${RESET} ${BOLD}$*${RESET}"; }
die()  { printf "%s\n" "${YELLOW}ERRORE:${RESET} $*" >&2; exit 1; }

require_app() {
  [ -d "$APP_DIR" ] || die "directory non trovata: $APP_DIR"
  command -v python3 >/dev/null 2>&1 || die "python3 non trovato nel PATH"
}

usage() {
  cat <<EOF
${BOLD}HW_persia_privato — agente intelligente per la telemedicina${RESET}

Uso: ./progetto.sh <comando> [argomenti]

Comandi:
  ${BOLD}test${RESET}                  suite principale (129 test) + regressione legacy (14)
  ${BOLD}train${RESET}                 training supervised: rigenera artifact MLP + report.json
  ${BOLD}analisi${RESET} '<json>'      analizza un paziente (CLI, persiste su SQLite)
  ${BOLD}dashboard${RESET}             genera la dashboard statica in data/dashboard.html
  ${BOLD}serve${RESET} [porta]         dashboard + API interattive (default porta 8000)
  ${BOLD}help${RESET}                  mostra questo aiuto

Esempi:
  ./progetto.sh test
  ./progetto.sh train
  ./progetto.sh analisi '{"pressione_sistolica":120,"pressione_diastolica":80,"frequenza_cardiaca":75,"temperatura":36.8,"saturazione_ossigeno":98,"glicemia":95}'
  ./progetto.sh serve            # poi apri http://127.0.0.1:8000/dashboard.html
  ./progetto.sh serve 8012       # porta personalizzata

Nota: la tab "7. Analisi interattiva" della dashboard richiede ${BOLD}serve${RESET}
(l'endpoint POST /api/valuta esiste solo con il server attivo).
EOF
}

cmd_test() {
  require_app
  info "Suite principale (supervised_telemedicina)"
  (cd "$APP_DIR" && PYTHONPATH=src python3 -m unittest discover -s tests)
  if [ -d "$LEGACY_DIR" ]; then
    info "Regressione legacy (rule-based/Q-learning)"
    (cd "$LEGACY_DIR" && python3 -m tests.test_examples)
  fi
  info "Tutti i test completati."
}

cmd_train() {
  require_app
  info "Training: rigenera artifact MLP + report (puo' richiedere qualche minuto)"
  (cd "$APP_DIR" && python3 main.py --build-ml)
  info "Artifact e report aggiornati."
}

cmd_analisi() {
  require_app
  local json="${1:-}"
  [ -n "$json" ] || die "uso: ./progetto.sh analisi '{\"pressione_sistolica\":120,...}' (JSON con i 6 parametri vitali)"
  info "Analisi runtime (safety gate -> MLP -> fallback; persiste su SQLite)"
  (cd "$APP_DIR" && python3 main.py -ML --parametri "$json")
}

cmd_dashboard() {
  require_app
  info "Generazione dashboard statica"
  (cd "$APP_DIR" && python3 tools/genera_dashboard.py)
}

cmd_serve() {
  require_app
  local porta="${1:-8000}"
  case "$porta" in
    ''|*[!0-9]*) die "porta non valida: $porta (usa un numero, es. 8000)" ;;
  esac
  info "Dashboard su http://127.0.0.1:${porta}/dashboard.html (Ctrl+C per fermare)"
  (cd "$APP_DIR" && PORT="$porta" python3 tools/genera_dashboard.py --serve)
}

case "${1:-}" in
  test)            shift; cmd_test "$@" ;;
  train|training)  cmd_train ;;
  analisi)         shift; cmd_analisi "$@" ;;
  dashboard)       cmd_dashboard ;;
  serve)           shift; cmd_serve "$@" ;;
  help|-h|--help)  usage ;;
  "")              usage ;;
  *)               die "comando sconosciuto: $1 — vedi ./progetto.sh help" ;;
esac
