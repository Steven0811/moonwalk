#!/usr/bin/env bash
# Watch the v1 rule-based moonwalk.
#
#   ./watch.sh                      live window, wheel foot, 4 s cycle, 30 s
#   ./watch.sh live  [options]      live Isaac Sim window in real time (for watching only: per hard rule 1 do not
#                                   judge behaviour from the live window; use 'gif' or 'run' for that)
#   ./watch.sh gif   [options]      headless recording -> runs/gait/<run>/walk_<foot>.gif, opened when done
#   ./watch.sh run   [options]      headless run, then print the acceptance table and the torque report
#
# options:  --foot wheel|ptfe (default wheel)   --cycle SECONDS (default 4)   --seconds N (default 30 / 20 for gif)
set -euo pipefail

V1="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB="${ISAACLAB:-$HOME/Desktop/IsaacLab}"
PY="$ISAACLAB/_isaac_sim/python.sh"
# Isaac Lab refuses to run inside conda/venv: strip those variables for every call
CLEAN=(env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_SHLVL -u VIRTUAL_ENV -u PYTHONPATH)

if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then sed -n '2,9p' "$0"; exit 0; fi
mode="${1:-live}"; [[ $# -gt 0 ]] && shift
foot=wheel; cycle=""; seconds=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --foot) foot="$2"; shift 2 ;;
    --cycle) cycle="$2"; shift 2 ;;
    --seconds) seconds="$2"; shift 2 ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "unknown option: $1 (see ./watch.sh --help)"; exit 1 ;;
  esac
done
[[ "$foot" == wheel || "$foot" == ptfe ]] || { echo "--foot must be wheel or ptfe"; exit 1; }
args=(--variant "$foot")
[[ -n "$cycle" ]] && args+=(--cycle "$cycle")

latest_run() { ls -td "$V1"/runs/gait/"$foot"_*"$1" 2>/dev/null | head -1; }

case "$mode" in
  live)
    echo "Opening the Isaac Sim window ($foot foot, cycle ${cycle:-4} s). Close the window or Ctrl+C to stop."
    "${CLEAN[@]}" "$ISAACLAB/isaaclab.sh" -p "$V1/tools/run_gait.py" "${args[@]}" \
      --seconds "${seconds:-30}" --realtime --viz kit --tag live
    ;;
  gif)
    echo "Recording ${seconds:-20} s headless ($foot foot) ..."
    "${CLEAN[@]}" "$ISAACLAB/isaaclab.sh" -p "$V1/tools/run_gait.py" "${args[@]}" \
      --seconds "${seconds:-20}" --gif --tag gif 2>&1 | grep '^\[run_gait\]' || true
    run="$(latest_run gif)"
    gif="$run/walk_$foot.gif"
    if [[ -f "$gif" ]]; then
      echo "GIF: $gif"
      command -v xdg-open >/dev/null && xdg-open "$gif" >/dev/null 2>&1 &
    else
      echo "no GIF written; log lines above"; exit 1
    fi
    ;;
  run)
    echo "Running ${seconds:-60} s headless ($foot foot) ..."
    "${CLEAN[@]}" "$ISAACLAB/isaaclab.sh" -p "$V1/tools/run_gait.py" "${args[@]}" \
      --seconds "${seconds:-60}" --tag watch 2>&1 | grep '^\[run_gait\]' || true
    run="$(latest_run watch)"
    "${CLEAN[@]}" "$PY" "$V1/tools/validate.py" "$run" | grep -v '^Warning'
    echo
    "${CLEAN[@]}" "$PY" "$V1/tools/measure_torque.py" "$run" | grep -v '^Warning'
    ;;
  *)
    echo "unknown mode '$mode' (live | gif | run); see ./watch.sh --help"; exit 1 ;;
esac
