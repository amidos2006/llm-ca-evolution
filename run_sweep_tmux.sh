#!/usr/bin/env bash
# Run every combination in the sweep grid inside tmux. sweep.py expands the grid,
# the jobs are dealt round-robin across shards, and each shard gets its own window.
set -euo pipefail

cd "$(dirname "$0")"

SESSION="ca-sweep"
SWEEP_FILE="sweep.yaml"
# 8 shards times fitness_workers 4 oversubscribes 24 vCPUs on purpose: a shard leaves its
# cores idle while it waits on Claude, so the extra workers fill those gaps.
SHARDS=8
LOG_DIR="logs"
ATTACH=0

usage() {
    cat <<'EOF'
Usage: ./run_sweep_tmux.sh [options]
  -s NAME   tmux session name (default: ca-sweep)
  -f FILE   sweep YAML passed to sweep.py (default: sweep.yaml)
  -n COUNT  shards, one tmux window each (default: 8)
  -l DIR    log directory (default: logs)
  -a        attach to the session once the windows are up
  -h        show this help
EOF
}

while getopts "s:f:n:l:ah" option; do
    case "$option" in
        s) SESSION="$OPTARG" ;;
        f) SWEEP_FILE="$OPTARG" ;;
        n) SHARDS="$OPTARG" ;;
        l) LOG_DIR="$OPTARG" ;;
        a) ATTACH=1 ;;
        *) usage; exit 1 ;;
    esac
done

command -v tmux >/dev/null || { echo "tmux is not installed"; exit 1; }
command -v uv >/dev/null || { echo "uv is not installed"; exit 1; }
[ -f "$SWEEP_FILE" ] || { echo "No sweep file at $SWEEP_FILE"; exit 1; }
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session $SESSION already exists, attach with: tmux attach -t $SESSION"
    exit 1
fi

# Never open more windows than there are jobs.
JOBS=$(uv run sweep.py --sweep-file "$SWEEP_FILE" --count)
if [ "$JOBS" -lt 1 ]; then
    echo "The grid is empty"
    exit 1
fi
if [ "$SHARDS" -gt "$JOBS" ]; then
    SHARDS="$JOBS"
fi

mkdir -p "$LOG_DIR"
tmux new-session -d -s "$SESSION" -n "shard-1"
# Keep dead panes around so a crashed shard still shows its traceback.
tmux set-option -t "$SESSION" remain-on-exit on

for index in $(seq 1 "$SHARDS"); do
    if [ "$index" -gt 1 ]; then
        tmux new-window -t "$SESSION" -n "shard-$index"
    fi
    command="uv run sweep.py --sweep-file '$SWEEP_FILE' --shard $index/$SHARDS 2>&1 | tee '$LOG_DIR/shard_$index.log'"
    tmux send-keys -t "$SESSION:shard-$index" "$command" C-m
done

echo "$JOBS job(s) split across $SHARDS shard(s) in session $SESSION"
echo "Logs: $LOG_DIR/shard_*.log"
if [ "$ATTACH" -eq 1 ]; then
    tmux attach -t "$SESSION"
else
    echo "Attach with: tmux attach -t $SESSION"
fi
