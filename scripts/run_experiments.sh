#!/usr/bin/env bash
# Launch the full Vietnamese CLIP ablation study inside a tmux session.

set -euo pipefail

SESSION="viet_clip"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_DIR/logs"

mkdir -p "$LOG_DIR"

# Kill existing session if present
tmux kill-session -t "$SESSION" 2>/dev/null || true

tmux new-session -d -s "$SESSION" -c "$REPO_DIR"

tmux send-keys -t "$SESSION" \
  "source .venv/bin/activate && \
  export CUDA_VISIBLE_DEVICES=0 && \
  python run_all.py 2>&1 | tee logs/all_experiments.log" \
  Enter

echo "Experiment launched in tmux session '$SESSION'."
echo "Attach with: tmux attach -t $SESSION"
echo "Logs:        $LOG_DIR/all_experiments.log"
