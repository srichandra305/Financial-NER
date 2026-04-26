#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# run.sh — Financial NER project launcher
# Usage:
#   ./run.sh install   → install dependencies
#   ./run.sh train     → train ML models
#   ./run.sh serve     → start FastAPI backend
#   ./run.sh frontend  → open the frontend in a browser
#   ./run.sh bert      → fine-tune DistilBERT (GPU recommended)
#   ./run.sh all       → install + train + serve
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

GREEN="\033[0;32m"; CYAN="\033[0;36m"; RESET="\033[0m"

banner() { echo -e "${CYAN}══════════════════════════════════════${RESET}"; echo -e "${GREEN}  $1${RESET}"; echo -e "${CYAN}══════════════════════════════════════${RESET}"; }

case "${1:-help}" in

  install)
    banner "Installing dependencies"
    pip install --upgrade pip
    pip install -r requirements.txt
    echo -e "${GREEN}✓ Done${RESET}"
    ;;

  train)
    banner "Training ML models"
    python models/train.py
    echo -e "${GREEN}✓ Model saved to models/best_model.pkl${RESET}"
    ;;

  serve)
    banner "Starting FastAPI backend  →  http://localhost:8000"
    echo "  API docs: http://localhost:8000/docs"
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
    ;;

  frontend)
    banner "Opening frontend"
    FRONTEND="$PROJECT_DIR/frontend/index.html"
    if command -v xdg-open &>/dev/null; then
      xdg-open "$FRONTEND"
    elif command -v open &>/dev/null; then
      open "$FRONTEND"
    else
      echo "Open this file in your browser: $FRONTEND"
    fi
    ;;

  bert)
    banner "Fine-tuning DistilBERT (this may take a while)"
    python models/bert_finetune.py
    echo -e "${GREEN}✓ BERT model saved to models/bert_ner/${RESET}"
    ;;

  all)
    bash "$0" install
    bash "$0" train
    bash "$0" serve
    ;;

  help|*)
    echo ""
    echo "Usage: ./run.sh <command>"
    echo ""
    echo "  install    Install Python dependencies"
    echo "  train      Train ML NER models"
    echo "  serve      Start the FastAPI backend"
    echo "  frontend   Open the frontend in a browser"
    echo "  bert       Fine-tune DistilBERT (optional)"
    echo "  all        Install + train + serve"
    echo ""
    ;;
esac
