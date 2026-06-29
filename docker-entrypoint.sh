#!/bin/bash
set -e

MODE=${1:-full}

if [ "$MODE" = "full" ]; then
    echo "Running full pipeline..."
    python scripts/run_full_pipeline.py --candidates "$CANDIDATES_PATH" --out "$OUT_PATH"
elif [ "$MODE" = "rank" ]; then
    echo "Running ranking only..."
    python src/rank.py --candidates "$CANDIDATES_PATH" --out "$OUT_PATH"
elif [ "$MODE" = "precompute" ]; then
    echo "Running precompute only..."
    python scripts/precompute.py --candidates "$CANDIDATES_PATH" --base-dir "$BASE_DIR"
else
    echo "Unknown mode: $MODE"
    echo "Usage: $0 {full|rank|precompute}"
    exit 1
fi
