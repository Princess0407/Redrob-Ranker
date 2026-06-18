#!/bin/bash

set -e

MODE="${1:-full}"

CANDIDATES_PATH="${CANDIDATES_PATH:-/app/candidates.jsonl}"
OUT_PATH="${OUT_PATH:-/app/out/submission.csv}"
BASE_DIR="${BASE_DIR:-/app}"

echo "=== Redrob Ranking System ==="
echo "Mode: $MODE"
echo "Candidates: $CANDIDATES_PATH"
echo "Output: $OUT_PATH"
echo "Base dir: $BASE_DIR"

case "$MODE" in
    full)
        echo ""
        echo "--- Step 1: Precompute (BM25 index + LightGBM training) ---"
        python /app/scripts/precompute.py \
            --candidates "$CANDIDATES_PATH" \
            --base-dir "$BASE_DIR"

        echo ""
        echo "--- Step 2: Rank (produce submission.csv) ---"
        python /app/src/rank.py \
            --candidates "$CANDIDATES_PATH" \
            --out "$OUT_PATH" \
            --base-dir "$BASE_DIR"

        echo ""
        echo "--- Step 3: Validate submission.csv ---"
        python /app/scripts/validate_submission.py --submission "$OUT_PATH"
        ;;

    precompute)
        python /app/scripts/precompute.py \
            --candidates "$CANDIDATES_PATH" \
            --base-dir "$BASE_DIR"
        ;;

    rank)
        python /app/src/rank.py \
            --candidates "$CANDIDATES_PATH" \
            --out "$OUT_PATH" \
            --base-dir "$BASE_DIR"
        ;;

    validate)
        python /app/scripts/validate_submission.py --submission "$OUT_PATH"
        ;;

    *)
        echo "Unknown mode: $MODE"
        echo "Usage: $0 [full|precompute|rank|validate]"
        exit 1
        ;;
esac

echo ""
echo "=== Done ==="
