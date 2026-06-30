import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("src"))
sys.path.insert(0, os.path.abspath("scripts"))

from scripts.app import load_jd_config, load_bm25, load_model
from src.features import build_feature_vector, FEATURE_COLUMNS
from src.retrieval import run_dual_pass_retrieval

def main():
    jd_config = load_jd_config()
    bm25, candidate_ids = load_bm25()
    model = load_model()

    cands_to_check = ["CAND_0000014", "CAND_0000043", "CAND_0000082", "CAND_0000034", "CAND_0000002"]
    candidates = []
    
    with open("candidates.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if c.get("candidate_id") in cands_to_check:
                candidates.append(c)

    full_stage1_ids, full_bm25_scores = run_dual_pass_retrieval(bm25, candidate_ids, jd_config)
    median_bm25 = float(np.median(list(full_bm25_scores.values()))) if full_bm25_scores else 0.0

    print(f"\nMedian BM25: {median_bm25}")
    for c in candidates:
        cid = c.get("candidate_id")
        bs = full_bm25_scores.get(cid, 0.0)
        fv = build_feature_vector(c, jd_config, bs, median_bm25)
        row = [fv[col] for col in FEATURE_COLUMNS]
        print(f"\n--- {cid} ---")
        for k, v in fv.items():
            if k in FEATURE_COLUMNS:
                print(f"  {k}: {v}")
        
        # Test lgbm score
        X = np.array([row], dtype=np.float32)
        score = model.predict(X)[0]
        print(f"  --> LGBM raw score: {score:.6f}")

if __name__ == "__main__":
    main()
