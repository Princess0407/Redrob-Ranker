from __future__ import annotations
import argparse
import json
import logging
import math
import os
import pickle
import sys
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap — allow src/ modules to be found when running as a script
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))       # .../scripts
_PROJECT_ROOT = os.path.dirname(_SCRIPTS_DIR)                    # .../
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
for _p in [_SRC_DIR, _PROJECT_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    import subprocess
    print("rank_bm25 module not found, installing rank-bm25...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rank-bm25==0.2.2"])
    from rank_bm25 import BM25Okapi

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [precompute] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def tokenize_candidate(candidate: dict) -> List[str]:
    """
    Build a BM25-indexable token list from a candidate record.
    Combines: skill names, career descriptions, headline, summary.

    Defensive: handles missing/null fields gracefully.
    """
    tokens = []

    # Skill names (weighted naturally by BM25 tf)
    for skill in (candidate.get("skills") or []):
        name = (skill.get("name") or "").strip()
        if name:
            tokens.extend(name.lower().split())

    # Career history descriptions
    for ch in (candidate.get("career_history") or []):
        desc = (ch.get("description") or "").strip()
        title = (ch.get("title") or "").strip()
        if desc:
            tokens.extend(desc.lower().split())
        if title:
            tokens.extend(title.lower().split())

    # Headline (summary is dropped entirely as it is almost entirely templated noise)
    profile = candidate.get("profile") or {}
    headline = (profile.get("headline") or "").strip()
    if headline:
        tokens.extend(headline.lower().split())

    # Certification names
    for cert in (candidate.get("certifications") or []):
        name = (cert.get("name") or "").strip()
        if name:
            tokens.extend(name.lower().split())

    return tokens


def stream_build_bm25_corpus(
    candidates_path: str,
    max_candidates: Optional[int] = None,
) -> Tuple[List[str], List[List[str]], int]:
    """
    Stream-read candidates.jsonl and build the BM25 corpus.

    Returns:
        (candidate_ids, tokenized_corpus, malformed_count)
    """
    candidate_ids = []
    corpus = []
    malformed_count = 0
    total_lines = 0

    logger.info("Building BM25 corpus from %s ...", candidates_path)
    t0 = time.time()

    with open(candidates_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            total_lines += 1

            try:
                candidate = json.loads(line)
            except json.JSONDecodeError as e:
                malformed_count += 1
                logger.warning("Malformed JSON at line %d (skipped): %s", line_num, e)
                continue

            cid = candidate.get("candidate_id")
            if not cid:
                malformed_count += 1
                logger.warning("Missing candidate_id at line %d (skipped)", line_num)
                continue

            tokens = tokenize_candidate(candidate)
            candidate_ids.append(cid)
            corpus.append(tokens)

            if line_num % 10000 == 0:
                elapsed = time.time() - t0
                logger.info(
                    "  Tokenized %d/%s candidates in %.1fs...",
                    line_num, max_candidates or "?", elapsed
                )

            if max_candidates and len(candidate_ids) >= max_candidates:
                break

    elapsed = time.time() - t0
    logger.info(
        "Corpus built: %d candidates, %d malformed lines, %.1fs",
        len(candidate_ids), malformed_count, elapsed
    )
    return candidate_ids, corpus, malformed_count


def build_bm25_index(corpus: List[List[str]]):
    """Build BM25 index from tokenized corpus. Returns BM25Okapi object."""
    logger.info("Building BM25Okapi index on %d documents...", len(corpus))
    t0 = time.time()
    bm25 = BM25Okapi(corpus)
    elapsed = time.time() - t0
    logger.info("BM25 index built in %.1fs", elapsed)
    return bm25


def compute_offline_weak_labels(
    candidates_path: str,
    jd_config,
    candidate_ids_set: set,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Compute weak labels for training WITHOUT using bm25_score (non-circularity guarantee).

    Label formula (Section 6):
        weak_label = hard_req_coverage × consistency_score

    bm25_score is EXPLICITLY EXCLUDED from label construction.

    Returns:
        (weak_labels_dict, hard_req_scores_dict, consistency_scores_dict)
    """
    from features import (
        c1_timeline_impossibility, c2_signup_anomaly, c3_salary_inversion,
        c4_assessment_contradiction, c5_engagement_mismatch,
        consistency_score as compute_consistency
    )
    from jd_parser import hard_req_coverage_score

    logger.info("Computing offline weak labels (no bm25_score)...")
    t0 = time.time()

    weak_labels = {}
    hard_req_scores = {}
    consistency_scores = {}
    processed = 0

    with open(candidates_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue

            cid = candidate.get("candidate_id")
            if not cid or cid not in candidate_ids_set:
                continue

            # Hard requirement coverage — uses skills and career text only
            hrc = hard_req_coverage_score(candidate, jd_config)

            # Consistency score — bm25_score=0.0 (excluded), median=0.0 (c5 won't fire)
            # Per non-circularity: c5 engagement mismatch is disabled at label-gen time
            # because we have no stage1 BM25 scores yet.
            c1 = c1_timeline_impossibility(candidate)
            c2 = c2_signup_anomaly(candidate)
            c3 = c3_salary_inversion(candidate)
            c4 = c4_assessment_contradiction(candidate)
            # c5 requires BM25 scores — skipped at offline label generation
            # The model will learn this signal from the feature itself
            cons = c1 * c2 * c3 * c4  # 4-check product (c5 excluded offline)

            # Pull adversarial functions — these are JD fit signals, not data integrity
            from features import (
                detect_description_title_mismatch,
                score_langchain_dabbler,
                score_title_skill_discontinuity,
                score_cv_speech_specialist,
            )

            # Compute consulting fraction inline (same logic as Param_F)
            consulting_m = sum(
                float(r.get("duration_months") or 0)
                for r in (candidate.get("career_history") or [])
                if r.get("industry", "") in {"IT Services", "Consulting", "Professional Services", "BPO"}
                and r.get("company_size", "") == "10001+"
            )
            total_m = sum(
                float(r.get("duration_months") or 0)
                for r in (candidate.get("career_history") or [])
            )
            cons_frac = (consulting_m / total_m) if total_m > 0 else 0.0

            jd_penalty = max(0.0, 1.0 - (
                0.90 * score_langchain_dabbler(candidate) +
                0.85 * score_title_skill_discontinuity(candidate) +
                0.75 * float(cons_frac > 0.95) +
                0.65 * float(detect_description_title_mismatch(candidate) > 0.5) +
                0.55 * score_cv_speech_specialist(candidate)
            ))

            wl = hrc * cons * jd_penalty

            hard_req_scores[cid] = hrc
            consistency_scores[cid] = cons
            weak_labels[cid] = wl

            processed += 1
            if processed % 10000 == 0:
                logger.info("  Weak labels: %d computed...", processed)

    elapsed = time.time() - t0
    logger.info(
        "Weak labels computed: %d candidates in %.1fs", len(weak_labels), elapsed
    )
    logger.info(
        "Label stats: min=%.4f, max=%.4f, mean=%.4f, >0: %d",
        min(weak_labels.values()),
        max(weak_labels.values()),
        sum(weak_labels.values()) / max(1, len(weak_labels)),
        sum(1 for v in weak_labels.values() if v > 0),
    )
    return weak_labels, hard_req_scores, consistency_scores


def extract_training_features(
    candidates_path: str,
    candidate_ids: List[str],
    jd_config,
    hard_req_scores: Dict[str, float],
    consistency_scores: Dict[str, float],
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract the full 22-feature matrix for all indexed candidates.
    bm25_score is set to 0.0 for all candidates at training time.

    Returns:
        (feature_matrix: np.ndarray of shape [N, 22], ordered_ids)
    """
    from features import build_feature_vector, FEATURE_COLUMNS

    logger.info("Extracting 22-feature matrix for %d candidates...", len(candidate_ids))
    t0 = time.time()

    cid_set = set(candidate_ids)
    feature_rows = {}

    with open(candidates_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue

            cid = candidate.get("candidate_id")
            if not cid or cid not in cid_set:
                continue

            try:
                fv = build_feature_vector(
                    candidate, jd_config,
                    bm25_score=0.0,  # Always 0 at training time
                    stage1_bm25_median=0.0,
                )
            except Exception as e:
                logger.warning("Feature extraction failed for %s: %s", cid, e)
                fv = {col: 0.0 for col in FEATURE_COLUMNS}

            feature_rows[cid] = [fv[col] for col in FEATURE_COLUMNS]

            if len(feature_rows) % 10000 == 0:
                logger.info("  Features: %d extracted...", len(feature_rows))

    # Build ordered matrix matching candidate_ids order
    matrix = []
    ordered_ids = []
    for cid in candidate_ids:
        if cid in feature_rows:
            matrix.append(feature_rows[cid])
            ordered_ids.append(cid)

    X = np.array(matrix, dtype=np.float32)
    elapsed = time.time() - t0
    logger.info(
        "Feature matrix shape: %s in %.1fs", X.shape, elapsed
    )
    return X, ordered_ids


def train_lightgbm(
    X: np.ndarray,
    weak_labels: Dict[str, float],
    ordered_ids: List[str],
    precomputed_dir: str,
) -> None:
    """
    Train LightGBM with objective='lambdarank' and eval_at=[5, 10, 50].

    LightGBM lambdarank has a hard limit of max_position (<=10000) rows per query.
    With 100K candidates, we split into multiple query groups of GROUP_SIZE each.
    Each group simulates a "mini-query" with the same JD — the model still learns
    to rank candidates by relevance within each group, then generalizes across groups.

    Labels are discretized to integer bins [0, 1, 2, 3] for lambdarank.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        import subprocess
        import sys
        logger.info("lightgbm module not found, installing lightgbm...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm==4.3.0"])
        import lightgbm as lgb
    from features import FEATURE_COLUMNS

    logger.info("Training LightGBM LambdaRank model...")
    t0 = time.time()

    # Get labels in the same order as feature matrix
    y_raw = np.array([weak_labels.get(cid, 0.0) for cid in ordered_ids], dtype=np.float32)

    # Discretize continuous labels to 4 levels: [0, 1, 2, 3]
    # 0: weak_label == 0 (failures)
    # 1: 0 < label <= 0.33
    # 2: 0.33 < label <= 0.66
    # 3: label > 0.66
    y_int = np.zeros(len(y_raw), dtype=np.int32)
    y_int[y_raw > 0] = 1
    y_int[y_raw > 0.33] = 2
    y_int[y_raw > 0.66] = 3

    logger.info(
        "Label distribution: 0=%d, 1=%d, 2=%d, 3=%d",
        (y_int == 0).sum(), (y_int == 1).sum(),
        (y_int == 2).sum(), (y_int == 3).sum()
    )

    # LightGBM lambdarank limit: max 10,000 rows per query group.
    # Split 100K candidates into groups of GROUP_SIZE.
    # Shuffle first so groups are stratified (not all positives in one group).
    GROUP_SIZE = 5000  # well within the 10K limit
    n = len(ordered_ids)

    # Shuffle indices to distribute positives evenly across groups
    rng = np.random.default_rng(seed=42)
    shuffle_idx = rng.permutation(n)
    X_shuffled = X[shuffle_idx]
    y_shuffled = y_int[shuffle_idx]

    # Build group sizes
    n_groups = (n + GROUP_SIZE - 1) // GROUP_SIZE  # ceiling division
    group = []
    for i in range(n_groups):
        start = i * GROUP_SIZE
        end = min(start + GROUP_SIZE, n)
        group.append(end - start)

    logger.info(
        "LambdaRank: %d candidates split into %d query groups of size ~%d",
        n, n_groups, GROUP_SIZE
    )

    train_data = lgb.Dataset(
        X_shuffled, label=y_shuffled,
        group=group,
        feature_name=FEATURE_COLUMNS,
    )

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [5, 10, 50],
        "num_leaves": 63,
        "learning_rate": 0.05,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    # Train without validation set (no split needed — we train on full data)
    model = lgb.train(
        params,
        train_data,
        num_boost_round=200,
        valid_sets=[train_data],
        callbacks=[
            lgb.log_evaluation(period=50),
            lgb.early_stopping(stopping_rounds=20, verbose=False),
        ],
    )

    elapsed = time.time() - t0
    logger.info("LightGBM training complete in %.1fs", elapsed)

    # Feature importance log
    importances = dict(zip(FEATURE_COLUMNS, model.feature_importance(importance_type="gain")))
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    logger.info("Top 5 feature importances (gain):")
    for fname, imp in sorted_imp[:5]:
        logger.info("  %s: %.2f", fname, imp)

    # Save model
    model_path = os.path.join(precomputed_dir, "lgbm_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info("LightGBM model saved to %s", model_path)


def save_artifacts(
    precomputed_dir: str,
    bm25,
    candidate_ids: List[str],
    weak_labels: Dict[str, float],
) -> None:
    """Save BM25 index, candidate IDs, and weak labels to precomputed/."""
    os.makedirs(precomputed_dir, exist_ok=True)

    bm25_path = os.path.join(precomputed_dir, "bm25_index.pkl")
    ids_path = os.path.join(precomputed_dir, "candidate_ids.pkl")
    labels_path = os.path.join(precomputed_dir, "weak_labels.pkl")

    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)
    logger.info("BM25 index saved: %s (%.1f MB)", bm25_path,
                os.path.getsize(bm25_path) / 1e6)

    with open(ids_path, "wb") as f:
        pickle.dump(candidate_ids, f)
    logger.info("Candidate IDs saved: %s (%d IDs)", ids_path, len(candidate_ids))

    with open(labels_path, "wb") as f:
        pickle.dump(weak_labels, f)
    logger.info("Weak labels saved: %s", labels_path)


def main(candidates_path: str, base_dir: str) -> None:
    """Main precomputation pipeline."""
    precomputed_dir = os.path.join(base_dir, "precomputed")
    data_dir = os.path.join(base_dir, "data")
    aliases_path = os.path.join(data_dir, "skill_aliases.json")

    os.makedirs(precomputed_dir, exist_ok=True)

    # Validate inputs
    if not os.path.isfile(candidates_path):
        logger.error("Candidates file not found: %s", candidates_path)
        sys.exit(1)
    if not os.path.isfile(aliases_path):
        logger.error("skill_aliases.json not found: %s", aliases_path)
        sys.exit(1)

    logger.info("=== Precompute Pipeline Starting ===")
    logger.info("Candidates: %s", candidates_path)
    logger.info("Base dir: %s", base_dir)
    t_total = time.time()

    # Step 1: Parse JD config
    from jd_parser import parse_jd
    jd_config = parse_jd(aliases_path)
    logger.info(
        "JD config: %d hard reqs, %d preferred reqs",
        len(jd_config.hard_requirements),
        len(jd_config.preferred_requirements)
    )

    # Step 2: Build BM25 corpus and index
    candidate_ids, corpus, malformed_count = stream_build_bm25_corpus(candidates_path)
    bm25 = build_bm25_index(corpus)

    # Free corpus from memory (no longer needed)
    del corpus

    # Step 3: Compute weak labels (NO bm25_score — non-circularity guarantee)
    candidate_ids_set = set(candidate_ids)
    weak_labels, hard_req_scores, consistency_scores = compute_offline_weak_labels(
        candidates_path, jd_config, candidate_ids_set
    )

    # Step 4: Save BM25 index + metadata
    save_artifacts(precomputed_dir, bm25, candidate_ids, weak_labels)

    # Step 5: Extract 22-feature matrix for training
    X, ordered_ids = extract_training_features(
        candidates_path, candidate_ids, jd_config, hard_req_scores, consistency_scores
    )

    # Step 6: Train LightGBM
    train_lightgbm(X, weak_labels, ordered_ids, precomputed_dir)

    total_elapsed = time.time() - t_total
    logger.info("=== Precompute Complete in %.1fs ===", total_elapsed)
    logger.info("Artifacts in: %s", precomputed_dir)

    # Print summary
    artifact_sizes = {}
    for fname in ["bm25_index.pkl", "candidate_ids.pkl", "weak_labels.pkl", "lgbm_model.pkl"]:
        fpath = os.path.join(precomputed_dir, fname)
        if os.path.isfile(fpath):
            artifact_sizes[fname] = os.path.getsize(fpath) / 1e6

    logger.info("Artifact sizes (MB):")
    for fname, size_mb in artifact_sizes.items():
        logger.info("  %s: %.1f MB", fname, size_mb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Offline pre-computation: BM25 indexing + LightGBM training"
    )
    parser.add_argument(
        "--candidates",
        default=os.path.join(_PROJECT_ROOT, "candidates.jsonl"),
        help="Path to candidates JSONL file (default: project_root/candidates.jsonl)",
    )
    parser.add_argument(
        "--base-dir",
        default=_PROJECT_ROOT,
        help="Base directory for data/ and precomputed/ (default: project root)",
    )
    args = parser.parse_args()

    candidates_path = os.path.abspath(args.candidates)
    base_dir = os.path.abspath(args.base_dir)

    main(candidates_path, base_dir)
