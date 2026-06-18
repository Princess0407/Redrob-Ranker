"""
retrieval.py

Dual-Pass BM25 Retrieval per Section 3 of the architecture document.

Stage 1: Load precomputed BM25 index, run two passes:
  Pass A: JD skill aliases (expanded via skill_aliases.json taxonomy)
  Pass B: Production-context keywords (deployed, scale, serving, latency, ...)
  Safety Net: Rare-term pool for niche terms (pinecone, lambdarank)

stage1_candidates = top_5000 ∪ rare_term_pool

No network calls. BM25 index must be precomputed via precompute.py.
"""

from __future__ import annotations

import logging
import os
import pickle
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NumpyBM25 — vectorized scorer backed by a precomputed sparse matrix
# ---------------------------------------------------------------------------

class NumpyBM25:
    """
    Drop-in replacement for BM25Okapi.get_scores() using a precomputed scipy
    sparse matrix of shape (vocab_size, n_docs).

    Each entry [term_idx, doc_idx] stores the precomputed value:
        idf(term) * bm25_tf_adjusted(term, doc)

    Scoring a query is a single sparse matrix-vector multiply:
        q_vec (vocab_size,) @ bm25_matrix (vocab_size × n_docs)
        -> scores (n_docs,)   — sub-10 ms for 214 tokens × 100K docs.

    Compared with BM25Okapi.get_scores():
        BM25Okapi: 214 Python loops × 100K dict lookups = ~9.5 s
        NumpyBM25: one scipy sparse matvec             = ~50 ms
    """

    def __init__(self, vocab: Dict[str, int], bm25_matrix) -> None:
        self.vocab = vocab
        self.bm25_matrix = bm25_matrix        # scipy CSR (vocab_size × n_docs)
        self._n_docs: int  = bm25_matrix.shape[1]
        self._n_vocab: int = bm25_matrix.shape[0]

    def get_scores(self, query_tokens: List[str]) -> np.ndarray:
        """
        Score all documents for a list of query tokens.
        Matches BM25Okapi.get_scores() signature exactly.
        Returns np.ndarray of shape (n_docs,), dtype float32.
        """
        q_vec = np.zeros(self._n_vocab, dtype=np.float32)
        matched = 0
        for t in query_tokens:
            idx = self.vocab.get(t)
            if idx is not None:
                q_vec[idx] = 1.0
                matched += 1
        if matched == 0:
            return np.zeros(self._n_docs, dtype=np.float32)
        # Single sparse matrix-vector multiply — the entire fast path
        return np.asarray(q_vec @ self.bm25_matrix, dtype=np.float32).flatten()


def load_numpy_bm25_artifacts(precomputed_dir: str) -> Optional[NumpyBM25]:
    """
    Load precomputed NumPy BM25 artifacts (vocab.pkl + bm25_matrix.npz).
    Returns a NumpyBM25 instance, or None if the artifacts don't exist yet
    (in which case callers should fall back to bm25_index.pkl).
    """
    vocab_path  = os.path.join(precomputed_dir, "vocab.pkl")
    matrix_path = os.path.join(precomputed_dir, "bm25_matrix.npz")

    if not (os.path.isfile(vocab_path) and os.path.isfile(matrix_path)):
        return None

    try:
        from scipy.sparse import load_npz
        t0 = time.perf_counter()
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
        bm25_matrix = load_npz(matrix_path)
        logger.info(
            "NumPy BM25 loaded: vocab=%d  shape=%s  in %.3f s",
            len(vocab), bm25_matrix.shape, time.perf_counter() - t0,
        )
        return NumpyBM25(vocab, bm25_matrix)
    except Exception as exc:
        logger.warning("Failed to load NumPy BM25 artifacts (%s) — falling back to BM25Okapi", exc)
        return None



def load_bm25_artifacts(precomputed_dir: str) -> Tuple[object, List[str], List[str]]:
    """
    Load the precomputed BM25 index and corpus metadata.

    Args:
        precomputed_dir: Path to the precomputed/ directory.

    Returns:
        (bm25_index, candidate_ids, tokenized_corpus)

    Raises:
        FileNotFoundError: If precomputed artifacts don't exist.
        RuntimeError: If artifacts are corrupted.
    """
    index_path = os.path.join(precomputed_dir, "bm25_index.pkl")
    ids_path = os.path.join(precomputed_dir, "candidate_ids.pkl")

    if not os.path.isfile(index_path):
        raise FileNotFoundError(
            f"BM25 index not found at {index_path}. "
            "Run precompute.py first."
        )
    if not os.path.isfile(ids_path):
        raise FileNotFoundError(
            f"Candidate IDs not found at {ids_path}. "
            "Run precompute.py first."
        )

    try:
        with open(index_path, "rb") as f:
            bm25 = pickle.load(f)
        with open(ids_path, "rb") as f:
            candidate_ids = pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load BM25 artifacts: {e}") from e

    logger.info(
        "BM25 index loaded: %d candidates indexed", len(candidate_ids)
    )
    return bm25, candidate_ids


def tokenize_query(terms: List[str]) -> List[str]:
    """
    Tokenize a list of query terms for BM25.
    Splits multi-word terms, lowercases, deduplicates.
    """
    tokens = []
    for term in terms:
        tokens.extend(term.lower().split())
    return list(set(tokens))


def run_dual_pass_retrieval(
    bm25,
    candidate_ids: List[str],
    jd_config,
    top_n: int = 5000,
) -> Tuple[List[str], Dict[str, float]]:
    """
    Execute dual-pass BM25 retrieval per Section 3.

    Pass A: All JD skill aliases (hard + preferred requirements)
    Pass B: Production-context keywords only
    Safety Net: Rare terms pool (pinecone, lambdarank)

    Returns:
        (stage1_candidate_ids, bm25_scores_dict)
        - stage1_candidate_ids: ordered list (best first) of top_5000 ∪ rare_pool
        - bm25_scores_dict: {candidate_id: float} for all retrieved candidates
    """
    t0 = time.time()

    # --- Pass A: Skill aliases (JD taxonomy expansion) ---
    query_a_terms = jd_config.get_all_query_terms()
    query_a_tokens = tokenize_query(query_a_terms)
    logger.info("Pass A query tokens (%d): %s...", len(query_a_tokens),
                query_a_tokens[:10])

    scores_a = bm25.get_scores(query_a_tokens)

    # --- Pass B: Production-context keywords ---
    query_b_tokens = tokenize_query(jd_config.production_keywords)
    logger.info("Pass B query tokens (%d): %s", len(query_b_tokens), query_b_tokens)

    scores_b = bm25.get_scores(query_b_tokens)

    # Union of scores: take max per candidate (better recall than sum)
    import numpy as np
    combined_scores = np.maximum(scores_a, scores_b)

    # Get top_n indices by score
    top_n_actual = min(top_n, len(candidate_ids))
    top_indices = np.argpartition(combined_scores, -top_n_actual)[-top_n_actual:]
    top_indices = top_indices[np.argsort(combined_scores[top_indices])[::-1]]

    top_candidates = [candidate_ids[i] for i in top_indices]
    top_scores = {candidate_ids[i]: float(combined_scores[i]) for i in top_indices}

    logger.info("Pass A+B union: %d candidates (target %d)", len(top_candidates), top_n)

    # --- Rare-term Safety Net ---
    rare_pool_ids = set()
    rare_pool_scores = {}

    for rare_term in jd_config.rare_terms:
        rare_tokens = tokenize_query([rare_term])
        rare_scores = bm25.get_scores(rare_tokens)
        # Any candidate with non-zero score for rare term qualifies
        rare_nonzero = np.where(rare_scores > 0)[0]
        for idx in rare_nonzero:
            cid = candidate_ids[idx]
            if cid not in top_scores:
                rare_pool_ids.add(cid)
                # Use the rare-term score, or existing if already in pool
                rare_pool_scores[cid] = max(
                    rare_pool_scores.get(cid, 0.0),
                    float(rare_scores[idx])
                )

    logger.info("Rare-term safety net added %d additional candidates", len(rare_pool_ids))

    # Merge: top_5000 ∪ rare_pool (maintain ordering — top candidates first)
    all_scores = {**top_scores, **rare_pool_scores}

    # Re-sort merged list
    all_ordered = sorted(all_scores.keys(), key=lambda cid: all_scores[cid], reverse=True)

    elapsed = time.time() - t0
    logger.info(
        "Dual-pass retrieval complete: %d candidates in %.2fs",
        len(all_ordered), elapsed
    )

    return all_ordered, all_scores


def retrieve_candidate_data(
    stage1_ids: List[str],
    candidates_path: str,
) -> Tuple[List[dict], Set[str]]:
    """
    Stream-read the candidates JSONL file and extract only the Stage 1 candidates.

    Args:
        stage1_ids: Ordered list of candidate IDs from retrieval.
        candidates_path: Path to candidates.jsonl.

    Returns:
        (candidates_list, missing_ids)
        - candidates_list: list of candidate dicts for stage1 IDs (order preserved)
        - missing_ids: IDs that were in stage1_ids but not found in the file
    """
    import json

    stage1_set = set(stage1_ids)
    found: Dict[str, dict] = {}
    malformed_count = 0

    with open(candidates_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError as e:
                malformed_count += 1
                logger.warning(
                    "Malformed JSON at line %d (skipped): %s", line_num, e
                )
                continue

            cid = candidate.get("candidate_id")
            if cid and cid in stage1_set:
                found[cid] = candidate
                if len(found) == len(stage1_set):
                    break  # All found — stop early

    if malformed_count > 0:
        logger.warning("Skipped %d malformed JSONL lines", malformed_count)

    missing_ids = stage1_set - set(found.keys())
    if missing_ids:
        logger.warning(
            "%d stage1 candidates not found in JSONL: %s...",
            len(missing_ids),
            list(missing_ids)[:5]
        )

    # Return in the original stage1 order (preserving BM25 rank)
    ordered = [found[cid] for cid in stage1_ids if cid in found]

    logger.info(
        "Retrieved %d candidate records from JSONL (%d missing)",
        len(ordered), len(missing_ids)
    )
    return ordered, missing_ids


if __name__ == "__main__":
    import sys
    import json
    import os

    base_dir = os.path.dirname(os.path.abspath(__file__))
    precomputed_dir = os.path.join(base_dir, "precomputed")
    candidates_path = os.path.join(base_dir, "candidates.jsonl")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from jd_parser import parse_jd
    jd_config = parse_jd(os.path.join(base_dir, "data", "skill_aliases.json"))

    print("Loading BM25 artifacts...")
    bm25, candidate_ids = load_bm25_artifacts(precomputed_dir)

    print(f"Running dual-pass retrieval on {len(candidate_ids)} indexed candidates...")
    stage1_ids, bm25_scores = run_dual_pass_retrieval(bm25, candidate_ids, jd_config)

    print(f"\nStage 1 retrieved: {len(stage1_ids)} candidates")
    print(f"Top 10 by BM25 score:")
    for i, cid in enumerate(stage1_ids[:10], 1):
        print(f"  {i:2d}. {cid}  score={bm25_scores[cid]:.4f}")

    import numpy as np
    scores = list(bm25_scores.values())
    print(f"\nScore stats: min={min(scores):.4f}, max={max(scores):.4f}, "
          f"median={float(np.median(scores)):.4f}, mean={float(np.mean(scores)):.4f}")
