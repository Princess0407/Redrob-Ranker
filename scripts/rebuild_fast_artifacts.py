from __future__ import annotations

import json
import logging
import os
import pickle
import sys
import time

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))       
_PROJECT_ROOT = os.path.dirname(_SCRIPTS_DIR)                    
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
for _p in [_SRC_DIR, _PROJECT_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE_DIR = _PROJECT_ROOT
PRECOMPUTED_DIR = os.path.join(BASE_DIR, "precomputed")
CANDIDATES_PATH = os.path.join(BASE_DIR, "candidates.jsonl")



def build_numpy_bm25_artifacts(bm25, precomputed_dir: str) -> None:
    """
    Build scipy sparse BM25 score matrix from an existing BM25Okapi object.

    Saves:
      vocab.pkl        - {term: row_index}  mapping (tiny, fast to load)
      bm25_matrix.npz  - scipy sparse CSR (vocab_size × n_docs), float32
                         Each entry [term_idx, doc_idx] = precomputed
                         idf(term) × bm25_tf_adjusted(term, doc)

    Scoring at runtime:
      q_vec (1 × vocab_size) @ bm25_matrix (vocab_size × n_docs)
      → (1 × n_docs) dense result in a single scipy sparse op (<10 ms).
    """
    try:
        from scipy.sparse import coo_matrix, save_npz
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy"])
        from scipy.sparse import coo_matrix, save_npz

    logger.info("Building NumPy sparse BM25 matrix …")
    t0 = time.perf_counter()

    k1: float = getattr(bm25, "k1", 1.5)
    b: float  = getattr(bm25, "b",  0.75)
    avgdl: float = float(bm25.avgdl)
    doc_len_arr = np.array(bm25.doc_len, dtype=np.float32)
    n_docs: int = int(bm25.corpus_size)

    # term -> row index 
    vocab: dict = {term: idx for idx, term in enumerate(bm25.idf.keys())}
    idf_array = np.array([bm25.idf[term] for term in vocab], dtype=np.float32)
    n_vocab: int = len(vocab)

    logger.info("  vocab_size=%d  n_docs=%d", n_vocab, n_docs)

    rows_list: list = []
    cols_list: list = []
    data_list: list = []

    checkpoint = max(1, n_docs // 10)
    for doc_idx, doc_freq_dict in enumerate(bm25.doc_freqs):
        dl = float(doc_len_arr[doc_idx])
        denom_k = k1 * (1.0 - b + b * dl / avgdl)
        for term, tf in doc_freq_dict.items():
            term_idx = vocab.get(term)
            if term_idx is None:
                continue
            tf_f = float(tf)
            tf_adj = (tf_f * (k1 + 1.0)) / (tf_f + denom_k)
            rows_list.append(term_idx)
            cols_list.append(doc_idx)
            data_list.append(float(idf_array[term_idx]) * tf_adj)
        if doc_idx % checkpoint == 0 and doc_idx > 0:
            logger.info("  … %d / %d docs processed", doc_idx, n_docs)

    nnz = len(data_list)
    logger.info("  COO built: nnz=%d  (%.1f s)", nnz, time.perf_counter() - t0)

    bm25_matrix = coo_matrix(
        (
            np.array(data_list, dtype=np.float32),
            (np.array(rows_list, dtype=np.int32),
             np.array(cols_list, dtype=np.int32)),
        ),
        shape=(n_vocab, n_docs),
    ).tocsr()

    elapsed = time.perf_counter() - t0
    logger.info("  CSR matrix: shape=%s  nnz=%d  (%.1f s total)",
                bm25_matrix.shape, bm25_matrix.nnz, elapsed)

    vocab_path  = os.path.join(precomputed_dir, "vocab.pkl")
    matrix_path = os.path.join(precomputed_dir, "bm25_matrix.npz")

    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
    save_npz(matrix_path, bm25_matrix)

    logger.info("  Saved vocab.pkl (%d terms)", n_vocab)
    logger.info("  Saved bm25_matrix.npz  (%.1f MB)",
                os.path.getsize(matrix_path) / 1e6)



def build_candidate_offset_index(candidates_path: str, precomputed_dir: str) -> None:
    """
    Scan candidates.jsonl once in binary mode and record the byte offset of
    each candidate_id.

    Saves candidate_offsets.pkl: {candidate_id: byte_offset}

    At runtime Stage 2 uses f.seek(offset) + f.readline() for each of the
    ~8500 stage-1 candidates instead of streaming all 487 MB.  Reduces
    Stage 2 from ~4 s to ~0.1–0.3 s.
    """
    logger.info("Building candidate byte-offset index …")
    t0 = time.perf_counter()

    offsets: dict = {}
    size_bytes = os.path.getsize(candidates_path)

    with open(candidates_path, "rb") as f:
        while True:
            offset = f.tell()
            raw_line = f.readline()
            if not raw_line:
                break
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                cid = json.loads(stripped).get("candidate_id")
                if cid:
                    offsets[cid] = offset
            except json.JSONDecodeError:
                pass

            if len(offsets) % 10_000 == 0 and len(offsets) > 0:
                pct = f.tell() / size_bytes * 100
                logger.info("  … %d candidates indexed (%.0f%% of file)", len(offsets), pct)

    elapsed = time.perf_counter() - t0
    logger.info("  Offset index: %d candidates in %.1f s", len(offsets), elapsed)

    out_path = os.path.join(precomputed_dir, "candidate_offsets.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(offsets, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("  Saved candidate_offsets.pkl  (%.1f MB)",
                os.path.getsize(out_path) / 1e6)



def export_lgbm_native(precomputed_dir: str) -> None:
    """
    Re-save lgbm_model.pkl in LightGBM's native text format.
    lgb.Booster(model_file=...) loads ~10-20x faster than pickle.
    """
    import lightgbm as lgb  

    pkl_path = os.path.join(precomputed_dir, "lgbm_model.pkl")
    txt_path = os.path.join(precomputed_dir, "lgbm_model.txt")

    logger.info("Exporting LightGBM model to native text format …")
    t0 = time.perf_counter()

    with open(pkl_path, "rb") as f:
        model = pickle.load(f)

    model.save_model(txt_path)
    logger.info("  Saved lgbm_model.txt  (%.1f MB)  in %.2f s",
                os.path.getsize(txt_path) / 1e6,
                time.perf_counter() - t0)

def main() -> None:
    logger.info("=" * 60)
    logger.info("REBUILD FAST ARTIFACTS")
    logger.info("=" * 60)
    t_total = time.perf_counter()

    bm25_pkl = os.path.join(PRECOMPUTED_DIR, "bm25_index.pkl")
    logger.info("Loading bm25_index.pkl (%.1f MB) …",
                os.path.getsize(bm25_pkl) / 1e6)
    t0 = time.perf_counter()
    with open(bm25_pkl, "rb") as f:
        bm25 = pickle.load(f)
    logger.info("  Loaded in %.2f s", time.perf_counter() - t0)

    build_numpy_bm25_artifacts(bm25, PRECOMPUTED_DIR)


    build_candidate_offset_index(CANDIDATES_PATH, PRECOMPUTED_DIR)

    export_lgbm_native(PRECOMPUTED_DIR)

    logger.info("=" * 60)
    logger.info("ALL ARTIFACTS BUILT in %.1f s", time.perf_counter() - t_total)
    logger.info("New files in precomputed/:")
    for fname in ["vocab.pkl", "bm25_matrix.npz", "candidate_offsets.pkl", "lgbm_model.txt"]:
        fpath = os.path.join(PRECOMPUTED_DIR, fname)
        if os.path.isfile(fpath):
            logger.info("  %-30s  %.1f MB", fname, os.path.getsize(fpath) / 1e6)
    logger.info("rank.py will auto-detect these and use the fast paths.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
