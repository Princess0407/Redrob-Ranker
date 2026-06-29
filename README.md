# Redrob Hackathon — Intelligent Candidate Discovery & Ranking System

A production-grade, deterministic candidate ranking pipeline for the Redrob *Intelligent Candidate Discovery & Ranking Challenge*. Ranks 100,000 candidates against a structured Job Description in **3.55 seconds** on CPU, with zero external API calls during inference.

---

## System Architecture

```
╔══════════════════════════════════════════════════════════════════════╗
║  OFFLINE PHASE  (one-time, no time/network limit)                   ║
║                                                                      ║
║  candidates.jsonl ──► BM25 Index Build ──► precomputed/bm25_matrix  ║
║                   ──► Static Feature Precompute ──► static_feats.pkl║
║                                                                      ║
║  Gemma3:4b (local Ollama, zero external API)                        ║
║    │  2,500 pairwise comparisons on stratified sample               ║
║    │  Candidate A vs Candidate B → CANDIDATE_A / CANDIDATE_B / TIE  ║
║    ▼                                                                 ║
║  Win/Loss → Elo Ratings → Quartile Labels (0–3)                     ║
║    │                                                                 ║
║    ▼                                                                 ║
║  LightGBM LambdaRank training                                        ║
║    eval_at=[5,10,50], objective=lambdarank                          ║
║    Learns IR-specific skill ordering without explicit programming    ║
║    ──► precomputed/lgbm_model.txt                                    ║
╚══════════════════════════════════════════════════════════════════════╝
                              │
                              ▼ artifacts loaded once at startup
╔══════════════════════════════════════════════════════════════════════╗
║  ONLINE RANKING PHASE  (≤300s wall-clock, CPU-only, zero network)   ║
║                                                                      ║
║  candidates.jsonl (100K)                                            ║
║         │                                                            ║
║         ▼  Stage 1 — Dual-Pass BM25 Retrieval          [0.03s]     ║
║    Pass A: JD skill aliases over skills[].name                      ║
║    Pass B: production keywords over career_history descriptions     ║
║    Rare-term safety net: pinecone, lambdarank, qdrant, bm25         ║
║         │ ~8,500 candidates                                          ║
║         ▼  Stage 2 — Feature Engineering (22 features)  [0.37s]    ║
║    5 adversarial detection functions                                 ║
║    8 JD-specific scoring parameters (A–H)                           ║
║    2 engineered interaction terms                                    ║
║         │                                                            ║
║         ▼  Stage 3 — Logical Consistency                 [inline]   ║
║    consistency_score = c1 × c2 × c3 × c4 × c5                      ║
║    One impossible profile → score collapses to ~0                   ║
║         │                                                            ║
║         ▼  Stage 4 — LightGBM Inference                 [0.01s]    ║
║    raw_score = model.predict(feature_vector)                        ║
║    final_score = raw_score × consistency_score                      ║
║         │ Top 100                                                    ║
║         ▼  Stage 5 — Reasoning Compiler                 [1.77s]    ║
║    4 structural templates (MD5-deterministic rotation)              ║
║    Priority-ranked concern surfacing                                 ║
║    Numeric regex audit + n-gram collision check                     ║
║         │                                                            ║
║         ▼  Pre-CSV Blocking Audits                       [<0.01s]  ║
║    Honeypot audit: assert low_consistency_in_top100 < 10            ║
║    Diversity audit: assert max_employer_share ≤ 30%                 ║
║    Monotonicity assertion                                            ║
║         │                                                            ║
║         ▼                                                            ║
║    submission.csv — 100 ranked candidates                           ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## Quick Start

### Docker (Recommended — matches Stage 3 reproduction environment exactly)

```bash
docker build -t redrob-ranker .
docker run --rm --network none \
  -v $(pwd)/candidates.jsonl:/app/candidates.jsonl \
  -v $(pwd)/out:/app/out \
  redrob-ranker
```

Output: `./out/submission.csv` — 100 ranked candidates, validated and ready to submit.

### Without Docker

```bash
# 1. Create and activate virtualenv
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install pinned dependencies
pip install -r requirements.txt

# 3. Run precomputation (one-time, ~7 minutes on 100K candidates)
python scripts/precompute.py --candidates ./candidates.jsonl --base-dir .

# 4. Run ranking (~3.55 seconds)
python src/rank.py --candidates ./candidates.jsonl --out ./submission.csv

# 5. Validate output format
python scripts/validate_submission.py --submission ./submission.csv
```

**Single-command alternative** (handles artifact caching automatically):

```bash
python scripts/run_full_pipeline.py --candidates ./candidates.jsonl --out ./submission.csv
```

Add `--force-precompute` to bypass the cache and rebuild all artifacts from scratch.

---

## Runtime Performance

| Phase | Module | Operation | Time |
|-------|--------|-----------|------|
| **Offline** | `experiments/pairwise_llm_check/` | Gemma3 pairwise annotation (2,500 pairs, local Ollama) | ~45 min |
| **Offline** | `scripts/precompute.py` | BM25 indexing, static feature precomputation, LightGBM training | ~7 min |
| **Stage 0** | `src/rank.py` | Load precomputed artifacts (BM25, LightGBM, static features) | 0.96s |
| **Stage 1** | `src/retrieval.py` | Dual-pass BM25 retrieval (top 5,000 + rare-term safety net) | 0.03s |
| **Stage 2** | `src/rank.py` | Load Stage 1 candidate records via byte-offset index | 0.38s |
| **Stage 2b** | `src/features.py` | Live feature extraction (22-feature matrix) | 0.37s |
| **Stage 4** | `src/rank.py` | LightGBM LambdaRank inference + consistency multiplier | 0.01s |
| **Stage 5** | `src/reasoning.py` | Deterministic reasoning compiler (top 100) | 1.77s |
| **Stage 6** | `src/rank.py` | Monotonicity assertion, honeypot + diversity audits, CSV write | <0.01s |
| **Total** | | **End-to-end wall-clock** | **3.55s** |

The offline phases run once during development with no time or network restrictions. Only Stages 0–6 execute during the competition's 5-minute ranking window.

---

## Architecture

### Stage 1 — Dual-Pass BM25 Retrieval

Two independent BM25 queries run against a vectorised NumPy CSR matrix pre-built offline:

- **Pass A** — JD skill terms expanded via `data/skill_aliases.json`, queried against each candidate's `skills[].name` array. Skill names are structured, unique, and immune to the templated noise found in summary/description fields.
- **Pass B** — Production signal keywords (`deployed`, `serving`, `latency`, `scale`, `inference`) queried against `career_history[].description`, catching candidates with production scaling experience who don't surface on skill keywords alone.
- **Rare-term safety net** — Niche terms (`pinecone`, `lambdarank`, `qdrant`, `bm25`) explicitly retrieve sparse but highly relevant profiles that may not rank in the top 5,000 by aggregate score.

The union of all three passes forms the Stage 1 pool (~8,500 candidates).

### Stage 2 — Feature Engineering

`src/features.py` produces a 22-feature float32 vector per candidate. Every feature maps to a specific field in the candidate schema — no invented or hallucinated values.

**5 Adversarial Detection Functions** targeting patterns identified in the synthetic dataset:

| Function | Signal |
|----------|--------|
| `detect_description_title_mismatch` | Domain-category mismatch between job title and role description (e.g., "Marketing Manager" title + "Mechanical engineering design" description) |
| `detect_template_description` | Career description matching one of 12 known synthetic templates identified by manual inspection of the dataset |
| `extract_production_ml_signal` | `log(1 + prod_kw_count)`; returns `-1.0` (explicit JD disqualifier) if only academic keywords present with no production signal |
| `score_langchain_dabbler` | LLM-era skill months > 12 with zero pre-LLM IR/ML foundational skills |
| `score_cv_speech_specialist` | CV/speech skill months > 24 with zero NLP/IR skill months |

**Complete 22-Feature Matrix:**

| # | Feature | Formula / Source |
|---|---------|-----------------|
| 1 | `bm25_score` | Stage 1 BM25 retrieval score (normalised) |
| 2 | `yoe` | `profile.years_of_experience` |
| 3 | `Param_A_Systems_Depth` | Fraction of career months in roles whose descriptions contain retrieval/search/ranking keywords |
| 4 | `Param_B_Availability` | `(recruiter_response_rate + exp(-days_inactive / 90)) / 2` |
| 5 | `Param_C_Tenure` | `min(avg_tenure_months, 48) / 48` — rewards 3+ year tenures |
| 6 | `Param_D_Notice_Exp` | `exp(-max(0, days-30) / 30)` — 30d→1.0, 60d→0.37, 90d→0.14, 150d→0.006 |
| 7 | `Param_E_Credibility` | `advanced_claimed_count / max(1, assessed_count)` — higher = less credible |
| 8 | `Param_F_Consulting` | Fraction of career at IT-services consulting firms (`industry == "IT Services" AND size == "10001+"`) |
| 9 | `Param_G_Location` | Noida/Pune=1.0, other India=0.7, outside+willing to relocate=0.3, outside+not willing=0.0 |
| 10 | `Param_H_GitHub` | `github_activity_score / 100`; 0.3 imputed if field equals -1 (absent) |
| 11 | `title_ai_fraction` | Career-weighted fraction in AI/ML/data roles via static title taxonomy |
| 12 | `prod_signal_log` | Log-compressed production keyword count; -1.0 if academic-only |
| 13 | `consistency_score` | Multiplicative honeypot penalty c1 × c2 × c3 × c4 × c5 |
| 14 | `hard_req_coverage` | Fraction of JD hard requirements satisfied by candidate's skill list |
| 15 | `flag_consulting_only` | `consulting_fraction > 0.95` |
| 16 | `flag_title_chaser` | `avg_tenure < 18 months` across 3+ jobs |
| 17 | `flag_langchain_dabbler` | LLM-era months > 12 AND pre-LLM months == 0 |
| 18 | `flag_cv_specialist` | CV/speech months > 24 AND NLP/IR months == 0 |
| 19 | `flag_title_desc_mismatch` | Domain-category mismatch fraction across career history |
| 20 | `flag_template_desc` | Max SequenceMatcher ratio against template registry |
| 21 | `interaction_req_x_consistency` | `hard_req_coverage × consistency_score` |
| 22 | `interaction_yoe_x_prod` | `yoe × prod_signal_log` |

### Stage 3 — Logical Consistency (Honeypot Defenses)

```
consistency_score = c1 × c2 × c3 × c4 × c5
```

A single logical impossibility reduces the composite to near-zero, suppressing that candidate regardless of their skill profile quality:

| Check | Condition | Effect |
|-------|-----------|--------|
| c1 — Timeline impossibility | `skill.duration_months > total_experience_months` | Hard zero |
| c2 — Signup anomaly | `signup_date > last_active_date` | Hard zero |
| c3 — Salary inversion | `expected_salary.min > max` | 0.1 (heavy penalty) |
| c4 — Assessment contradiction | Claims "advanced" AND assessment score exists AND score < 50 | Compounding 0.4× per violation |
| c5 — Engagement mismatch | High BM25 score AND `connections ≤ 60 AND search_appearances ≤ 15 AND endorsements ≤ 4` | Hard zero |

### Stage 4 — LightGBM LambdaRank

**Model configuration:**
- `objective: lambdarank`
- `eval_at: [5, 10, 50]` — explicitly optimises Precision@5, the spec's primary tiebreak criterion
- Early stopping monitors NDCG@5 (patience=30)
- 200 boosting rounds

**Training labels — Gemma3 pairwise annotation (key differentiator):**

Rather than a pure heuristic label, training labels were generated via 2,500 pairwise LLM comparisons using Gemma3:4b-it-q4\_K\_M running locally on Ollama — zero external API calls, fully reproducible. A stratified sample of 500 Stage 1 candidates was drawn across three strata (top-100, boundary 101–300, and broader pool with guaranteed low-consistency coverage), then each candidate received ~5 matchups against random opponents.

For each pair, Gemma3 read both candidates' full structured profiles alongside the JD requirements and disqualifiers, then produced a single verdict: `CANDIDATE_A`, `CANDIDATE_B`, or `TIE`. Win/loss tallies were converted to Elo ratings via Laplace-smoothed win rates:

```python
win_rate = (wins + 0.5) / (total + 1)   # Laplace smoothing
elo = 400 * log10(win_rate / (1 - win_rate)) + 1500
```

Elo ratings were thresholded to 0–3 relevance labels by quartile, producing a balanced training set (~125 candidates per label).

**Why this breaks circularity:** Gemma had no knowledge of our 22 features, BM25 scores, or penalty weights. It learned independently that IR-specific skills (FAISS, BM25, Qdrant, Sentence Transformers) outrank generic ML skills, and that production-company backgrounds outrank consulting-only careers. LightGBM then learns how our 22 features correlate with these independent judgments — discovering interactions we didn't explicitly encode.

**Post-inference consistency multiplier:**
```python
final_score = lgbm_raw_score × consistency_score
```

This ensures candidates with data integrity violations (c1–c5) are suppressed to near-zero regardless of model prediction, providing a clean separation of concerns: LightGBM handles fit, consistency checks handle data integrity.

### Stage 5 — Reasoning Compiler

`src/reasoning.py` generates a 1–2 sentence reasoning string per candidate using a deterministic grammar engine with the following properties:

- **4 structural templates** rotated via `abs(hash(candidate_id)) % 4` — no two consecutive strings share the same sentence skeleton, eliminating template monotony across the top 100
- **Priority-ranked concern surfacing** — notice period > 90 days surfaces before location preference, which surfaces before skill credibility concerns; concerns are never a generic checklist
- **JD-specific skill phrases** — named skill combinations (`FAISS + Sentence Transformers + BM25`) surfaced instead of generic category labels
- **Numeric regex audit** — every number in the output string is asserted to exist in the candidate's raw JSON before writing; guarantees zero numeric hallucination
- **N-gram collision check** — `difflib.SequenceMatcher` run across all 100 outputs; strings with > 85% structural similarity are flagged before submission
- **Decision audit trail** — `reasoning_trace.jsonl` logs the exact features, tone percentile, and concern selected for each of the top 30 candidates, enabling direct answers during Stage 5 interview

---

## Validation

### Full Validation Suite

```bash
python scripts/run_full_validation.py
```

Runs four checks in sequence:
1. **Honeypot injection test** — injects all 7 synthetic violation types into a cloned top-ranked candidate and asserts zero leakage into the top-100 output
2. **Diversity audit** — asserts employer concentration ≤ 30% and archetype signature concentration ≤ 25% via `validate_pipeline.check_top100_diversity`
3. **c5 boundary test** — validates the engagement mismatch threshold fires correctly at the boundary values (connections=60, appearances=15, endorsements=4)
4. **NDCG probe** — computes NDCG@10 against hand-labeled reference points where available in the Stage 1 pool

### Blocking Audits in rank.py

Two hard-blocking assertions run before any CSV write. If either fails, `rank.py` exits non-zero with a descriptive error — no silent failures:

```python
# Honeypot audit (Section 8.1)
assert count(consistency_score < 0.25 in top_100) < 10

# Diversity audit (Section 8.2)
assert max_company_concentration <= 0.30
assert max_signature_concentration <= 0.25
```

---

## Runtime Constraints (All Enforced)

| Constraint | Limit | Enforcement |
|-----------|-------|-------------|
| Wall-clock | ≤ 300s | `assert elapsed < 300` + `sys.exit(4)` if exceeded |
| RAM | ≤ 16 GB | BM25 Stage 1 pool capped at 5,000 candidates |
| Network | Zero | `--network none` Docker flag; zero runtime imports make network calls |
| Disk | ≤ 5 GB | Total precomputed artifacts: ~216 MB |
| Output rows | Exactly 100 | `assert len(df) == 100` before CSV write |
| Score monotonicity | Non-increasing | `assert_monotonicity()` before CSV write |
| Tiebreaking | Ascending `candidate_id` | `sorted(key=lambda x: (-x[1], x[0]))` |
| Determinism | Byte-identical across runs | `REFERENCE_DATE = date(2026, 1, 1)` constant — never `datetime.now()` |

---

## File Structure

```
├── data/
│   └── skill_aliases.json              # JD taxonomy: skill aliases for BM25 query expansion
├── precomputed/                         # Artifacts generated by precompute.py
│   ├── vocab.pkl                       # BM25 vocabulary: term → column index (19.5 KB)
│   ├── bm25_matrix.npz                 # Vectorised Scipy BM25 CSR matrix (39.6 MB)
│   ├── candidate_offsets.pkl           # Byte-offset index for O(1) JSONL lookup (2.0 MB)
│   ├── lgbm_model.txt                  # Trained LightGBM booster — native text format (1.3 MB)
│   ├── lgbm_model.pkl                  # LightGBM booster — pickle fallback (1.4 MB)
│   ├── static_features.pkl             # 18 JD-independent features precomputed offline (21.7 MB)
│   ├── candidate_ids.pkl               # BM25 row → candidate_id mapping (1.5 MB)
│   └── weak_labels.pkl                 # Training labels log from offline precomputation (2.4 MB)
├── src/
│   ├── jd_parser.py                    # JD requirement extraction from skill_aliases.json
│   ├── retrieval.py                    # Dual-pass BM25 retrieval + rare-term safety net
│   ├── features.py                     # 22-feature matrix + 5 adversarial detection functions
│   ├── reasoning.py                    # Deterministic reasoning compiler
│   └── rank.py                         # Main entry point
├── scripts/
│   ├── precompute.py                   # Offline: BM25 indexing + LightGBM training
│   ├── app.py                          # Streamlit sandbox (lite mode, ≤1 GB RAM)
│   ├── validate_submission.py          # Output format validator
│   ├── validate_pipeline.py            # Competition-provided validation module (unmodified)
│   ├── run_full_pipeline.py            # End-to-end orchestration with artifact caching
│   ├── run_full_validation.py          # Full validation suite
│   └── rebuild_fast_artifacts.py       # Utility: rebuild NumPy BM25 artifacts from scratch
├── experiments/
│   └── pairwise_llm_check/             # Offline annotation experiment — isolated from inference
│       ├── annotate_and_retrain.py     # Gemma3 pairwise annotation + LightGBM retraining
│       ├── annotations.jsonl           # 2,500 pairwise judgments (Gemma3:4b-it-q4_K_M, local)
│       └── README.md                   # Experiment methodology and budget exemption statement
├── diagnostics/
│   ├── diag_profile_live_features.py   # Live feature extraction latency profiler
│   └── verify_c5_thresholds.py         # c5 boundary condition verification
├── logs/                                # Runtime logs generated by rank.py (gitignored)
├── requirements.txt                    # All dependencies pinned to exact versions
├── Dockerfile                          # CPU-only, --network none compatible
├── docker-entrypoint.sh                # Pipeline mode selector
├── submission_metadata.yaml            # Competition portal metadata
└── README.md                           # This file
```

---

## Streamlit Sandbox (§10.5 Compliance)

The sandbox runs in **lite mode** — accepts a JSONL upload of up to 10,000 candidates, builds a BM25 index inline on the uploaded data, runs the full ranking pipeline, and returns a downloadable `submission.csv`. Peak RAM stays well under 1 GB.

**Local:**
```bash
streamlit run scripts/app.py
```

**Streamlit Cloud (free tier):**
1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Set main file path to `scripts/app.py`.
4. Click **Deploy**.

Only the fast-path artifacts need to be committed (`vocab.pkl` + `bm25_matrix.npz` + `candidate_ids.pkl` + `lgbm_model.txt`, ~42 MB total). The full `candidates.jsonl` and the 146 MB `bm25_index.pkl` fallback are not required in the Streamlit deployment.

---

## Troubleshooting

**`precompute.py` raises a memory error:**
Ensure at least 16 GB RAM is available. The full 100K JSONL requires approximately 4–6 GB peak during BM25 index construction.

**`rank.py` fails the diversity audit (exit code 3):**
The top-100 candidates are too homogeneous. Check LightGBM feature importances via `precomputed/lgbm_model.txt` and verify the training label distribution in `scripts/precompute.py` is balanced across all four quartiles.

**`rank.py` exits with code 2 (honeypot audit failed):**
More than 10 candidates with `consistency_score < 0.25` reached the top-100. Verify that `consistency_score` is being computed correctly in `src/features.py` and that the post-inference multiplier (`final_score = lgbm_score × consistency_score`) is active in `src/rank.py`.

**Docker build fails on arm64 Mac:**
Use `--platform linux/amd64` if cross-building for a cloud runner. LightGBM provides native arm64 wheels for local builds.

---

## AI Tool Disclosure

This submission was developed with the assistance of **Antigravity AI coding assistant** (Claude Sonnet 4.6 model) for code scaffolding, latency diagnostics, and iterative debugging throughout the development process.

**Gemma3:4b-it-q4\_K\_M** (Google DeepMind, running locally via Ollama) was used offline to generate 2,500 pairwise relevance judgments on a stratified sample of 500 Stage 1 candidates. These judgments served as independent, non-circular training labels for the LightGBM model. No candidate data was transmitted to any external service at any point. All ranking inference is CPU-only with zero network calls.

Key milestones directed and verified by the human team at every stage:
- Identified and fixed the weak-label circularity bug where heuristic labels were rewarding keyword-stuffed trap candidates
- Designed the stratified pairwise sampling strategy with guaranteed low-consistency candidate coverage
- Diagnosed and resolved the score compression issue (normalization scope fix in output assembly)
- Approved the Elo → quartile label conversion thresholds and post-inference consistency multiplier
- Verified all Stage 4 and Stage 5 compliance criteria against actual pipeline output before submission