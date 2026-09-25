"""
Amazon ML Challenge 2026 - Entity Resolution Pipeline
=====================================================
Optimized for Kaggle (30GB RAM, 12-hour limit).

Pipeline:
  Stage 1: TF-IDF char n-gram blocking (country-partitioned, sparse_dot_topn)
  Stage 2: Feature scoring with rapidfuzz
  Stage 3: Threshold-based matching, output generation

Usage on Kaggle:
  1. Upload your dataset as a Kaggle Dataset
  2. Paste this script into a Kaggle Notebook cell
  3. Update DATA_DIR to your Kaggle input path
  4. Run the cell
"""

import os
import gc
import csv
import time
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import awesome_cossim_topn
from rapidfuzz import fuzz

# ============================================================
# CONFIGURATION — Update DATA_DIR for your Kaggle environment
# ============================================================
# For Kaggle: DATA_DIR = '/kaggle/input/YOUR_DATASET_NAME'
# For local:  DATA_DIR = 'student_resource/dataset'
DATA_DIR = 'student_resource/dataset'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR  = os.path.join(DATA_DIR, 'test')
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Blocking parameters (tuned from evaluation: 96%+ recall)
TOP_K          = 60          # top-K candidates per signal
NTOP           = 80          # sparse_dot_topn n_top per row
SIM_THRESHOLD  = 0.15        # minimum cosine similarity to keep
MAX_FEAT_NAME  = 60000       # TF-IDF vocab size for names
MAX_FEAT_ADDR  = 40000       # TF-IDF vocab size for addresses
BATCH_S1       = 50000       # S1 entities per batch for similarity

# Matching thresholds (precision-heavy for F_0.5)
NAME_HIGH_THRESH      = 82   # rapidfuzz score — confident name match
NAME_MED_THRESH       = 55   # moderate name similarity
ADDR_SUPPORT_THRESH   = 65   # address must support if name is moderate


# ============================================================
# DATA LOADING
# ============================================================
def load_source(filepath, country_filter=None):
    """Load a source TSV, optionally filtered by country. Returns parallel lists."""
    ids, names, addrs = [], [], []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)  # skip header
        for row in reader:
            if len(row) < 4:
                continue
            if country_filter and row[3] != country_filter:
                continue
            ids.append(row[0])
            names.append(row[1] if row[1] else '')
            addrs.append(row[2] if row[2] else '')
    return ids, names, addrs


def load_ground_truth(filepath, s1_filter=None):
    """Load ground truth into dict: s1_id -> set of matched IDs."""
    gt = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            if len(row) < 2:
                continue
            s1_id = row[0]
            if s1_filter is not None and s1_id not in s1_filter:
                continue
            gt[s1_id] = set(row[1].split(',')) if row[1].strip() else set()
    return gt


def get_countries(filepath):
    """Get set of unique countries in a source file."""
    countries = set()
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            if len(row) >= 4:
                countries.add(row[3])
    return countries


# ============================================================
# STAGE 1: BLOCKING (TF-IDF + sparse_dot_topn)
# ============================================================
def blocking_for_country(s1_file, s2_file, s3_file, country):
    """Generate candidate pairs for one country using TF-IDF cosine similarity.
    
    Uses sparse_dot_topn for 10-100x speedup over naive sparse matmul.
    Processes S1 entities in batches to control memory.
    """
    print(f"\n{'='*55}")
    print(f"  BLOCKING: {country}")
    print(f"{'='*55}")
    t_start = time.time()

    # Load data
    print(f"  Loading S1...")
    s1_ids, s1_names, s1_addrs = load_source(s1_file, country)
    print(f"    {len(s1_ids)} S1 entities")

    print(f"  Loading S2...")
    s2_ids, s2_names, s2_addrs = load_source(s2_file, country)
    print(f"    {len(s2_ids)} S2 entities")

    print(f"  Loading S3...")
    s3_ids, s3_names, s3_addrs = load_source(s3_file, country)
    print(f"    {len(s3_ids)} S3 entities")

    cand_ids   = s2_ids   + s3_ids
    cand_names = s2_names + s3_names
    cand_addrs = s2_addrs + s3_addrs
    del s2_ids, s2_names, s2_addrs, s3_ids, s3_names, s3_addrs
    gc.collect()

    n_s1, n_cand = len(s1_ids), len(cand_ids)
    print(f"  Total candidates: {n_cand}")

    if n_s1 == 0:
        return {}, {}

    # ---------- Name-based blocking ----------
    print(f"  [Name] Fitting TF-IDF (max_features={MAX_FEAT_NAME})...")
    t0 = time.time()
    vec_name = TfidfVectorizer(
        analyzer='char_wb', ngram_range=(3, 4),
        max_features=MAX_FEAT_NAME, dtype=np.float32, sublinear_tf=True
    )
    cand_name_mat = vec_name.fit_transform(cand_names)
    print(f"    Shape: {cand_name_mat.shape}  ({time.time()-t0:.0f}s)")

    # Process S1 in batches
    name_candidates = defaultdict(set)  # s1_id -> set of candidate indices
    for batch_start in range(0, n_s1, BATCH_S1):
        batch_end = min(batch_start + BATCH_S1, n_s1)
        batch_names = s1_names[batch_start:batch_end]

        s1_name_mat = vec_name.transform(batch_names)
        sim = awesome_cossim_topn(s1_name_mat, cand_name_mat.T,
                                  NTOP, SIM_THRESHOLD, use_threads=True)

        for i in range(sim.shape[0]):
            row = sim.getrow(i)
            if row.nnz > 0:
                s1_id = s1_ids[batch_start + i]
                name_candidates[s1_id].update(row.indices.tolist())

        if batch_end % 100000 == 0 or batch_end == n_s1:
            print(f"    [Name] {batch_end}/{n_s1} done...")
        del s1_name_mat, sim
        gc.collect()

    del cand_name_mat, vec_name
    gc.collect()
    print(f"  [Name] Complete ({time.time()-t0:.0f}s)")

    # ---------- Address-based blocking ----------
    print(f"  [Addr] Fitting TF-IDF (max_features={MAX_FEAT_ADDR})...")
    t0 = time.time()
    vec_addr = TfidfVectorizer(
        analyzer='char_wb', ngram_range=(3, 4),
        max_features=MAX_FEAT_ADDR, dtype=np.float32, sublinear_tf=True
    )
    cand_addr_mat = vec_addr.fit_transform(cand_addrs)
    print(f"    Shape: {cand_addr_mat.shape}  ({time.time()-t0:.0f}s)")

    addr_candidates = defaultdict(set)
    for batch_start in range(0, n_s1, BATCH_S1):
        batch_end = min(batch_start + BATCH_S1, n_s1)
        batch_addrs = s1_addrs[batch_start:batch_end]

        s1_addr_mat = vec_addr.transform(batch_addrs)
        sim = awesome_cossim_topn(s1_addr_mat, cand_addr_mat.T,
                                  NTOP, SIM_THRESHOLD, use_threads=True)

        for i in range(sim.shape[0]):
            row = sim.getrow(i)
            if row.nnz > 0:
                s1_id = s1_ids[batch_start + i]
                addr_candidates[s1_id].update(row.indices.tolist())

        if batch_end % 100000 == 0 or batch_end == n_s1:
            print(f"    [Addr] {batch_end}/{n_s1} done...")
        del s1_addr_mat, sim
        gc.collect()

    del cand_addr_mat, vec_addr
    gc.collect()
    print(f"  [Addr] Complete ({time.time()-t0:.0f}s)")

    # ---------- Union name + address candidates ----------
    candidates = {}
    for s1_id in s1_ids:
        idx_set = name_candidates.get(s1_id, set()) | addr_candidates.get(s1_id, set())
        candidates[s1_id] = [cand_ids[j] for j in idx_set]

    # Build a lookup dict for scoring: entity_id -> (name, address)
    cand_lookup = {}
    for j, cid in enumerate(cand_ids):
        cand_lookup[cid] = (cand_names[j], cand_addrs[j])
    s1_lookup = {}
    for j, sid in enumerate(s1_ids):
        s1_lookup[sid] = (s1_names[j], s1_addrs[j])

    avg = sum(len(v) for v in candidates.values()) / max(len(candidates), 1)
    elapsed = time.time() - t_start
    print(f"\n  {country}: avg {avg:.1f} candidates/entity, done in {elapsed:.0f}s")

    return candidates, {**s1_lookup, **cand_lookup}


# ============================================================
# STAGE 2: FEATURE SCORING
# ============================================================
def score_pair(s1_name, s1_addr, c_name, c_addr):
    """Compute similarity features for one (S1, candidate) pair."""
    s1_n = str(s1_name).lower().strip()
    c_n  = str(c_name).lower().strip()
    s1_a = str(s1_addr).lower().strip()
    c_a  = str(c_addr).lower().strip()

    name_token_sort = fuzz.token_sort_ratio(s1_n, c_n)
    name_ratio      = fuzz.ratio(s1_n, c_n)

    addr_missing = (not s1_a or not c_a or s1_a == '<null>' or c_a == '<null>')
    if addr_missing:
        addr_token_sort = 0
    else:
        addr_token_sort = fuzz.token_sort_ratio(s1_a, c_a)

    return name_token_sort, name_ratio, addr_token_sort, addr_missing


def match_candidates(candidates, entity_lookup):
    """Score all candidate pairs and apply matching thresholds.
    
    Returns:
        matched: dict  s1_id -> list of matched entity IDs
    """
    matched = {}
    total_pairs = sum(len(v) for v in candidates.values())
    print(f"\n  Scoring {total_pairs} candidate pairs...")
    t0 = time.time()
    scored = 0

    for s1_id, cand_list in candidates.items():
        s1_info = entity_lookup.get(s1_id)
        if not s1_info:
            matched[s1_id] = []
            continue

        s1_name, s1_addr = s1_info
        matches = []

        for c_id in cand_list:
            c_info = entity_lookup.get(c_id)
            if not c_info:
                continue
            c_name, c_addr = c_info

            name_ts, name_r, addr_ts, addr_miss = score_pair(
                s1_name, s1_addr, c_name, c_addr
            )

            # Decision logic (precision-heavy for F_0.5):
            # High name similarity → match
            if name_ts >= NAME_HIGH_THRESH:
                matches.append(c_id)
            # Moderate name + strong address → match
            elif name_ts >= NAME_MED_THRESH and addr_ts >= ADDR_SUPPORT_THRESH:
                matches.append(c_id)

        matched[s1_id] = matches
        scored += len(cand_list)

        if scored % 500000 == 0:
            print(f"    {scored}/{total_pairs} pairs scored...")

    elapsed = time.time() - t0
    total_matches = sum(len(v) for v in matched.values())
    print(f"  Scoring done in {elapsed:.0f}s. Total matches: {total_matches}")
    return matched


# ============================================================
# STAGE 3: OUTPUT GENERATION
# ============================================================
def write_output(all_candidates, all_matched, s1_file):
    """Write candidate_pairs.tsv and matching_results.tsv."""
    # Ensure every S1 entity has a row (even those with no candidates)
    print("\n  Writing output files...")

    # Collect all S1 IDs from the source file
    all_s1_ids = []
    with open(s1_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            if row:
                all_s1_ids.append(row[0])

    # candidate_pairs.tsv
    cand_path = os.path.join(OUTPUT_DIR, 'candidate_pairs.tsv')
    with open(cand_path, 'w', encoding='utf-8', newline='') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in all_s1_ids:
            cands = all_candidates.get(s1_id, [])
            f.write(f"{s1_id}\t{','.join(cands)}\n")
    print(f"    {cand_path} written ({len(all_s1_ids)} rows)")

    # matching_results.tsv
    match_path = os.path.join(OUTPUT_DIR, 'matching_results.tsv')
    with open(match_path, 'w', encoding='utf-8', newline='') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in all_s1_ids:
            matches = all_matched.get(s1_id, [])
            f.write(f"{s1_id}\t{','.join(matches)}\n")
    print(f"    {match_path} written ({len(all_s1_ids)} rows)")


# ============================================================
# VALIDATION (optional, for training set)
# ============================================================
def evaluate_on_train(sample_size=5000):
    """Run blocking + matching on a training sample and report F_0.5."""
    print(f"\n{'#'*60}")
    print(f"  TRAINING EVALUATION (sample_size={sample_size})")
    print(f"{'#'*60}")

    s1_file = os.path.join(TRAIN_DIR, 'train_source1.tsv')
    s2_file = os.path.join(TRAIN_DIR, 'train_source2.tsv')
    s3_file = os.path.join(TRAIN_DIR, 'train_source3.tsv')
    gt_file = os.path.join(TRAIN_DIR, 'train_ground_truth.tsv')

    # Get S1 sample
    s1_country = {}
    count = 0
    with open(s1_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            if count >= sample_size:
                break
            if len(row) >= 4:
                s1_country[row[0]] = row[3]
                count += 1

    s1_ids_set = set(s1_country.keys())
    gt = load_ground_truth(gt_file, s1_ids_set)
    countries = set(s1_country.values())

    all_candidates = {}
    entity_lookup = {}

    for country in countries:
        cands, lookup = blocking_for_country(s1_file, s2_file, s3_file, country)
        # Filter to only our sample
        for sid in list(cands.keys()):
            if sid not in s1_ids_set:
                del cands[sid]
        all_candidates.update(cands)
        entity_lookup.update(lookup)

    # Evaluate blocking recall
    total_true = total_found = 0
    for s1_id, true_matches in gt.items():
        if not true_matches:
            continue
        cand_set = set(all_candidates.get(s1_id, []))
        total_true += len(true_matches)
        total_found += len(true_matches & cand_set)

    blocking_recall = total_found / total_true if total_true else 0
    print(f"\n  Blocking Recall: {blocking_recall:.4f} ({blocking_recall*100:.2f}%)")

    # Run matching
    matched = match_candidates(all_candidates, entity_lookup)

    # Evaluate F_0.5
    f05_scores = []
    for s1_id in s1_ids_set:
        true_m = gt.get(s1_id, set())
        pred_m = set(matched.get(s1_id, []))

        if not true_m and not pred_m:
            f05_scores.append(1.0)  # correct singleton
            continue
        if not true_m and pred_m:
            f05_scores.append(0.0)  # false merge on singleton
            continue
        if true_m and not pred_m:
            f05_scores.append(0.0)  # missed everything
            continue

        tp = len(true_m & pred_m)
        precision = tp / len(pred_m) if pred_m else 0
        recall    = tp / len(true_m) if true_m else 0

        if precision + recall == 0:
            f05_scores.append(0.0)
        else:
            f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
            f05_scores.append(f05)

    macro_f05 = sum(f05_scores) / len(f05_scores)
    print(f"\n  *** Macro F_0.5: {macro_f05:.4f} ({macro_f05*100:.2f}%) ***")
    print(f"  Entities evaluated: {len(f05_scores)}")

    return macro_f05


# ============================================================
# MAIN: TEST PIPELINE
# ============================================================
def run_test_pipeline():
    """Run full pipeline on test data and generate submission files."""
    print(f"\n{'#'*60}")
    print(f"  TEST PIPELINE")
    print(f"{'#'*60}")
    t_total = time.time()

    s1_file = os.path.join(TEST_DIR, 'test_source1.tsv')
    s2_file = os.path.join(TEST_DIR, 'test_source2.tsv')
    s3_file = os.path.join(TEST_DIR, 'test_source3.tsv')

    countries = get_countries(s1_file)
    print(f"  Countries in test set: {countries}")

    all_candidates = {}
    entity_lookup  = {}

    for country in sorted(countries):
        cands, lookup = blocking_for_country(s1_file, s2_file, s3_file, country)
        all_candidates.update(cands)
        entity_lookup.update(lookup)
        gc.collect()

    # Stage 2: Score and match
    all_matched = match_candidates(all_candidates, entity_lookup)

    # Stage 3: Write output
    write_output(all_candidates, all_matched, s1_file)

    elapsed = time.time() - t_total
    print(f"\n  Total pipeline time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Done! Files are in {OUTPUT_DIR}/")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == '__main__':
    # Step 1: Evaluate on training sample (optional but recommended)
    # Uncomment the next line to run training evaluation first:
    # evaluate_on_train(sample_size=5000)

    # Step 2: Run on test data and generate submission files
    run_test_pipeline()
