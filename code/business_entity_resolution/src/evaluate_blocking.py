"""
Evaluate blocking recall on training data.

Takes a sample of Source 1 training entities, runs the TF-IDF blocking
pipeline, and measures what fraction of ground-truth matches are captured
in the candidate set. This tells us the recall ceiling for Stage 2.

Usage:
    python evaluate_blocking.py [--sample-size N]

Run from the repository root.
"""

import csv
import time
import sys
import os
import argparse

# Add src directory to path
sys.path.insert(0, os.path.dirname(__file__))
from blocking import (
    load_entities_by_country,
    get_s1_ids_by_country,
    run_blocking_for_country,
)


def load_ground_truth(filepath, s1_filter=None):
    """Load ground truth into a dict: s1_id -> set of matched IDs."""
    gt = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)  # skip header
        for row in reader:
            if len(row) < 2:
                continue
            s1_id = row[0]
            if s1_filter is not None and s1_id not in s1_filter:
                continue
            matches = set(row[1].split(',')) if row[1].strip() else set()
            gt[s1_id] = matches
    return gt


def main():
    parser = argparse.ArgumentParser(description='Evaluate blocking recall')
    parser.add_argument('--sample-size', type=int, default=10000,
                        help='Number of S1 entities to evaluate (default: 10000)')
    parser.add_argument('--top-k-name', type=int, default=50,
                        help='Top-K candidates from name similarity (default: 50)')
    parser.add_argument('--top-k-addr', type=int, default=30,
                        help='Top-K candidates from address similarity (default: 30)')
    parser.add_argument('--batch-size', type=int, default=1000,
                        help='Batch size for similarity computation (default: 1000)')
    args = parser.parse_args()

    # UPDATED KAGGLE DATASET PATH HERE
    data_dir = '/kaggle/input/datasets/sparshrastogicsv/amazon-ml-2026'
    
    s1_file = os.path.join(data_dir, 'train_source1.tsv')
    s2_file = os.path.join(data_dir, 'train_source2.tsv')
    s3_file = os.path.join(data_dir, 'train_source3.tsv')
    gt_file = os.path.join(data_dir, 'train_ground_truth.tsv')

    print(f"{'='*60}")
    print(f"  BLOCKING EVALUATION")
    print(f"  Sample size:  {args.sample_size}")
    print(f"  Top-K (name): {args.top_k_name}")
    print(f"  Top-K (addr): {args.top_k_addr}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"{'='*60}\n")

    # Step 1: Get S1 sample IDs grouped by country
    print("Step 1: Loading S1 sample...")
    t_start = time.time()
    country_s1_ids = get_s1_ids_by_country(s1_file, args.sample_size)
    all_s1_ids = set()
    for ids in country_s1_ids.values():
        all_s1_ids.update(ids)
    for c, ids in country_s1_ids.items():
        print(f"  {c}: {len(ids)} entities")
    print(f"  Total: {len(all_s1_ids)} entities\n")

    # Step 2: Load ground truth for sample
    print("Step 2: Loading ground truth...")
    gt = load_ground_truth(gt_file, all_s1_ids)
    total_true_matches = sum(len(v) for v in gt.values())
    entities_with_matches = sum(1 for v in gt.values() if v)
    singletons = sum(1 for v in gt.values() if not v)
    print(f"  Entities with matches: {entities_with_matches}")
    print(f"  Singletons (no matches): {singletons}")
    print(f"  Total true matches: {total_true_matches}\n")

    # Step 3: Run blocking per country
    all_candidates = {}

    for country, s1_ids_list in country_s1_ids.items():
        print(f"\n{'='*50}")
        print(f"  Processing country: {country} ({len(s1_ids_list)} S1 entities)")
        print(f"{'='*50}")
        t_country = time.time()

        s1_id_set = set(s1_ids_list)

        # Load S1 entities for this country
        print("  Loading S1 data...")
        s1_ids, s1_names, s1_addrs = load_entities_by_country(
            s1_file, country, s1_id_set
        )

        # Load S2 candidates for this country
        print("  Loading S2 candidates...")
        t0 = time.time()
        s2_ids, s2_names, s2_addrs = load_entities_by_country(s2_file, country)
        print(f"    {len(s2_ids)} S2 entities ({time.time()-t0:.1f}s)")

        # Load S3 candidates for this country
        print("  Loading S3 candidates...")
        t0 = time.time()
        s3_ids, s3_names, s3_addrs = load_entities_by_country(s3_file, country)
        print(f"    {len(s3_ids)} S3 entities ({time.time()-t0:.1f}s)")

        # Combine S2 + S3
        cand_ids = s2_ids + s3_ids
        cand_names = s2_names + s3_names
        cand_addrs = s2_addrs + s3_addrs
        del s2_ids, s2_names, s2_addrs, s3_ids, s3_names, s3_addrs

        print(f"  Total candidates: {len(cand_ids)}")

        # Run blocking
        country_cands = run_blocking_for_country(
            s1_ids, s1_names, s1_addrs,
            cand_ids, cand_names, cand_addrs,
            top_k_name=args.top_k_name,
            top_k_addr=args.top_k_addr,
            batch_size=args.batch_size,
        )
        all_candidates.update(country_cands)

        # Per-country recall
        country_true = 0
        country_found = 0
        for sid in s1_ids_list:
            true_m = gt.get(sid, set())
            if not true_m:
                continue
            pred_c = set(country_cands.get(sid, []))
            country_true += len(true_m)
            country_found += len(true_m & pred_c)
        country_recall = country_found / country_true if country_true else 0

        elapsed = time.time() - t_country
        print(f"\n  {country} Recall: {country_recall:.4f} ({country_recall*100:.2f}%)")
        print(f"  {country} done in {elapsed:.1f}s")

    # Step 4: Overall evaluation
    print(f"\n{'='*60}")
    print(f"  OVERALL RESULTS")
    print(f"{'='*60}")

    total_true = 0
    total_found = 0
    total_cand_count = 0
    eval_entities = 0
    perfect_recall_count = 0
    missed_examples = []

    for s1_id, true_matches in gt.items():
        if not true_matches:
            continue
        eval_entities += 1
        cand_set = set(all_candidates.get(s1_id, []))
        found = true_matches & cand_set
        missed = true_matches - cand_set

        total_true += len(true_matches)
        total_found += len(found)
        total_cand_count += len(cand_set)

        if not missed:
            perfect_recall_count += 1
        elif len(missed_examples) < 5:
            missed_examples.append((s1_id, missed))

    recall = total_found / total_true if total_true > 0 else 0
    avg_cands = total_cand_count / eval_entities if eval_entities else 0
    perfect_pct = perfect_recall_count / eval_entities * 100 if eval_entities else 0

    print(f"  Entities evaluated (with matches): {eval_entities}")
    print(f"  Total true matches:    {total_true}")
    print(f"  Found in candidates:   {total_found}")
    print(f"  Missed:                {total_true - total_found}")
    print(f"")
    print(f"  *** Blocking Recall:   {recall:.4f} ({recall*100:.2f}%) ***")
    print(f"  Entities with 100% recall: {perfect_recall_count}/{eval_entities} ({perfect_pct:.1f}%)")
    print(f"  Avg candidates/entity: {avg_cands:.1f}")
    print(f"")
    print(f"  Total elapsed: {time.time()-t_start:.1f}s")

    if missed_examples:
        print(f"\n  Sample of missed matches (for debugging):")
        for s1_id, missed in missed_examples:
            print(f"    {s1_id} missed: {missed}")

    # Save results - writing to /kaggle/working/ so it saves correctly in Kaggle
    results_file = '/kaggle/working/blocking_eval_results.txt'
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write(f"Sample size: {args.sample_size}\n")
        f.write(f"Top-K name: {args.top_k_name}\n")
        f.write(f"Top-K addr: {args.top_k_addr}\n")
        f.write(f"Blocking Recall: {recall:.4f}\n")
        f.write(f"Perfect recall entities: {perfect_recall_count}/{eval_entities}\n")
        f.write(f"Avg candidates/entity: {avg_cands:.1f}\n")
    print(f"\n  Results saved to {results_file}")


if __name__ == '__main__':
    main()