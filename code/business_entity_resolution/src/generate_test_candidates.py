"""
Generate candidate pairs for the test set, sharded across workers.

Usage:
    python code/business_entity_resolution/src/generate_test_candidates.py --worker-id 0 --total-workers 100
"""

import csv
import time
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(__file__))
from blocking import (
    load_entities_by_country,
    run_blocking_for_country,
)

def get_all_s1_ids(filepath):
    """Load S1 entity IDs grouped by country, preserving order."""
    country_ids = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            if len(row) < 4:
                continue
            eid, country = row[0], row[3]
            if country not in country_ids:
                country_ids[country] = []
            country_ids[country].append(eid)
    return country_ids


def main():
    parser = argparse.ArgumentParser(description='Generate Test Candidates (Sharded)')
    parser.add_argument('--worker-id', type=int, default=0, help='Worker ID (0 to total_workers-1)')
    parser.add_argument('--total-workers', type=int, default=100, help='Total number of workers')
    parser.add_argument('--top-k-name', type=int, default=50)
    parser.add_argument('--top-k-addr', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=50000)
    args = parser.parse_args()

    assert 0 <= args.worker_id < args.total_workers

    # Kaggle test directory path
    data_dir = '/kaggle/input/datasets/sparshrastogicsv/amazon-ml-2026'
    if not os.path.exists(data_dir):
        # Fallback to local training dir for testing if test dir doesn't exist locally
        data_dir = 'dataset/test' if os.path.exists('dataset/test') else 'student_resource/dataset/train'
        
    s1_file = os.path.join(data_dir, 'test_source1.tsv')
    s2_file = os.path.join(data_dir, 'test_source2.tsv')
    s3_file = os.path.join(data_dir, 'test_source3.tsv')
    
    # If test files aren't in this folder (e.g. testing locally with train data), fallback to train
    if not os.path.exists(s1_file):
        s1_file = os.path.join(data_dir, 'train_source1.tsv')
        s2_file = os.path.join(data_dir, 'train_source2.tsv')
        s3_file = os.path.join(data_dir, 'train_source3.tsv')
    
    out_file = f'/kaggle/working/candidates_worker_{args.worker_id:03d}.tsv'
    if not os.path.exists('/kaggle/working'):
        out_file = f'candidates_worker_{args.worker_id:03d}.tsv'

    print(f"{'='*60}")
    print(f"  GENERATING TEST CANDIDATES (Worker {args.worker_id}/{args.total_workers})")
    print(f"{'='*60}\n")

    t_start = time.time()
    
    country_s1_ids = get_all_s1_ids(s1_file)
    
    # Filter to this worker's chunk for each country
    worker_s1_ids = {}
    total_assigned = 0
    
    for country, ids in country_s1_ids.items():
        # Sort to ensure deterministic sharding across all workers
        ids.sort()
        n = len(ids)
        chunk_size = (n + args.total_workers - 1) // args.total_workers
        
        start_idx = args.worker_id * chunk_size
        end_idx = min(start_idx + chunk_size, n)
        
        if start_idx < n:
            worker_s1_ids[country] = ids[start_idx:end_idx]
            total_assigned += len(worker_s1_ids[country])
            print(f"  {country}: assigned {len(worker_s1_ids[country])} entities (indices {start_idx} to {end_idx-1}) out of {n}")
        else:
            print(f"  {country}: no entities assigned to this worker.")

    print(f"\n  Total entities assigned to Worker {args.worker_id}: {total_assigned}")
    if total_assigned == 0:
        print("Nothing to do!")
        return

    # Run blocking per country for the assigned chunk
    all_candidates = {}

    for country, s1_ids_list in worker_s1_ids.items():
        if not s1_ids_list:
            continue
            
        print(f"\n{'='*50}")
        print(f"  Processing country: {country} ({len(s1_ids_list)} S1 entities)")
        print(f"{'='*50}")

        s1_id_set = set(s1_ids_list)

        # Load S1 entities for this worker's chunk
        print("  Loading assigned S1 data...")
        s1_ids, s1_names, s1_addrs = load_entities_by_country(s1_file, country, s1_id_set)

        # We must load ALL S2 and S3 candidates for the country to build the TF-IDF space
        print("  Loading S2 candidates...")
        s2_ids, s2_names, s2_addrs = load_entities_by_country(s2_file, country)

        print("  Loading S3 candidates...")
        s3_ids, s3_names, s3_addrs = load_entities_by_country(s3_file, country)

        cand_ids = s2_ids + s3_ids
        cand_names = s2_names + s3_names
        cand_addrs = s2_addrs + s3_addrs
        del s2_ids, s2_names, s2_addrs, s3_ids, s3_names, s3_addrs

        print(f"  Total candidates for {country}: {len(cand_ids)}")

        # Run blocking
        country_cands = run_blocking_for_country(
            s1_ids, s1_names, s1_addrs,
            cand_ids, cand_names, cand_addrs,
            top_k_name=args.top_k_name,
            top_k_addr=args.top_k_addr,
            batch_size=args.batch_size,
        )
        all_candidates.update(country_cands)

    # Write output to TSV
    print(f"\n  Writing results to {out_file}...")
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        # Ensure we write them in the original sorted order
        for country, ids in worker_s1_ids.items():
            for s1_id in ids:
                if s1_id in all_candidates:
                    cands = all_candidates[s1_id]
                    cand_str = ",".join(cands)
                    f.write(f"{s1_id}\t{cand_str}\n")
            
    print(f"  Done in {time.time()-t_start:.1f}s")

if __name__ == '__main__':
    main()
