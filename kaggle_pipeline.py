import os
import gc
import csv
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from rapidfuzz import fuzz
import xgboost as xgb

# ==========================================
# CONFIGURATION
# ==========================================
# Adjust paths if running on Kaggle. For example, Kaggle datasets are usually in /kaggle/input/...
DATA_DIR = 'dataset' 
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# We limit the number of candidates per entity to keep memory manageable
TOP_K_NAME = 30
TOP_K_ADDR = 20

# Features for TF-IDF
MAX_FEATURES_NAME = 50000
MAX_FEATURES_ADDR = 30000

# ==========================================
# UTILS & BLOCKING
# ==========================================
def load_source_data(filepath, country_filter=None):
    ids, names, addrs, countries = [], [], [], []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            if len(row) < 4: continue
            eid, name, addr, country = row[0], row[1], row[2], row[3]
            if country_filter and country != country_filter: continue
            ids.append(eid)
            names.append(name if name else '')
            addrs.append(addr if addr else '')
            countries.append(country)
    return ids, names, addrs, countries

def tfidf_top_k(query_vecs, candidate_vecs, top_k, batch_size=50):
    """Memory-efficient top-k sparse dot product"""
    cand_T = candidate_vecs.T.tocsc()
    n_queries = query_vecs.shape[0]
    results = []
    
    for start in range(0, n_queries, batch_size):
        end = min(start + batch_size, n_queries)
        batch = query_vecs[start:end]
        
        sim = batch @ cand_T
        
        for i in range(sim.shape[0]):
            row = sim.getrow(i)
            if row.nnz == 0:
                results.append([])
                continue
                
            data = row.data
            indices = row.indices
            
            k = min(top_k, len(data))
            if k == len(data):
                pairs = list(zip(indices, data))
            else:
                top_idx = np.argpartition(data, -k)[-k:]
                pairs = [(indices[j], data[j]) for j in top_idx]
                
            results.append(pairs)
            
        if end % 10000 == 0:
            print(f"    Processed {end}/{n_queries} queries...")
            
    return results

def get_candidates_for_country(s1_file, s2_file, s3_file, target_country):
    print(f"\n--- Blocking for {target_country} ---")
    s1_ids, s1_names, s1_addrs, _ = load_source_data(s1_file, target_country)
    s2_ids, s2_names, s2_addrs, _ = load_source_data(s2_file, target_country)
    s3_ids, s3_names, s3_addrs, _ = load_source_data(s3_file, target_country)
    
    cand_ids = s2_ids + s3_ids
    cand_names = s2_names + s3_names
    cand_addrs = s2_addrs + s3_addrs
    
    del s2_ids, s2_names, s2_addrs, s3_ids, s3_names, s3_addrs
    gc.collect()
    
    print(f"Loaded {len(s1_ids)} S1, {len(cand_ids)} candidates")
    
    if len(s1_ids) == 0:
        return {}
        
    print("TF-IDF Names...")
    vec_name = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), max_features=MAX_FEATURES_NAME, dtype=np.float32)
    cand_name_vecs = vec_name.fit_transform(cand_names)
    s1_name_vecs = vec_name.transform(s1_names)
    name_results = tfidf_top_k(s1_name_vecs, cand_name_vecs, TOP_K_NAME)
    del cand_name_vecs, s1_name_vecs, vec_name; gc.collect()
    
    print("TF-IDF Addresses...")
    vec_addr = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), max_features=MAX_FEATURES_ADDR, dtype=np.float32)
    cand_addr_vecs = vec_addr.fit_transform(cand_addrs)
    s1_addr_vecs = vec_addr.transform(s1_addrs)
    addr_results = tfidf_top_k(s1_addr_vecs, cand_addr_vecs, TOP_K_ADDR)
    del cand_addr_vecs, s1_addr_vecs, vec_addr; gc.collect()
    
    candidates = {}
    for i, s1_id in enumerate(s1_ids):
        cand_idx = set()
        for idx, _ in name_results[i]: cand_idx.add(idx)
        for idx, _ in addr_results[i]: cand_idx.add(idx)
        candidates[s1_id] = [cand_ids[idx] for idx in cand_idx]
        
    return candidates

# ==========================================
# FEATURE ENGINEERING
# ==========================================
def extract_features(s1_name, s1_addr, c_name, c_addr):
    s1_n, s1_a = str(s1_name).lower(), str(s1_addr).lower()
    c_n, c_a = str(c_name).lower(), str(c_addr).lower()
    
    features = {
        'name_ratio': fuzz.ratio(s1_n, c_n),
        'name_token_sort': fuzz.token_sort_ratio(s1_n, c_n),
        'addr_ratio': fuzz.ratio(s1_a, c_a),
        'addr_token_sort': fuzz.token_sort_ratio(s1_a, c_a),
        'addr_missing': 1 if not s1_a or not c_a or '<null>' in s1_a or '<null>' in c_a else 0,
        'name_len_diff': abs(len(s1_n) - len(c_n)),
    }
    return features

# ==========================================
# PIPELINE EXECUTION
# ==========================================
def run_pipeline():
    # 1. GENERATE TEST CANDIDATES
    print("Generating candidates for Test Set...")
    countries = ['US', 'India', 'France']
    test_candidates = {}
    
    for c in countries:
        cands = get_candidates_for_country(
            os.path.join(TEST_DIR, 'test_source1.tsv'),
            os.path.join(TEST_DIR, 'test_source2.tsv'),
            os.path.join(TEST_DIR, 'test_source3.tsv'),
            c
        )
        test_candidates.update(cands)
        
    # Write candidate pairs
    cand_file = os.path.join(OUTPUT_DIR, 'candidate_pairs.tsv')
    with open(cand_file, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1, cands in test_candidates.items():
            f.write(f"{s1}\t{','.join(cands)}\n")
            
    # For a real Kaggle submission, you would extract features for these candidate pairs,
    # train an XGBoost model on the train set, and predict probabilities.
    # Since we need to generate matching_results.tsv, here is a heuristic proxy 
    # for the model that uses the features directly.
    
    print("\nScoring candidates...")
    
    # Load all test data into dicts for fast lookup
    print("Loading test data into memory for scoring...")
    test_data = {}
    for fname in ['test_source1.tsv', 'test_source2.tsv', 'test_source3.tsv']:
        ids, names, addrs, _ = load_source_data(os.path.join(TEST_DIR, fname))
        for i, n, a in zip(ids, names, addrs):
            test_data[i] = (n, a)
            
    # Generate final matches
    match_file = os.path.join(OUTPUT_DIR, 'matching_results.tsv')
    
    with open(match_file, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id, cand_list in test_candidates.items():
            matches = []
            s1_n, s1_a = test_data.get(s1_id, ("", ""))
            
            for c_id in cand_list:
                c_n, c_a = test_data.get(c_id, ("", ""))
                feats = extract_features(s1_n, s1_a, c_n, c_a)
                
                # Heuristic logic (Proxy for XGBoost decision tree)
                # If names are very similar, or moderately similar but addresses are very similar
                if feats['name_token_sort'] > 85:
                    matches.append(c_id)
                elif feats['name_token_sort'] > 60 and feats['addr_token_sort'] > 75:
                    matches.append(c_id)
                    
            f.write(f"{s1_id}\t{','.join(matches)}\n")
            
    print(f"\nPipeline complete! Output files saved to {OUTPUT_DIR}/")

if __name__ == '__main__':
    run_pipeline()
