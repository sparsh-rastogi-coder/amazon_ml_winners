"""
Stage 1: Blocking / Candidate Generation for Entity Resolution

Uses TF-IDF with character n-grams for fuzzy text matching,
partitioned by country for scalability.

Optimized with sparse_dot_topn for 10-100x speedup and memory efficiency
to run successfully on Kaggle environments.
"""

import csv
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import gc
import time
import re
from sparse_dot_topn import awesome_cossim_topn

def preprocess_name(name):
    """Strip .com/.net and add spaces to handle smashed website domains."""
    name = str(name).lower()
    name = re.sub(r'\.(com|net|org|in|co\.in|co|us|info)$', ' ', name)
    return name

def load_entities_by_country(filepath, target_country, entity_filter=None):
    """Load entities from a TSV file filtered by country."""
    ids, names, addresses = [], [], []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)  # skip header
        for row in reader:
            if len(row) < 4:
                continue
            eid, name, addr, country = row[0], row[1], row[2], row[3]
            if country != target_country:
                continue
            if entity_filter is not None and eid not in entity_filter:
                continue
            ids.append(eid)
            names.append(preprocess_name(name if name else ''))
            addresses.append(addr if addr else '')
    return ids, names, addresses

def get_countries(filepath, sample_size=None):
    """Get the set of unique countries in a source file."""
    countries = set()
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for i, row in enumerate(reader):
            if sample_size and i >= sample_size:
                break
            if len(row) >= 4:
                countries.add(row[3])
    return countries

def get_s1_ids_by_country(filepath, sample_size=None):
    """Load S1 entity IDs grouped by country."""
    country_ids = {}
    count = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            if sample_size and count >= sample_size:
                break
            if len(row) < 4:
                continue
            eid, country = row[0], row[3]
            if country not in country_ids:
                country_ids[country] = []
            country_ids[country].append(eid)
            count += 1
    return country_ids

def run_blocking_for_country(s1_ids, s1_names, s1_addrs,
                             cand_ids, cand_names, cand_addrs,
                             top_k_name=50, top_k_addr=30,
                             name_ngram_range=(3, 4),
                             addr_ngram_range=(3, 4),
                             name_max_features=80000,
                             addr_max_features=50000,
                             batch_size=50000):  # Much larger batch size safely supported now
    """Run TF-IDF blocking for a single country partition using sparse_dot_topn."""
    n_s1 = len(s1_ids)
    n_cand = len(cand_ids)
    candidates = {sid: set() for sid in s1_ids}

    # === Name-based blocking ===
    print(f"  [Name] Fitting TF-IDF on {n_cand} candidates...")
    t0 = time.time()
    name_vectorizer = TfidfVectorizer(
        analyzer='char_wb', ngram_range=name_ngram_range,
        max_features=name_max_features, dtype=np.float32,
        sublinear_tf=True
    )
    cand_name_vecs = name_vectorizer.fit_transform(cand_names)
    print(f"  [Name] TF-IDF shape: {cand_name_vecs.shape} (took {time.time()-t0:.1f}s)")

    print(f"  [Name] Finding top-{top_k_name} candidates using sparse_dot_topn...")
    t0 = time.time()
    
    for batch_start in range(0, n_s1, batch_size):
        batch_end = min(batch_start + batch_size, n_s1)
        s1_name_batch = name_vectorizer.transform(s1_names[batch_start:batch_end])
        
        sim = awesome_cossim_topn(s1_name_batch, cand_name_vecs.T, top_k_name, 0.1)
        
        for i in range(sim.shape[0]):
            row = sim.getrow(i)
            if row.nnz > 0:
                s1_id = s1_ids[batch_start + i]
                candidates[s1_id].update(row.indices.tolist())
                
        if batch_end % 5000 == 0 or batch_end == n_s1:
            print(f"    {batch_end}/{n_s1} queries processed...")

    print(f"  [Name] Done ({time.time()-t0:.1f}s)")
    del cand_name_vecs, name_vectorizer
    gc.collect()

    # === Address-based blocking ===
    print(f"  [Addr] Fitting TF-IDF on {n_cand} candidates...")
    t0 = time.time()
    addr_vectorizer = TfidfVectorizer(
        analyzer='char_wb', ngram_range=addr_ngram_range,
        max_features=addr_max_features, dtype=np.float32,
        sublinear_tf=True
    )
    cand_addr_vecs = addr_vectorizer.fit_transform(cand_addrs)
    print(f"  [Addr] TF-IDF shape: {cand_addr_vecs.shape} (took {time.time()-t0:.1f}s)")

    print(f"  [Addr] Finding top-{top_k_addr} candidates using sparse_dot_topn...")
    t0 = time.time()
    
    for batch_start in range(0, n_s1, batch_size):
        batch_end = min(batch_start + batch_size, n_s1)
        s1_addr_batch = addr_vectorizer.transform(s1_addrs[batch_start:batch_end])
        
        sim = awesome_cossim_topn(s1_addr_batch, cand_addr_vecs.T, top_k_addr, 0.1)
        
        for i in range(sim.shape[0]):
            row = sim.getrow(i)
            if row.nnz > 0:
                s1_id = s1_ids[batch_start + i]
                candidates[s1_id].update(row.indices.tolist())
                
        if batch_end % 5000 == 0 or batch_end == n_s1:
            print(f"    {batch_end}/{n_s1} queries processed...")

    print(f"  [Addr] Done ({time.time()-t0:.1f}s)")
    del cand_addr_vecs, addr_vectorizer
    gc.collect()

    # Map candidate indices to actual IDs
    print(f"  Mapping indices to entity IDs...")
    final_cands = {}
    for s1_id, idx_set in candidates.items():
        final_cands[s1_id] = [cand_ids[idx] for idx in idx_set]

    avg_cands = sum(len(v) for v in final_cands.values()) / max(len(final_cands), 1)
    print(f"  Average candidates per entity: {avg_cands:.1f}")

    return final_cands
