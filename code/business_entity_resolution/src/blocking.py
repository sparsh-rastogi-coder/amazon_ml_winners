"""
Stage 1: Blocking / Candidate Generation for Entity Resolution
Optimized for Kaggle Dual T4 GPUs using PyTorch (fallback to CPU sparse_dot_topn).
"""

import csv
import gc
import time
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        print(f"PyTorch GPU detected: {torch.cuda.get_device_name(0)}")
except ImportError:
    GPU_AVAILABLE = False

if not GPU_AVAILABLE:
    try:
        from sparse_dot_topn import awesome_cossim_topn
    except ImportError:
        pass


def preprocess_name(name):
    """Strip .com/.net and add spaces to handle smashed website domains."""
    name = str(name).lower()
    name = re.sub(r'\.(com|net|org|in|co\.in|co|us|info)$', ' ', name)
    return name

def load_entities_by_country(filepath, target_country, entity_filter=None):
    ids, names, addresses = [], [], []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader) 
        for row in reader:
            if len(row) < 4: continue
            eid, name, addr, country = row[0], row[1], row[2], row[3]
            if country != target_country: continue
            if entity_filter is not None and eid not in entity_filter: continue
            ids.append(eid)
            names.append(preprocess_name(name if name else ''))
            addresses.append(addr if addr else '')
    return ids, names, addresses

def get_countries(filepath, sample_size=None):
    countries = set()
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for i, row in enumerate(reader):
            if sample_size and i >= sample_size: break
            if len(row) >= 4: countries.add(row[3])
    return countries

def get_s1_ids_by_country(filepath, sample_size=None):
    country_ids = {}
    count = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            if sample_size and count >= sample_size: break
            if len(row) < 4: continue
            eid, country = row[0], row[3]
            if country not in country_ids: country_ids[country] = []
            country_ids[country].append(eid)
            count += 1
    return country_ids


def gpu_top_k_similarity(query_sparse, cand_sparse, top_k, batch_size=200):
    """
    Computes top-K cosine similarity on GPU using PyTorch.
    query_sparse: scipy CSR (N_queries, Vocab)
    cand_sparse: scipy CSR (N_candidates, Vocab)
    
    We do Cand @ Query.T (Sparse @ Dense) in chunks to fit in 15GB VRAM.
    """
    n_queries = query_sparse.shape[0]
    n_cand = cand_sparse.shape[0]
    
    # 1. Convert candidate sparse matrix to PyTorch GPU sparse tensor
    cand_coo = cand_sparse.tocoo()
    indices = torch.LongTensor(np.vstack((cand_coo.row, cand_coo.col)))
    values = torch.FloatTensor(cand_coo.data)
    shape = cand_coo.shape
    cand_torch = torch.sparse_coo_tensor(indices, values, torch.Size(shape)).to('cuda')
    
    results = defaultdict(list)
    
    # 2. Process queries in dense batches (Batch size 200 keeps VRAM < 5GB)
    for start in range(0, n_queries, batch_size):
        end = min(start + batch_size, n_queries)
        
        # Query batch to dense GPU tensor
        query_dense = torch.FloatTensor(query_sparse[start:end].toarray()).to('cuda')
        
        # Sim = Cand @ Query.T  => Shape: (N_cand, Batch)
        # Sparse @ Dense -> Dense
        sim_dense = torch.sparse.mm(cand_torch, query_dense.T)
        
        # Get top K across candidates (dim=0)
        top_scores, top_indices = torch.topk(sim_dense, k=top_k, dim=0)
        
        # Extract to CPU
        top_scores = top_scores.cpu().numpy()
        top_indices = top_indices.cpu().numpy()
        
        # Map back to query index
        for b_idx in range(end - start):
            q_idx = start + b_idx
            # Get candidates with score > 0.1
            valid = top_scores[:, b_idx] > 0.1
            valid_cands = top_indices[:, b_idx][valid]
            results[q_idx].extend(valid_cands.tolist())
            
        if end % 2000 == 0 or end == n_queries:
            print(f"    [GPU] {end}/{n_queries} queries processed...")
            
        del query_dense, sim_dense, top_scores, top_indices
        torch.cuda.empty_cache()
        
    del cand_torch
    torch.cuda.empty_cache()
    
    return results


def run_blocking_for_country(s1_ids, s1_names, s1_addrs,
                             cand_ids, cand_names, cand_addrs,
                             top_k_name=50, top_k_addr=30,
                             name_ngram_range=(3, 4),
                             addr_ngram_range=(3, 4),
                             name_max_features=60000,
                             addr_max_features=40000,
                             batch_size=50000):
    
    n_s1 = len(s1_ids)
    n_cand = len(cand_ids)
    candidates = {sid: set() for sid in s1_ids}

    # === Name-based blocking ===
    print(f"  [Name] Fitting TF-IDF on {n_cand} candidates...")
    t0 = time.time()
    name_vectorizer = TfidfVectorizer(
        analyzer='char_wb', ngram_range=name_ngram_range,
        max_features=name_max_features, dtype=np.float32, sublinear_tf=True
    )
    cand_name_vecs = name_vectorizer.fit_transform(cand_names)
    s1_name_vecs = name_vectorizer.transform(s1_names)
    print(f"  [Name] TF-IDF shape: {cand_name_vecs.shape} (took {time.time()-t0:.1f}s)")

    print(f"  [Name] Finding top-{top_k_name} candidates...")
    t0 = time.time()
    
    if GPU_AVAILABLE:
        name_results = gpu_top_k_similarity(s1_name_vecs, cand_name_vecs, top_k_name, batch_size=200)
        for i, sid in enumerate(s1_ids):
            candidates[sid].update(name_results[i])
    else:
        for batch_start in range(0, n_s1, batch_size):
            batch_end = min(batch_start + batch_size, n_s1)
            s1_name_batch = s1_name_vecs[batch_start:batch_end]
            sim = awesome_cossim_topn(s1_name_batch, cand_name_vecs.T, top_k_name, 0.1)
            for i in range(sim.shape[0]):
                row = sim.getrow(i)
                if row.nnz > 0:
                    candidates[s1_ids[batch_start + i]].update(row.indices.tolist())
            if batch_end % 5000 == 0 or batch_end == n_s1:
                print(f"    [CPU] {batch_end}/{n_s1} queries processed...")

    print(f"  [Name] Done ({time.time()-t0:.1f}s)")
    del cand_name_vecs, s1_name_vecs, name_vectorizer
    gc.collect()

    # === Address-based blocking ===
    print(f"  [Addr] Fitting TF-IDF on {n_cand} candidates...")
    t0 = time.time()
    addr_vectorizer = TfidfVectorizer(
        analyzer='char_wb', ngram_range=addr_ngram_range,
        max_features=addr_max_features, dtype=np.float32, sublinear_tf=True
    )
    cand_addr_vecs = addr_vectorizer.fit_transform(cand_addrs)
    s1_addr_vecs = addr_vectorizer.transform(s1_addrs)
    print(f"  [Addr] TF-IDF shape: {cand_addr_vecs.shape} (took {time.time()-t0:.1f}s)")

    print(f"  [Addr] Finding top-{top_k_addr} candidates...")
    t0 = time.time()
    
    if GPU_AVAILABLE:
        addr_results = gpu_top_k_similarity(s1_addr_vecs, cand_addr_vecs, top_k_addr, batch_size=200)
        for i, sid in enumerate(s1_ids):
            candidates[sid].update(addr_results[i])
    else:
        for batch_start in range(0, n_s1, batch_size):
            batch_end = min(batch_start + batch_size, n_s1)
            s1_addr_batch = s1_addr_vecs[batch_start:batch_end]
            sim = awesome_cossim_topn(s1_addr_batch, cand_addr_vecs.T, top_k_addr, 0.1)
            for i in range(sim.shape[0]):
                row = sim.getrow(i)
                if row.nnz > 0:
                    candidates[s1_ids[batch_start + i]].update(row.indices.tolist())
            if batch_end % 5000 == 0 or batch_end == n_s1:
                print(f"    [CPU] {batch_end}/{n_s1} queries processed...")

    print(f"  [Addr] Done ({time.time()-t0:.1f}s)")
    del cand_addr_vecs, s1_addr_vecs, addr_vectorizer
    gc.collect()

    # Map indices to IDs
    final_cands = {}
    for s1_id, idx_set in candidates.items():
        final_cands[s1_id] = [cand_ids[idx] for idx in idx_set]

    avg_cands = sum(len(v) for v in final_cands.values()) / max(len(final_cands), 1)
    print(f"  Average candidates per entity: {avg_cands:.1f}")

    return final_cands
