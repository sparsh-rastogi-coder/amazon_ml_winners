"""
Stage 1: Blocking / Candidate Generation for Entity Resolution

Optimized with:
- unidecode & wordninja for cross-script transliteration and domain parsing.
- TF-IDF sparse_dot_topn for 100x speed.
- Numeric indexing to catch missing-address/heavy-alias edge cases.
"""

import csv
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import gc
import time
import re
import collections
import wordninja
from sparse_dot_topn import awesome_cossim_topn

NOISE_WORDS_REGEX = re.compile(
    r'\b(llc|inc|ltd|pvt|private|limited|corp|corporation|services|center|partners|enterprises|co|m/?s|mr|mrs|smt)\b', 
    re.IGNORECASE
)

def preprocess_text(text, is_name=True):
    """Normalize text: split domain names."""
    if not text: return ""
    text = str(text).lower()
    
    if is_name:
        tokens = []
        for word in text.split():
            if re.search(r'\.(com|in|net|org|co\.in|us|info)$', word):
                word = re.sub(r'\.(com|in|net|org|co\.in|us|info)$', '', word)
                tokens.extend(wordninja.split(word))
            else:
                tokens.append(word)
        text = " ".join(tokens)
        
    text = re.sub(r'\s+', ' ', text).strip()
    return text

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
            names.append(preprocess_text(name, is_name=True))
            addresses.append(preprocess_text(addr, is_name=False))
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


def extract_numbers(text):
    if not text: return set()
    nums = set()
    for n_str in re.findall(r'\d+', str(text)):
        n = int(n_str)
        if n > 9:  # Ignore small numbers 0-9 to avoid candidate explosion
            nums.add(str(n))
    return nums


def run_blocking_for_country(s1_ids, s1_names, s1_addrs,
                             cand_ids, cand_names, cand_addrs,
                             top_k_name=50, top_k_addr=30,
                             name_ngram_range=(3, 4),
                             addr_ngram_range=(3, 4),
                             name_max_features=80000,
                             addr_max_features=50000,
                             batch_size=50000):
    
    n_s1 = len(s1_ids)
    n_cand = len(cand_ids)
    candidates = {sid: set() for sid in s1_ids}

    # === Channel 1: Name-based TF-IDF blocking ===
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
    for batch_start in range(0, n_s1, batch_size):
        batch_end = min(batch_start + batch_size, n_s1)
        s1_name_batch = s1_name_vecs[batch_start:batch_end]
        sim = awesome_cossim_topn(s1_name_batch, cand_name_vecs.T, top_k_name, 0.05, use_threads=True, n_jobs=4)
        for i in range(sim.shape[0]):
            row = sim.getrow(i)
            if row.nnz > 0:
                candidates[s1_ids[batch_start + i]].update(row.indices.tolist())
    print(f"  [Name] Done ({time.time()-t0:.1f}s)")
    del cand_name_vecs, s1_name_vecs, name_vectorizer
    gc.collect()

    # === Channel 2: Address-based TF-IDF blocking ===
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
    for batch_start in range(0, n_s1, batch_size):
        batch_end = min(batch_start + batch_size, n_s1)
        s1_addr_batch = s1_addr_vecs[batch_start:batch_end]
        sim = awesome_cossim_topn(s1_addr_batch, cand_addr_vecs.T, top_k_addr, 0.05, use_threads=True, n_jobs=4)
        for i in range(sim.shape[0]):
            row = sim.getrow(i)
            if row.nnz > 0:
                candidates[s1_ids[batch_start + i]].update(row.indices.tolist())
    print(f"  [Addr] Done ({time.time()-t0:.1f}s)")
    del cand_addr_vecs, s1_addr_vecs, addr_vectorizer
    gc.collect()

    # === Channel 3: Combined Name+Address blocking ===
    print(f"  [Comb] Fitting TF-IDF on {n_cand} candidates...")
    t0 = time.time()
    comb_vectorizer = TfidfVectorizer(
        analyzer='char_wb', ngram_range=(3, 4),
        max_features=80000, dtype=np.float32, sublinear_tf=True
    )
    cand_comb = [f"{n} {a}" for n, a in zip(cand_names, cand_addrs)]
    cand_comb_vecs = comb_vectorizer.fit_transform(cand_comb)
    del cand_comb
    
    s1_comb = [f"{n} {a}" for n, a in zip(s1_names, s1_addrs)]
    s1_comb_vecs = comb_vectorizer.transform(s1_comb)
    del s1_comb
    print(f"  [Comb] TF-IDF shape: {cand_comb_vecs.shape} (took {time.time()-t0:.1f}s)")

    print(f"  [Comb] Finding top-30 candidates...")
    t0 = time.time()
    for batch_start in range(0, n_s1, batch_size):
        batch_end = min(batch_start + batch_size, n_s1)
        s1_comb_batch = s1_comb_vecs[batch_start:batch_end]
        sim = awesome_cossim_topn(s1_comb_batch, cand_comb_vecs.T, 30, 0.05, use_threads=True, n_jobs=4)
        for i in range(sim.shape[0]):
            row = sim.getrow(i)
            if row.nnz > 0:
                candidates[s1_ids[batch_start + i]].update(row.indices.tolist())
    print(f"  [Comb] Done ({time.time()-t0:.1f}s)")
    del cand_comb_vecs, s1_comb_vecs, comb_vectorizer
    gc.collect()

    # === Channel 4: Numeric Blocking Index (for heavy alias/missing text cases) ===
    print(f"  [Num] Building numeric index for candidates...")
    t0 = time.time()
    cand_num_idx = collections.defaultdict(list)
    for c_idx, (cname, caddr) in enumerate(zip(cand_names, cand_addrs)):
        nums = extract_numbers(cname) | extract_numbers(caddr)
        for num in nums:
            cand_num_idx[num].append(c_idx)
            
    # Filter highly frequent numbers (e.g. zip codes, generic street names like '400')
    # If a number is shared by >100 entities in one country, it's useless for blocking
    cand_num_idx = {num: idxs for num, idxs in cand_num_idx.items() if len(idxs) <= 100}
    
    num_matches_added = 0
    for i, sid in enumerate(s1_ids):
        s1_nums = extract_numbers(s1_names[i]) | extract_numbers(s1_addrs[i])
        for num in s1_nums:
            if num in cand_num_idx:
                candidates[sid].update(cand_num_idx[num])
                num_matches_added += len(cand_num_idx[num])
    
    print(f"  [Num] Done ({time.time()-t0:.1f}s). Added approx {num_matches_added/max(1,n_s1):.1f} candidates per entity via pure numeric match.")

    # Map candidate indices to actual IDs
    print(f"  Mapping indices to entity IDs...")
    final_cands = {}
    for s1_id, idx_set in candidates.items():
        final_cands[s1_id] = [cand_ids[idx] for idx in idx_set]

    avg_cands = sum(len(v) for v in final_cands.values()) / max(len(final_cands), 1)
    print(f"  Average candidates per entity: {avg_cands:.1f}")

    return final_cands
