"""
Stage 1: Blocking / Candidate Generation for Entity Resolution

Uses TF-IDF with character n-grams for fuzzy text matching,
partitioned by country for scalability.

Design decisions:
- char_wb n-grams (3,4): handles typos, abbreviations, transliterations
  across English, Hindi, and French without language-specific rules.
- Country partitioning: reduces search space by 2-3x.
- Separate name and address retrieval, then union: ensures high recall
  even when one field is missing or very noisy.
- sublinear_tf: dampens the effect of very frequent terms.
- Sparse matrix cosine similarity: memory-efficient for large candidate sets.
"""

import csv
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import gc
import time


def load_entities_by_country(filepath, target_country, entity_filter=None):
    """Load entities from a TSV file filtered by country.

    Streams through the file and only keeps rows matching the target country.
    This is memory-efficient for large files (5M+ rows).

    Args:
        filepath: Path to TSV file (source1/2/3)
        target_country: Country string to filter by (e.g. 'US', 'India')
        entity_filter: Optional set of entity IDs to include. If None, includes all.

    Returns:
        Tuple of (ids, names, addresses) as parallel lists
    """
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
            names.append(name if name else '')
            addresses.append(addr if addr else '')
    return ids, names, addresses


def get_countries(filepath, sample_size=None):
    """Get the set of unique countries in a source file, optionally limited to first N rows."""
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
    """Load S1 entity IDs grouped by country.

    Returns:
        Dict mapping country -> list of entity IDs
    """
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


def tfidf_top_k(query_vecs, candidate_vecs, top_k, batch_size=1000):
    """Find top-K most similar candidates for each query using sparse cosine similarity.

    Uses sparse matrix multiplication (query @ candidates.T) which naturally
    produces a sparse result — only candidate pairs sharing at least one n-gram
    get a non-zero score. This is far more memory-efficient than dense similarity.

    Args:
        query_vecs: Sparse TF-IDF matrix for queries (n_queries, vocab_size)
        candidate_vecs: Sparse TF-IDF matrix for candidates (n_candidates, vocab_size)
        top_k: Number of top candidates to return per query
        batch_size: Number of queries to process per batch

    Returns:
        List of lists of (candidate_index, score) tuples, one list per query
    """
    cand_T = candidate_vecs.T.tocsc()
    n_queries = query_vecs.shape[0]
    results = []

    for start in range(0, n_queries, batch_size):
        end = min(start + batch_size, n_queries)
        batch = query_vecs[start:end]

        # Sparse @ sparse.T → sparse (only non-zero overlaps)
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
                # Take all
                pairs = list(zip(indices.tolist(), data.tolist()))
            else:
                # Partial sort: get top-K by score
                top_idx = np.argpartition(data, -k)[-k:]
                pairs = [(indices[j], data[j]) for j in top_idx]

            results.append(pairs)

        if (end % 5000 == 0) or end == n_queries:
            print(f"    {end}/{n_queries} queries processed...")

    return results


def run_blocking_for_country(s1_ids, s1_names, s1_addrs,
                             cand_ids, cand_names, cand_addrs,
                             top_k_name=50, top_k_addr=30,
                             name_ngram_range=(3, 4),
                             addr_ngram_range=(3, 4),
                             name_max_features=80000,
                             addr_max_features=50000,
                             batch_size=1000):
    """Run TF-IDF blocking for a single country partition.

    Builds separate TF-IDF models for names and addresses, retrieves
    top-K candidates from each, and returns their union.

    Args:
        s1_ids: List of Source 1 entity IDs
        s1_names: List of Source 1 business names
        s1_addrs: List of Source 1 business addresses
        cand_ids: List of candidate (S2+S3) entity IDs
        cand_names: List of candidate business names
        cand_addrs: List of candidate business addresses
        top_k_name: Number of top candidates to retrieve by name similarity
        top_k_addr: Number of top candidates to retrieve by address similarity
        name_ngram_range: Character n-gram range for name vectorizer
        addr_ngram_range: Character n-gram range for address vectorizer
        name_max_features: Max vocabulary size for name vectorizer
        addr_max_features: Max vocabulary size for address vectorizer
        batch_size: Batch size for similarity computation

    Returns:
        Dict mapping S1 entity IDs to lists of candidate entity IDs
    """
    n_s1 = len(s1_ids)
    n_cand = len(cand_ids)

    # === Name-based blocking ===
    print(f"  [Name] Fitting TF-IDF on {n_cand} candidates...")
    t0 = time.time()
    name_vectorizer = TfidfVectorizer(
        analyzer='char_wb', ngram_range=name_ngram_range,
        max_features=name_max_features, dtype=np.float32,
        sublinear_tf=True
    )
    cand_name_vecs = name_vectorizer.fit_transform(cand_names)
    s1_name_vecs = name_vectorizer.transform(s1_names)
    print(f"  [Name] TF-IDF shape: {cand_name_vecs.shape} (took {time.time()-t0:.1f}s)")

    print(f"  [Name] Finding top-{top_k_name} candidates...")
    t0 = time.time()
    name_results = tfidf_top_k(s1_name_vecs, cand_name_vecs, top_k_name, batch_size)
    print(f"  [Name] Done ({time.time()-t0:.1f}s)")

    # Free memory
    del cand_name_vecs, s1_name_vecs, name_vectorizer
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
    s1_addr_vecs = addr_vectorizer.transform(s1_addrs)
    print(f"  [Addr] TF-IDF shape: {cand_addr_vecs.shape} (took {time.time()-t0:.1f}s)")

    print(f"  [Addr] Finding top-{top_k_addr} candidates...")
    t0 = time.time()
    addr_results = tfidf_top_k(s1_addr_vecs, cand_addr_vecs, top_k_addr, batch_size)
    print(f"  [Addr] Done ({time.time()-t0:.1f}s)")

    del cand_addr_vecs, s1_addr_vecs, addr_vectorizer
    gc.collect()

    # === Union candidates ===
    print(f"  Merging name + address candidates...")
    candidates = {}
    for i, s1_id in enumerate(s1_ids):
        cand_indices = set()

        # From name results
        if i < len(name_results):
            for idx, score in name_results[i]:
                cand_indices.add(idx)

        # From address results
        if i < len(addr_results):
            for idx, score in addr_results[i]:
                cand_indices.add(idx)

        # Map indices → candidate IDs
        candidates[s1_id] = [cand_ids[idx] for idx in cand_indices]

    avg_cands = sum(len(v) for v in candidates.values()) / max(len(candidates), 1)
    print(f"  Average candidates per entity: {avg_cands:.1f}")

    return candidates
