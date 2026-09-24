import csv
from collections import defaultdict
import statistics
import os

def jaccard(s1, s2):
    set1 = set(s1.lower().split())
    set2 = set(s2.lower().split())
    if not set1 and not set2: return 1.0
    if not set1 or not set2: return 0.0
    return len(set1.intersection(set2)) / len(set1.union(set2))

def main():
    gt_file = 'student_resource/dataset/train/train_ground_truth.tsv'
    s1_file = 'student_resource/dataset/train/train_source1.tsv'
    s2_file = 'student_resource/dataset/train/train_source2.tsv'
    s3_file = 'student_resource/dataset/train/train_source3.tsv'

    target_sample_size = 200000
    sampled_s1_ids = set()
    gt_map = {}
    
    print("Reading Ground Truth...")
    with open(gt_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for i, row in enumerate(reader):
            if i >= target_sample_size: break
            if not row: continue
            s1_id = row[0]
            matches = row[1].split(',') if len(row) > 1 and row[1].strip() else []
            gt_map[s1_id] = matches
            sampled_s1_ids.add(s1_id)

    target_m_ids = set()
    for matches in gt_map.values():
        target_m_ids.update(matches)

    print("Reading Source 1...")
    s1_data = {}
    with open(s1_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            if not row: continue
            s1_id = row[0]
            if s1_id in sampled_s1_ids:
                s1_data[s1_id] = {
                    'name': row[1] if len(row) > 1 else '',
                    'address': row[2] if len(row) > 2 else '',
                    'country': row[3] if len(row) > 3 else ''
                }
                sampled_s1_ids.remove(s1_id)
                if not sampled_s1_ids: break

    print("Reading Sources 2 & 3...")
    target_data = {}
    for src in [s2_file, s3_file]:
        remaining = set(target_m_ids)
        with open(src, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)
            for row in reader:
                if not row: continue
                m_id = row[0]
                if m_id in target_m_ids:
                    target_data[m_id] = {
                        'name': row[1] if len(row) > 1 else '',
                        'address': row[2] if len(row) > 2 else '',
                        'source': 'S2' if 'source2' in src else 'S3'
                    }
                    remaining.remove(m_id)
                    if not remaining: break

    print("Computing Insights...")
    matches_per_country = {'US': [], 'India': []}
    s2_match_count = 0
    s3_match_count = 0
    
    name_sims = []
    addr_sims = []
    perfect_name_matches = 0
    perfect_addr_matches = 0
    
    for s1_id, matches in gt_map.items():
        s1_info = s1_data.get(s1_id)
        if not s1_info: continue
        
        c = s1_info['country']
        if c in matches_per_country:
            matches_per_country[c].append(len(matches))
        
        for m_id in matches:
            m_info = target_data.get(m_id)
            if not m_info: continue
            
            if m_info['source'] == 'S2': s2_match_count += 1
            else: s3_match_count += 1
            
            # Name Similarity
            n_sim = jaccard(s1_info['name'], m_info['name'])
            name_sims.append(n_sim)
            if n_sim == 1.0: perfect_name_matches += 1
                
            # Address Similarity
            if s1_info['address'] and m_info['address'] and m_info['address'] != '<null>':
                a_sim = jaccard(s1_info['address'], m_info['address'])
                addr_sims.append(a_sim)
                if a_sim == 1.0: perfect_addr_matches += 1

    with open('extended_eda_results.txt', 'w', encoding='utf-8') as out:
        out.write("=== EXTENDED EDA INSIGHTS (Based on 200k sample) ===\n\n")
        
        out.write("1. Match Distribution by Country\n")
        for c, counts in matches_per_country.items():
            if counts:
                out.write(f"  {c}: Average {statistics.mean(counts):.2f} matches per entity (Max: {max(counts)})\n")
            
        out.write("\n2. Target Source Distribution\n")
        total_m = s2_match_count + s3_match_count
        if total_m > 0:
            out.write(f"  Matches originating from Source 2: {s2_match_count} ({s2_match_count/total_m*100:.1f}%)\n")
            out.write(f"  Matches originating from Source 3: {s3_match_count} ({s3_match_count/total_m*100:.1f}%)\n")
            
        out.write("\n3. Textual Similarity of Matches (Jaccard Word Overlap)\n")
        if name_sims:
            out.write(f"  Average Name Similarity: {statistics.mean(name_sims):.2f}\n")
            out.write(f"  Median Name Similarity:  {statistics.median(name_sims):.2f}\n")
            out.write(f"  Exact Word-for-Word Name Matches: {perfect_name_matches} ({perfect_name_matches/len(name_sims)*100:.1f}%)\n")
        
        if addr_sims:
            out.write(f"  Average Address Similarity: {statistics.mean(addr_sims):.2f}\n")
            out.write(f"  Median Address Similarity:  {statistics.median(addr_sims):.2f}\n")
            out.write(f"  Exact Word-for-Word Address Matches: {perfect_addr_matches} ({perfect_addr_matches/len(addr_sims)*100:.1f}%)\n")

    print("Done")

if __name__ == '__main__':
    main()
