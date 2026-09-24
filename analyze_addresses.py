import csv
import sys
import random

def normalize_name(name):
    if not name: return ""
    return name.lower().strip()

def normalize_address(address):
    if not address: return ""
    return address.lower().strip()

def main():
    ground_truth_file = 'student_resource/dataset/train/train_ground_truth.tsv'
    sources = [
        'student_resource/dataset/train/train_source1.tsv',
        'student_resource/dataset/train/train_source2.tsv',
        'student_resource/dataset/train/train_source3.tsv'
    ]
    
    # 1. Load ground truth pairs to check a sample
    # We will just load the first 10,000 rows to find examples
    all_entity_ids = set()
    gt_pairs = []
    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader) # skip header
        for _ in range(10000):
            row = next(reader, None)
            if not row: break
            s1_id = row[0]
            matched_ids = row[1].split(',') if len(row) > 1 else []
            all_entity_ids.add(s1_id)
            all_entity_ids.update(matched_ids)
            for m in matched_ids:
                gt_pairs.append((s1_id, m))
                
    # 2. Extract details
    details = {}
    remaining_ids = set(all_entity_ids)
    for source in sources:
        if not remaining_ids: break
        with open(source, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if not row: continue
                eid = row[0]
                if eid in remaining_ids:
                    details[eid] = {
                        'name': row[1] if len(row) > 1 else '',
                        'address': row[2] if len(row) > 2 else '',
                        'country': row[3] if len(row) > 3 else ''
                    }
                    remaining_ids.remove(eid)
                    if not remaining_ids: break
                    
    # 3. Find examples where name is the same but address is different
    same_name_diff_addr = []
    
    for s1, m2 in gt_pairs:
        d1 = details.get(s1)
        d2 = details.get(m2)
        if d1 and d2:
            n1 = normalize_name(d1['name'])
            n2 = normalize_name(d2['name'])
            a1 = normalize_address(d1['address'])
            a2 = normalize_address(d2['address'])
            
            # If names are exactly the same (or very close) but addresses are quite different
            if n1 == n2 and n1 != "":
                # Check if addresses are significantly different (e.g. no overlap in words)
                a1_words = set(a1.replace(',', '').split())
                a2_words = set(a2.replace(',', '').split())
                
                # If they have very few words in common, they are likely different addresses
                if a1 and a2 and len(a1_words.intersection(a2_words)) <= 1:
                    same_name_diff_addr.append((d1, d2))
                    
    # 4. Print results
    with open('address_analysis.txt', 'w', encoding='utf-8') as out_f:
        out_f.write(f"Found {len(same_name_diff_addr)} examples in sample where exactly the same name has very different addresses, but they ARE marked as matches.\n\n")
        for d1, d2 in same_name_diff_addr[:10]:
            out_f.write(f"Name: {d1['name']}\n")
            out_f.write(f"  Address 1: {d1['address']} ({d1['country']})\n")
            out_f.write(f"  Address 2: {d2['address']} ({d2['country']})\n")
            out_f.write("\n")

if __name__ == '__main__':
    main()
