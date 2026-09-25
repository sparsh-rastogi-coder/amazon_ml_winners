import csv
import sys

def get_entity_details(entity_ids):
    sources = [
        'student_resource/dataset/train/train_source1.tsv',
        'student_resource/dataset/train/train_source2.tsv',
        'student_resource/dataset/train/train_source3.tsv'
    ]
    
    details = {}
    remaining_ids = set(entity_ids)
    
    for source in sources:
        if not remaining_ids:
            break
        with open(source, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if not row:
                    continue
                eid = row[0]
                if eid in remaining_ids:
                    details[eid] = {
                        'name': row[1] if len(row) > 1 else '',
                        'address': row[2] if len(row) > 2 else '',
                        'country': row[3] if len(row) > 3 else ''
                    }
                    remaining_ids.remove(eid)
                    if not remaining_ids:
                        break
    return details

def main():
    ground_truth_file = 'student_resource/dataset/train/train_ground_truth.tsv'
    
    gt_rows = []
    all_entity_ids = set()
    
    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader) # skip header
        
        # Skip the first 5 rows
        for _ in range(5):
            next(reader, None)
            
        # Read the next 10 rows (rows 6 to 15)
        for _ in range(10):
            row = next(reader, None)
            if row:
                s1_id = row[0]
                matched_ids = row[1].split(',') if len(row) > 1 else []
                gt_rows.append({'source1': s1_id, 'matches': matched_ids})
                all_entity_ids.add(s1_id)
                all_entity_ids.update(matched_ids)
                
    # Get details for all collected IDs
    details = get_entity_details(all_entity_ids)
    
    # Print the results to a file
    with open('output_next_10.txt', 'w', encoding='utf-8') as out_f:
        for i, row in enumerate(gt_rows, 6): # Start numbering from 6
            s1_id = row['source1']
            matches = row['matches']
            
            out_f.write(f"--- Row {i} ---\n")
            
            s1_info = details.get(s1_id, {})
            out_f.write(f"Source 1: {s1_id}\n")
            out_f.write(f"  Name:    {s1_info.get('name', 'N/A')}\n")
            out_f.write(f"  Address: {s1_info.get('address', 'N/A')}\n")
            out_f.write(f"  Country: {s1_info.get('country', 'N/A')}\n")
            out_f.write("Matches:\n")
            for m_id in matches:
                m_info = details.get(m_id, {})
                out_f.write(f"  - Match: {m_id}\n")
                out_f.write(f"    Name:    {m_info.get('name', 'N/A')}\n")
                out_f.write(f"    Address: {m_info.get('address', 'N/A')}\n")
                out_f.write(f"    Country: {m_info.get('country', 'N/A')}\n")
            out_f.write("\n\n")

if __name__ == '__main__':
    main()
