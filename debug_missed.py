import csv

def get_entity_details(entity_ids, data_dir='student_resource/dataset/train'):
    sources = [
        f'{data_dir}/train_source1.tsv',
        f'{data_dir}/train_source2.tsv',
        f'{data_dir}/train_source3.tsv'
    ]
    details = {}
    remaining = set(entity_ids)
    for src in sources:
        if not remaining: break
        with open(src, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)
            for row in reader:
                if not row: continue
                eid = row[0]
                if eid in remaining:
                    details[eid] = {
                        'name': row[1] if len(row) > 1 else '',
                        'address': row[2] if len(row) > 2 else '',
                        'country': row[3] if len(row) > 3 else ''
                    }
                    remaining.remove(eid)
                    if not remaining: break
    return details

missed_pairs = [
    ('S1-633779655', ['S2-24729487']),
    ('S1-883468618', ['S3-149911356']),
    ('S1-335752330', ['S2-897778962']),
    ('S1-630223771', ['S3-587424087', 'S3-697578371', 'S3-815538600']),
    ('S1-89093509',  ['S3-3598545']),
]

all_ids = set()
for s1, missed in missed_pairs:
    all_ids.add(s1)
    all_ids.update(missed)

details = get_entity_details(all_ids)

with open('debug_missed.txt', 'w', encoding='utf-8') as f:
    for s1_id, missed_ids in missed_pairs:
        s1 = details.get(s1_id, {})
        f.write(f"{'='*70}\n")
        f.write(f"SOURCE:  {s1_id}\n")
        f.write(f"  Name:    {s1.get('name','?')}\n")
        f.write(f"  Address: {s1.get('address','?')}\n")
        f.write(f"  Country: {s1.get('country','?')}\n")
        for m_id in missed_ids:
            m = details.get(m_id, {})
            f.write(f"\nMISSED:  {m_id}\n")
            f.write(f"  Name:    {m.get('name','?')}\n")
            f.write(f"  Address: {m.get('address','?')}\n")
            f.write(f"  Country: {m.get('country','?')}\n")
        f.write(f"\n")

print("Done! See debug_missed.txt")
