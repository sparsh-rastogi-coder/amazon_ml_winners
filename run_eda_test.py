import csv
from collections import Counter
import os

def analyze_source(filepath):
    stats = {
        'total_rows': 0,
        'missing_name': 0,
        'missing_address': 0,
        'missing_country': 0,
        'countries': Counter()
    }
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader, None)
        for row in reader:
            if not row: continue
            stats['total_rows'] += 1
            
            name = row[1].strip() if len(row) > 1 else ""
            address = row[2].strip() if len(row) > 2 else ""
            country = row[3].strip() if len(row) > 3 else ""
            
            if not name or name.lower() == '<null>': stats['missing_name'] += 1
            if not address or address.lower() == '<null>': stats['missing_address'] += 1
            if not country or country.lower() == '<null>': stats['missing_country'] += 1
            
            if country and country.lower() != '<null>':
                stats['countries'][country] += 1
    return stats

def main():
    sources = ['test_source1.tsv', 'test_source2.tsv', 'test_source3.tsv']
    base_dir = 'student_resource/dataset/test/'
    
    with open('eda_results_test.txt', 'w', encoding='utf-8') as out:
        for src in sources:
            print(f"Processing {src}...")
            out.write(f"=== {src.upper()} STATS ===\n")
            filepath = os.path.join(base_dir, src)
            if not os.path.exists(filepath):
                out.write("File missing.\n\n")
                continue
                
            s_stats = analyze_source(filepath)
            
            if s_stats['total_rows'] == 0:
                out.write("File empty.\n\n")
                continue
                
            out.write(f"Total Rows: {s_stats['total_rows']}\n")
            out.write(f"Missing Names: {s_stats['missing_name']} ({s_stats['missing_name']/s_stats['total_rows']*100:.2f}%)\n")
            out.write(f"Missing Addresses: {s_stats['missing_address']} ({s_stats['missing_address']/s_stats['total_rows']*100:.2f}%)\n")
            out.write(f"Missing Countries: {s_stats['missing_country']} ({s_stats['missing_country']/s_stats['total_rows']*100:.2f}%)\n")
            
            out.write("Top 5 Countries:\n")
            for c, count in s_stats['countries'].most_common(5):
                out.write(f"  {c}: {count} ({count/s_stats['total_rows']*100:.2f}%)\n")
            out.write(f"Total Unique Countries: {len(s_stats['countries'])}\n")
            out.write("\n")
            
    print("EDA complete. Results saved to eda_results_test.txt")

if __name__ == '__main__':
    main()
