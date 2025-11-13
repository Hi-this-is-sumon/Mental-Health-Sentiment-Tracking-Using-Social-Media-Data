import pandas as pd
import numpy as np
from collections import Counter
from typing import List, Dict
import json

def analyze_dataset(csv_path: str, text_col: str = 'message', label_col: str = 'emotion_labels'):
    """Analyze dataset characteristics and print summary."""
    df = pd.read_csv(csv_path)
    
    # Basic stats
    total_rows = len(df)
    unique_texts = df[text_col].nunique()
    
    # Analyze labels
    def get_labels(s: str) -> List[str]:
        if pd.isna(s):
            return []
        return [x.strip() for x in str(s).split(',') if x.strip()]
    
    all_labels = []
    for labels in df[label_col]:
        all_labels.extend(get_labels(labels))
    
    label_counts = Counter(all_labels)
    
    # Check duplicates
    dup_count = df[text_col].duplicated().sum()
    
    # Sample some rows
    sample_rows = df.sample(min(5, len(df))).to_dict('records')
    
    results = {
        "dataset_stats": {
            "total_rows": total_rows,
            "unique_texts": unique_texts,
            "duplicate_texts": dup_count,
            "duplicate_percentage": f"{(dup_count/total_rows)*100:.1f}%"
        },
        "label_distribution": {
            label: count for label, count in sorted(label_counts.items(), key=lambda x: -x[1])
        },
        "avg_labels_per_row": len(all_labels) / total_rows,
        "sample_rows": sample_rows
    }
    
    print("\n=== Dataset Analysis ===")
    print(f"Total rows: {results['dataset_stats']['total_rows']}")
    print(f"Unique texts: {results['dataset_stats']['unique_texts']}")
    print(f"Duplicate texts: {results['dataset_stats']['duplicate_texts']} ({results['dataset_stats']['duplicate_percentage']})")
    print(f"\nAverage labels per row: {results['avg_labels_per_row']:.2f}")
    
    print("\nLabel distribution:")
    for label, count in results['label_distribution'].items():
        print(f"  {label}: {count} ({count/total_rows*100:.1f}%)")
    
    print("\nSample rows:")
    for i, row in enumerate(results['sample_rows'], 1):
        print(f"\nRow {i}:")
        print(f"Text: {row[text_col][:100]}...")
        print(f"Labels: {row[label_col]}")
    
    return results

if __name__ == '__main__':
    analyze_dataset('data/sample.csv')