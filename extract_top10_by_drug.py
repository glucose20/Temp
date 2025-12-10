"""
Extract Top 10 proteins with highest affinity for each drug from prediction files
"""

import pandas as pd
import os
from datetime import datetime

def extract_top10_by_drug(input_file, output_file):
    """
    Extract top 10 proteins with highest affinity prediction for each drug
    
    Args:
        input_file: Path to input CSV file with columns: drug_id, prot_id, pred
        output_file: Path to save output CSV
    """
    
    print(f"{'='*80}")
    print(f"Processing: {input_file}")
    print(f"{'='*80}\n")
    
    # Read CSV
    print("Loading data...")
    df = pd.read_csv(input_file)
    total_rows = len(df)
    total_drugs = df['drug_id'].nunique()
    print(f"✓ Loaded {total_rows:,} predictions")
    print(f"✓ Found {total_drugs:,} unique drugs\n")
    
    # Sort by drug_id and pred (descending) to get highest affinity first
    print("Sorting by affinity...")
    df_sorted = df.sort_values(['drug_id', 'pred'], ascending=[True, False])
    
    # Get top 10 for each drug
    print("Extracting top 10 for each drug...")
    top10_list = []
    
    for drug_id, group in df_sorted.groupby('drug_id'):
        top10 = group.head(10)
        top10_list.append(top10)
    
    # Combine all top10 results
    df_top10 = pd.concat(top10_list, ignore_index=True)
    
    print(f"✓ Extracted {len(df_top10):,} rows (top 10 per drug)\n")
    
    # Add rank column within each drug
    print("Adding rank column...")
    df_top10['rank'] = df_top10.groupby('drug_id').cumcount() + 1
    
    # Reorder columns: drug_id, rank, prot_id, pred
    df_top10 = df_top10[['drug_id', 'rank', 'prot_id', 'pred']]
    
    # Save to file
    print(f"Saving to: {output_file}")
    df_top10.to_csv(output_file, index=False)
    print(f"✓ Saved {len(df_top10):,} rows\n")
    
    # Show statistics
    print("Statistics:")
    print(f"  Average affinity in top 10: {df_top10['pred'].mean():.3f}")
    print(f"  Max affinity: {df_top10['pred'].max():.3f}")
    print(f"  Min affinity: {df_top10['pred'].min():.3f}")
    
    # Show sample
    print("\nSample (first drug's top 10):")
    first_drug = df_top10['drug_id'].iloc[0]
    sample = df_top10[df_top10['drug_id'] == first_drug]
    print(sample.to_string(index=False))
    print(f"\n{'='*80}\n")
    
    return df_top10


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract top 10 proteins by affinity for each drug')
    parser.add_argument('--input', type=str, required=True,
                       help='Input CSV file path')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV file path (default: auto-generated)')
    
    args = parser.parse_args()
    
    # Generate output filename if not provided
    if args.output is None:
        base_name = os.path.basename(args.input)
        name_without_ext = os.path.splitext(base_name)[0]
        output_dir = os.path.dirname(args.input)
        args.output = os.path.join(output_dir, f"{name_without_ext}_Top10.csv")
    
    # Process file
    extract_top10_by_drug(args.input, args.output)
    
    print("✅ Done!")
