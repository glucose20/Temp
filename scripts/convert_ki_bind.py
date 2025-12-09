"""
Convert Ki_bind.tsv to DTA training format
Filter by sequence length ONLY (no affinity filtering)
"""

import pandas as pd
import argparse


def convert_ki_bind(input_file, output_file, min_seq_len=50, max_seq_len=1022):
    """
    Convert Ki_bind.tsv to DTA format with sequence length filtering only
    
    Args:
        input_file: Path to Ki_bind.tsv
        output_file: Output file path
        min_seq_len: Minimum protein sequence length
        max_seq_len: Maximum protein sequence length (1022 for ESM2, 2048 for ESM-C)
    """
    
    print(f"{'='*80}")
    print("Ki_bind.tsv to DTA Converter (Sequence Length Filter Only)")
    print(f"{'='*80}")
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print(f"Sequence length: {min_seq_len}-{max_seq_len} aa")
    print(f"Affinity: NO FILTERING (keep all values)")
    print(f"{'='*80}\n")
    
    # Read file
    print("Loading data...")
    df = pd.read_csv(input_file, sep='\t')
    print(f"✓ Loaded {len(df):,} entries")
    
    original_count = len(df)
    
    # Check for required columns
    required_cols = ['drug_id', 'target_id', 'smiles', 'target_seq', 'affinity']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Missing columns: {missing_cols}")
        return
    
    # Filter by sequence length ONLY
    print("\nFiltering by sequence length...")
    df['seq_len'] = df['target_seq'].str.len()
    before = len(df)
    df = df[(df['seq_len'] >= min_seq_len) & (df['seq_len'] <= max_seq_len)]
    print(f"  Kept {len(df):,} / {before:,} entries ({len(df)/before*100:.1f}%)")
    
    # Remove any NaN values
    print("\nRemoving entries with missing values...")
    before = len(df)
    df = df.dropna(subset=['drug_id', 'target_id', 'smiles', 'target_seq', 'affinity'])
    print(f"  Kept {len(df):,} / {before:,} entries")
    
    # Remove duplicates (keep highest affinity)
    print("\nRemoving duplicates...")
    before = len(df)
    df = df.sort_values('affinity', ascending=False)
    df = df.drop_duplicates(subset=['drug_id', 'target_id'], keep='first')
    print(f"  Kept {len(df):,} / {before:,} entries ({len(df)/before*100:.1f}%)")
    
    # Rename columns to match davis.txt format
    # davis.txt format: drug_id prot_id drug_smile prot_seq label
    df_output = df[['drug_id', 'target_id', 'smiles', 'target_seq', 'affinity']].copy()
    df_output.columns = ['drug_id', 'prot_id', 'drug_smile', 'prot_seq', 'label']
    
    # Statistics
    print(f"\n{'='*80}")
    print("FINAL STATISTICS")
    print(f"{'='*80}")
    print(f"Original entries:       {original_count:>12,}")
    print(f"Final entries:          {len(df_output):>12,}")
    print(f"Reduction:              {(1-len(df_output)/original_count)*100:>11.1f}%")
    print(f"\nUnique drugs:           {df_output['drug_id'].nunique():>12,}")
    print(f"Unique proteins:        {df_output['prot_id'].nunique():>12,}")
    print(f"\nAffinity (pKd):")
    print(f"  Mean:                 {df_output['label'].mean():>12.2f}")
    print(f"  Median:               {df_output['label'].median():>12.2f}")
    print(f"  Min:                  {df_output['label'].min():>12.2f}")
    print(f"  Max:                  {df_output['label'].max():>12.2f}")
    print(f"  Std:                  {df_output['label'].std():>12.2f}")
    
    # Affinity distribution
    low_aff = (df_output['label'] < 4.0).sum()
    mid_aff = ((df_output['label'] >= 4.0) & (df_output['label'] <= 10.0)).sum()
    high_aff = (df_output['label'] > 10.0).sum()
    print(f"\nAffinity distribution:")
    print(f"  pKd < 4.0 (weak):     {low_aff:>12,} ({low_aff/len(df_output)*100:>5.1f}%)")
    print(f"  4.0 ≤ pKd ≤ 10.0:     {mid_aff:>12,} ({mid_aff/len(df_output)*100:>5.1f}%)")
    print(f"  pKd > 10.0 (strong):  {high_aff:>12,} ({high_aff/len(df_output)*100:>5.1f}%)")
    
    # Sequence length stats
    seq_lens = df_output['prot_seq'].str.len()
    print(f"\nSequence length:")
    print(f"  Mean:                 {seq_lens.mean():>12.0f} aa")
    print(f"  Median:               {seq_lens.median():>12.0f} aa")
    print(f"  Min:                  {seq_lens.min():>12.0f} aa")
    print(f"  Max:                  {seq_lens.max():>12.0f} aa")
    print(f"{'='*80}\n")
    
    # Save to file (space-separated, no header, like davis.txt)
    print(f"Saving to: {output_file}")
    df_output.to_csv(output_file, sep=' ', index=False, header=False)
    print("✓ Saved successfully")
    
    # Also save CSV with headers for inspection
    inspect_file = output_file.replace('.txt', '_inspect.csv')
    df_output.to_csv(inspect_file, index=False)
    print(f"✓ Saved inspection file: {inspect_file}")
    
    print(f"\n{'='*80}")
    print("✅ CONVERSION COMPLETE!")
    print(f"{'='*80}")
    print(f"\nNext steps:")
    print(f"1. Generate embeddings:")
    print(f"   python code_prepareEmb/get_pretrain.py --dataset ki_bind")
    print(f"\n2. Train model:")
    print(f"   python code/train_full.py --dataset ki_bind --epochs 1000")
    print(f"{'='*80}\n")
    
    return df_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert Ki_bind.tsv to DTA format (sequence filter only)')
    parser.add_argument('--input', type=str, 
                       default='./data/Ki_bind.tsv',
                       help='Input Ki_bind.tsv file')
    parser.add_argument('--output', type=str,
                       default='./data/dta-origin-dataset/ki_bind.txt',
                       help='Output file path')
    parser.add_argument('--min-seq-len', type=int, default=50,
                       help='Minimum protein sequence length')
    parser.add_argument('--max-seq-len', type=int, default=1022,
                       help='Maximum protein sequence length (1022 for ESM2, 2048 for ESM-C)')
    
    args = parser.parse_args()
    
    convert_ki_bind(
        input_file=args.input,
        output_file=args.output,
        min_seq_len=args.min_seq_len,
        max_seq_len=args.max_seq_len
    )
