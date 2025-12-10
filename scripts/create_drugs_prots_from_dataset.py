"""
Extract unique drugs and proteins from dataset file (e.g., kd_bind.txt)
and create drugs.csv and prots.csv files
"""

import pandas as pd
import argparse
import os


def create_drugs_prots_files(dataset_file, output_dir, dataset_name):
    """
    Extract unique drugs and proteins from dataset file
    
    Args:
        dataset_file: Path to dataset .txt file (e.g., kd_bind.txt)
        output_dir: Output directory for drugs.csv and prots.csv
        dataset_name: Dataset name (e.g., kd_bind)
    """
    
    print(f"{'='*80}")
    print(f"Creating drugs.csv and prots.csv for {dataset_name}")
    print(f"{'='*80}")
    print(f"Input:  {dataset_file}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\n")
    
    # Read dataset file
    print("Loading dataset...")
    df = pd.read_csv(dataset_file, sep=' ', header=None)
    df.columns = ['drug_id', 'prot_id', 'drug_smile', 'prot_seq', 'label']
    print(f"✓ Loaded {len(df):,} entries")
    
    # Extract unique drugs
    print("\nExtracting unique drugs...")
    drugs_df = df[['drug_id', 'drug_smile']].drop_duplicates(subset='drug_id')
    drugs_df = drugs_df.sort_values('drug_id').reset_index(drop=True)
    print(f"✓ Found {len(drugs_df):,} unique drugs")
    
    # Extract unique proteins
    print("\nExtracting unique proteins...")
    prots_df = df[['prot_id', 'prot_seq']].drop_duplicates(subset='prot_id')
    prots_df = prots_df.sort_values('prot_id').reset_index(drop=True)
    print(f"✓ Found {len(prots_df):,} unique proteins")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save drugs.csv
    drugs_file = os.path.join(output_dir, f'{dataset_name}_drugs.csv')
    drugs_df.to_csv(drugs_file, index=False)
    print(f"\n✓ Saved {drugs_file}")
    
    # Save prots.csv
    prots_file = os.path.join(output_dir, f'{dataset_name}_prots.csv')
    prots_df.to_csv(prots_file, index=False)
    print(f"✓ Saved {prots_file}")
    
    # Statistics
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total entries:          {len(df):,}")
    print(f"Unique drugs:           {len(drugs_df):,}")
    print(f"Unique proteins:        {len(prots_df):,}")
    print(f"\nDrug SMILES length:")
    print(f"  Mean:                 {drugs_df['drug_smile'].str.len().mean():.0f} chars")
    print(f"  Max:                  {drugs_df['drug_smile'].str.len().max():.0f} chars")
    print(f"\nProtein sequence length:")
    print(f"  Mean:                 {prots_df['prot_seq'].str.len().mean():.0f} aa")
    print(f"  Max:                  {prots_df['prot_seq'].str.len().max():.0f} aa")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract drugs and proteins from dataset file')
    parser.add_argument('--dataset-file', type=str, required=True,
                       help='Path to dataset .txt file')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for drugs.csv and prots.csv')
    parser.add_argument('--dataset-name', type=str, required=True,
                       help='Dataset name (e.g., kd_bind, ki_bind)')
    
    args = parser.parse_args()
    
    create_drugs_prots_files(
        dataset_file=args.dataset_file,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name
    )
