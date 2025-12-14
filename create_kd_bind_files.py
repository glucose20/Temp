"""
Quick script to create kd_bind_drugs.csv and kd_bind_prots.csv from Kd_bind.tsv
"""

import pandas as pd
import os

# Read the TSV file
print("Loading Kd_bind.tsv...")
df = pd.read_csv('data/Kd_bind.tsv', sep='\t')
print(f"Loaded {len(df):,} entries")
print(f"Columns: {list(df.columns)}")

# The TSV has columns: drug_id, target_id, smiles, target_seq, origin_affinity, affinity
# We need to rename to match what the code expects: drug_smile and prot_seq

# Extract unique drugs
print("\nExtracting unique drugs...")
drugs_df = df[['drug_id', 'smiles']].drop_duplicates(subset='drug_id')
drugs_df = drugs_df.rename(columns={'smiles': 'drug_smile'})  # Rename to drug_smile
drugs_df = drugs_df.sort_values('drug_id').reset_index(drop=True)
print(f"Found {len(drugs_df):,} unique drugs")

# Extract unique proteins  
print("\nExtracting unique proteins...")
prots_df = df[['target_id', 'target_seq']].drop_duplicates(subset='target_id')
prots_df = prots_df.rename(columns={'target_id': 'prot_id', 'target_seq': 'prot_seq'})  # Rename columns
prots_df = prots_df.sort_values('prot_id').reset_index(drop=True)
print(f"Found {len(prots_df):,} unique proteins")

# Create output directory
os.makedirs('data/dta-5fold-dataset/kd_bind', exist_ok=True)

# Save CSV files
drugs_file = 'data/dta-5fold-dataset/kd_bind/kd_bind_drugs.csv'
prots_file = 'data/dta-5fold-dataset/kd_bind/kd_bind_prots.csv'

drugs_df.to_csv(drugs_file, index=False)
print(f"\n✓ Saved {drugs_file}")
print(f"  Columns: {list(drugs_df.columns)}")

prots_df.to_csv(prots_file, index=False)
print(f"✓ Saved {prots_file}")
print(f"  Columns: {list(prots_df.columns)}")

print("\n" + "="*80)
print("DONE! CSV files created successfully.")
print("="*80)
