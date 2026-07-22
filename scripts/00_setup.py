# --- Standard library imports ---
import os
from pathlib import Path

def setup_vault():
    vault_paths = [
        "../data/raw/genomic/a_flavus",
        "../data/raw/genomic/s_aureus",
        "../data/raw/genomic/negative_controls",
        "../data/structural/pdb",
        "../data/structural/alphafold",
        "../data/processed/genomic",
        "../data/processed/structural",
        "../data/metadata/amrfinder_results"
    ]
    for path in vault_paths:
        Path(path).mkdir(parents=True, exist_ok=True)
    print("Vault initialized: All required directory structures verified.")

if __name__ == "__main__":
    setup_vault()