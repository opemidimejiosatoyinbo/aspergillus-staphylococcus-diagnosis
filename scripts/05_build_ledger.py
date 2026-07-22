# --- Standard library imports ---
import csv                     # for writing the ledger as a CSV file
import random                    # used ONLY for the train/test split assignment (a legitimate use)
from pathlib import Path             # for clean, cross-platform file paths
from datetime import date               # to timestamp when each entry was catalogued


# ============================================================
# CONFIGURATION
# These paths match exactly what earlier scripts actually produced.
# ============================================================
RAW_GENOMIC_DIR = Path("../data/raw/genomic")
STRUCTURAL_PDB_DIR = Path("../data/structural/pdb")
STRUCTURAL_AF_DIR = Path("../data/structural/alphafold")
RESISTANCE_CSV = Path("../data/metadata/resistance_calls.csv")   # produced by 04_call_resistance.py
OUT_CSV = Path("../data/metadata/immutable_ledger.csv")

# Fraction of samples held out as a strict, untouched test set -- decided once, here, before any model touches the data
TEST_SPLIT_FRACTION = 0.15

# Using a fixed random seed means the SAME samples end up in the test set every time this script runs -- important for reproducibility. Without this, re-running the script could accidentally shuffle a sample from train into test between runs, which would quietly leak information and undermine the whole point of a held-out test set.
random.seed(42)


# ============================================================
# STEP 1: Scan real genomic files actually present on disk
# ============================================================
def scan_real_genomic_samples() -> list[dict]:
    """
    Walks the actual genomic data folders and creates one row per REAL genome file found. No fabricated counts -- if a folder is empty, it contributes zero rows, honestly.
    """
    rows = []

    # Maps each real subfolder to the label/organism it represents.
    folder_config = {
        "a_flavus":  {"organism": "Aspergillus_flavus",     "label_positive": 1},
        "s_aureus":  {"organism": "Staphylococcus_aureus",   "label_positive": 1},
    }

    for subfolder, meta in folder_config.items():
        search_dir = RAW_GENOMIC_DIR / subfolder
        # rglob searches recursively -- genome files from NCBI Datasets typically end up nested inside per-accession subfolders.
        for fasta_path in search_dir.rglob("*.fna"):
            sample_id = fasta_path.parent.name   # NCBI accession folder name, e.g. GCF_000...
            rows.append({
                "sample_id": sample_id,
                "organism": meta["organism"],
                "label_positive": meta["label_positive"],
                "resistance_status": "NA",   # filled in below, only for S. aureus, from real AMR results
                "data_type": "genomic",
                "file_path": str(fasta_path),
                "source_db": "NCBI (RefSeq)",
                "split": None,   # assigned later, after we know the real total count
                "date_acquired": str(date.today()),
            })

    # Negative controls get their own pass, since their folder structure is slightly different (named by species under negative_controls/).
    negative_dir = RAW_GENOMIC_DIR / "negative_controls"
    for fasta_path in negative_dir.rglob("*.fna"):
        sample_id = fasta_path.parent.name
        rows.append({
            "sample_id": sample_id,
            "organism": "negative_control",
            "label_positive": 0,
            "resistance_status": "NA",
            "data_type": "genomic",
            "file_path": str(fasta_path),
            "source_db": "NCBI (RefSeq)",
            "split": None,
            "date_acquired": str(date.today()),
        })

    return rows


# ============================================================
# STEP 2: Merge in REAL resistance calls from 04_call_resistance.py
# ============================================================
def merge_real_resistance_calls(genomic_rows: list[dict]) -> list[dict]:
    """
    Reads the real resistance_calls.csv produced by 04_call_resistance.py and fills in the resistance_status field for matching S. aureus samples. If that file doesn't exist yet, resistance stays "NA" and a warning is printed -- rather than silently guessing.
    """
    if not RESISTANCE_CSV.exists():
        print(f"[WARNING] {RESISTANCE_CSV} not found -- resistance_status will")
        print("          remain 'NA' for all S. aureus samples. Run")
        print("          04_call_resistance.py first for real resistance labels.")
        return genomic_rows

    # Build a lookup: sample_id -> resistance_status, from the real AMR results
    resistance_lookup = {}
    with open(RESISTANCE_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            resistance_lookup[row["sample_id"]] = row["resistance_status"]

    for row in genomic_rows:
        if row["sample_id"] in resistance_lookup:
            row["resistance_status"] = resistance_lookup[row["sample_id"]]

    return genomic_rows


# ============================================================
# STEP 3: Scan real structural files actually present on disk
# ============================================================
def scan_real_structural_samples() -> list[dict]:
    """
    Walks the real structural data folders (PDB + AlphaFold) and creates one row per REAL .pdb file found.
    """
    rows = []

    for pdb_path in STRUCTURAL_PDB_DIR.glob("*.pdb"):
        rows.append({
            "sample_id": pdb_path.stem,   # filename without extension
            "organism": "Staphylococcus_aureus",
            "label_positive": "NA",
            "resistance_status": "NA",
            "data_type": "structural_experimental",
            "file_path": str(pdb_path),
            "source_db": "RCSB PDB",
            "split": "TRAIN",   # structural reference data isn't held out for testing
            "date_acquired": str(date.today()),
        })

    for pdb_path in STRUCTURAL_AF_DIR.glob("*.pdb"):
        rows.append({
            "sample_id": pdb_path.stem,
            "organism": "Aspergillus_flavus",
            "label_positive": "NA",
            "resistance_status": "NA",
            "data_type": "structural_predicted",
            "file_path": str(pdb_path),
            "source_db": "AlphaFold DB",
            "split": "TRAIN",
            "date_acquired": str(date.today()),
        })

    return rows


# ============================================================
# STEP 4: Assign the train/test split -- ONCE, on real samples only
# ============================================================
def assign_train_test_split(genomic_rows: list[dict]) -> list[dict]:
    """
    Assigns each REAL genomic sample to either TRAIN or TEST, using a fixed random seed so this split is reproducible across re-runs. Structural samples are excluded from this split (see note above -- they serve as reference structures, not held-out test cases).
    """
    for row in genomic_rows:
        row["split"] = "TEST" if random.random() < TEST_SPLIT_FRACTION else "TRAIN"
    return genomic_rows


# ============================================================
# MAIN: build the real ledger from real files, and nothing else
# ============================================================
def build_ledger():
    print("Scanning genomic and structural folders for REAL files...")
    print("-" * 55)

    genomic_rows = scan_real_genomic_samples()
    print(f"Found {len(genomic_rows)} real genomic file(s) on disk.")

    if len(genomic_rows) == 0:
        print("\n[WARNING] No real genomic files found. This likely means")
        print("          01_download_genomes.py has not been run yet, or")
        print("          failed to download anything. The ledger will still")
        print("          be written, but it will be empty for genomic data.")

    genomic_rows = merge_real_resistance_calls(genomic_rows)
    genomic_rows = assign_train_test_split(genomic_rows)

    structural_rows = scan_real_structural_samples()
    print(f"Found {len(structural_rows)} real structural file(s) on disk.")

    all_rows = genomic_rows + structural_rows

    if len(all_rows) == 0:
        print("\n[ERROR] No real data found anywhere. Nothing to write.")
        print("        Run 01_download_genomes.py and 02_download_structures.py first.")
        return

    # --- Write the real ledger ---
    Path("../data/metadata").mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id", "organism", "label_positive", "resistance_status",
        "data_type", "file_path", "source_db", "split", "date_acquired",
    ]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    # --- Print an honest summary of exactly what got catalogued ---
    train_count = sum(1 for r in genomic_rows if r["split"] == "TRAIN")
    test_count = sum(1 for r in genomic_rows if r["split"] == "TEST")
    resistant_count = sum(1 for r in genomic_rows if r.get("resistance_status") == "RESISTANT")

    print("-" * 55)
    print(f"Ledger written: {OUT_CSV}")
    print(f"  Total real entries:        {len(all_rows)}")
    print(f"  Genomic (train):            {train_count}")
    print(f"  Genomic (test, held out):    {test_count}")
    print(f"  Confirmed resistant samples:  {resistant_count}")
    print(f"  Structural files:             {len(structural_rows)}")
    print("\nEvery row above corresponds to a real file that actually exists on disk.")


if __name__ == "__main__":
    build_ledger()
