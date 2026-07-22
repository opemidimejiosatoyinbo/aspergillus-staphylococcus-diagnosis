# --- Standard library imports ---
import subprocess    # lets us call the external 'amrfinder' tool from Python
import shutil          # used to check whether 'amrfinder' is installed
import sys              # used to exit cleanly if a required tool is missing
import csv                # used to write our final, real resistance ledger
from pathlib import Path      # for clean, cross-platform file paths


# ============================================================
# CONFIGURATION
# These paths match what 00_setup.py already created.
# ============================================================
GENOME_DIR = Path("../data/raw/genomic/s_aureus")
RESULTS_DIR = Path("../data/metadata/amrfinder_results")
SUMMARY_CSV = Path("../data/metadata/resistance_calls.csv")


# ============================================================
# STEP 0: Confirm AMRFinderPlus is actually installed
# ============================================================
def check_amrfinder_installed():
    """
    Checks whether the 'amrfinder' command-line tool is available. If missing, prints clear install instructions and exits -- there is no way to do real resistance calling without it.
    """
    if shutil.which("amrfinder") is None:
        print("[ERROR] The 'amrfinder' command-line tool was not found.")
        print("        This script cannot detect resistance genes without it.")
        print()
        print("        Install it now by running these commands in your terminal:")
        print("            conda install -y -c bioconda -c conda-forge ncbi-amrfinderplus")
        print("            amrfinder -u        # downloads the latest reference database")
        print()
        print("        Then re-run this script.")
        sys.exit(1)
    else:
        print("[OK] 'amrfinder' found -- proceeding with real resistance scanning.\n")


# ============================================================
# STEP 1: Find every real, downloaded S. aureus genome file
# ============================================================
def find_genome_files():
    """
    Searches the S. aureus genome folder for real FASTA (.fna) files that were actually downloaded by 01_download_genomes.py.

    Returns a list of Path objects. If this list is empty, it means Phase 1's genome download step needs to be run (or re-run) before this script can do anything meaningful.
    """
    genome_files = list(GENOME_DIR.rglob("*.fna"))

    if not genome_files:
        print("[ERROR] No .fna genome files found under", GENOME_DIR)
        print("        Run 01_download_genomes.py first -- there is nothing")
        print("        real for this script to scan yet.")
        sys.exit(1)

    print(f"[OK] Found {len(genome_files)} real S. aureus genome file(s) to scan.\n")
    return genome_files


# ============================================================
# STEP 2: Run AMRFinderPlus on ONE genome file, and interpret whether mecA or mecC was actually found in the results.
# ============================================================
def scan_genome_for_resistance(genome_path: Path) -> dict:
    """
    Runs AMRFinderPlus against a single real genome FASTA file and checks its output for the mecA / mecC resistance genes.

    Returns a dictionary summarizing what was found for this sample -- this becomes one row in our final resistance_calls.csv.
    """
    sample_id = genome_path.parent.name
    out_file = RESULTS_DIR / f"{sample_id}_amr.tsv"

    # OPTIMIZATION: if we already have a real AMRFinder result file for this sample from a previous run, don't re-scan it (that took a genuinely long time for 200 real genomes) -- just re-parse the existing real output with the corrected column name below.
    if out_file.exists():
        print(f"  Re-parsing existing result for {sample_id} (skipping re-scan)...")
    else:
        print(f"  Scanning {sample_id} (no existing result found)...")

        # This is the REAL command being run -- equivalent to typing this directly into the terminal yourself: amrfinder -n <genome.fna> --organism Staphylococcus_aureus --output <out_file>
        command = [
            "amrfinder",
            "-n", str(genome_path),                    # -n = nucleotide FASTA input
            "--organism", "Staphylococcus_aureus",       # tells AMRFinder which organism-specific rules to apply
            "--output", str(out_file),
        ]

        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"    [WARNING] AMRFinder failed on {sample_id}:")
            print(f"    {result.stderr.strip()}")
            return {
                "sample_id": sample_id,
                "genome_file": str(genome_path),
                "resistance_status": "SCAN_FAILED",
                "genes_detected": "",
            }

    # --- Read AMRFinder's real output and check for mecA / mecC ---
    detected_genes = []
    if out_file.exists():
        with open(out_file, "r") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                # BUG FIX: AMRFinder's real output column is called "Element symbol", NOT "Gene symbol" -- the earlier version of this script used the wrong column name, which meant this check silently found nothing on EVERY sample, regardless of what AMRFinder actually detected. Confirmed by inspecting a real output file directly, which showed genuine resistance markers (e.g. parC_S80F, fosB) sitting under "Element symbol".
                gene_symbol = row.get("Element symbol", "")

                # Real AMRFinder output can report mecA/mecC with allele or variant suffixes (e.g. "mecA1", "mecA_5"), so we check whether the symbol STARTS WITH "meca"/"mecc" rather than requiring an exact match -- this catches genuine variants without loosening the match so much that we'd accidentally catch unrelated genes.
                symbol_lower = gene_symbol.lower()
                if symbol_lower.startswith("meca") or symbol_lower.startswith("mecc"):
                    detected_genes.append(gene_symbol)

    resistance_status = "RESISTANT" if detected_genes else "SUSCEPTIBLE"
    print(f"    -> {resistance_status}"
          + (f" ({', '.join(detected_genes)} detected)" if detected_genes else ""))

    return {
        "sample_id": sample_id,
        "genome_file": str(genome_path),
        "resistance_status": resistance_status,
        "genes_detected": ";".join(detected_genes),
    }


# ============================================================
# MAIN: run the full, real resistance-calling process
# ============================================================
def call_resistance():
    print("Phase 1 (continued): Calling resistance genes (mecA/mecC)...")
    print("-" * 55)

    check_amrfinder_installed()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    genome_files = find_genome_files()

    # Scan every real genome file, one at a time, and collect real results.
    all_results = []
    for genome_path in genome_files:
        result = scan_genome_for_resistance(genome_path)
        all_results.append(result)

    # --- Write the final, real summary CSV ---
    fieldnames = ["sample_id", "genome_file", "resistance_status", "genes_detected"]
    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    # --- Print an honest summary of what was actually found ---
    resistant_count = sum(1 for r in all_results if r["resistance_status"] == "RESISTANT")
    susceptible_count = sum(1 for r in all_results if r["resistance_status"] == "SUSCEPTIBLE")
    failed_count = sum(1 for r in all_results if r["resistance_status"] == "SCAN_FAILED")

    print("-" * 55)
    print(f"Resistance scanning complete on {len(all_results)} real genome(s):")
    print(f"  RESISTANT (mecA/mecC detected): {resistant_count}")
    print(f"  SUSCEPTIBLE (no mecA/mecC):      {susceptible_count}")
    print(f"  Failed scans:                    {failed_count}")
    print(f"\nReal per-sample results saved in: {RESULTS_DIR}/")
    print(f"Summary table saved to:            {SUMMARY_CSV}")


if __name__ == "__main__":
    call_resistance()
