# --- Standard library imports ---
import subprocess     # to call the 'datasets' CLI tool from Python
import shutil            # to check whether 'datasets' is installed
import zipfile              # to unzip downloaded genome packages
import json                    # to parse the JSON-lines accession summary
import random                     # to draw a reproducible random sample of accessions
import sys                           # to exit cleanly on missing tools
import time                             # to pause briefly between retries
from pathlib import Path                    # for clean, cross-platform file paths


# ============================================================
# CONFIGURATION
# ============================================================
RAW_GENOMIC_DIR = Path("../data/raw/genomic")

MAX_SAUREUS_GENOMES = 200   # a manageable, representative sample -- not all 2,108
RANDOM_SEED = 42               # fixed seed -- same sample every time this script runs
MAX_RETRIES = 3                   # how many times to retry a failed download before giving up
RETRY_DELAY_SECONDS = 5              # pause between retry attempts


# ============================================================
# STEP 0: Confirm 'datasets' CLI is installed
# ============================================================
def check_datasets_tool_installed():
    if shutil.which("datasets") is None:
        print("[ERROR] The 'datasets' command-line tool was not found.")
        print("        Install it with: conda install -c conda-forge ncbi-datasets-cli")
        sys.exit(1)
    print("[OK] 'datasets' CLI found -- proceeding with real downloads.\n")


# ============================================================
# A small, reusable helper: run a subprocess command with automatic retries -- because we've now seen, more than once, that a failed network call often just needs one more attempt.
# ============================================================
def run_with_retries(command: list, description: str) -> subprocess.CompletedProcess:
    """
    Runs a command via subprocess, retrying up to MAX_RETRIES times if it fails, with a short pause between attempts.
    """
    last_result = None
    for attempt in range(1, MAX_RETRIES + 1):
        if attempt > 1:
            print(f"    Retry attempt {attempt}/{MAX_RETRIES} for: {description}")
            time.sleep(RETRY_DELAY_SECONDS)

        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            return result   # success -- no need to retry further

        last_result = result
        print(f"    [Attempt {attempt} failed] {description}")

    # If we reach here, every attempt failed -- return the last result so the caller can inspect and report the real error.
    return last_result


# ============================================================
# STEP 1: Get the full list of available accessions for a taxon, so we can control exactly how many genomes we actually download.
# ============================================================
def get_available_accessions(taxon_name: str, assembly_level: str, released_after: str) -> list:
    """
    Queries NCBI for the list of ALL genome accessions matching our filters, WITHOUT downloading any sequence data yet -- this is a small, fast metadata-only query. Restricted to RefSeq -- suitable for well-curated organisms like S. aureus, where RefSeq coverage is genuinely large.
    """
    command = [
        "datasets", "summary", "genome", "taxon", taxon_name,
        "--assembly-level", assembly_level,
        "--assembly-source", "RefSeq",
        "--released-after", released_after,
        "--as-json-lines",
    ]

    result = run_with_retries(command, f"fetching accession list for {taxon_name}")

    if result.returncode != 0:
        print(f"[WARNING] Could not fetch accession list for {taxon_name}:")
        print(result.stderr)
        return []

    accessions = []
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        record = json.loads(line)
        accession = record.get("accession")
        if accession:
            accessions.append(accession)

    return accessions


def get_available_accessions_any_source(taxon_name: str, assembly_level: str) -> list:
    """
    Same idea as get_available_accessions(), but WITHOUT restricting to RefSeq -- includes GenBank-deposited genomes too. This is necessary for organisms like A. flavus, where RefSeq's small curated subset left us with only 1 real sample (confirmed empirically), even though 337 real assemblies actually exist once GenBank is included. No --released-after filter here either, since we need the full available pool for an organism with much sparser deposits overall.
    """
    command = [
        "datasets", "summary", "genome", "taxon", taxon_name,
        "--assembly-level", assembly_level,
        "--as-json-lines",
    ]

    result = run_with_retries(command, f"fetching accession list for {taxon_name} (any source)")

    if result.returncode != 0:
        print(f"[WARNING] Could not fetch accession list for {taxon_name}:")
        print(result.stderr)
        return []

    accessions = []
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        record = json.loads(line)
        accession = record.get("accession")
        if accession:
            accessions.append(accession)

    return accessions


# ============================================================
# STEP 2: Download a specific, controlled list of accessions, IN SMALL BATCHES -- large single downloads (200+ genomes, 200+ MB) were failing zip validation on this connection, even though the raw bytes transferred successfully. Smaller batches are much less likely to hit this, and any batch that DOES fail can be retried on its own without redoing everything.
# ============================================================
BATCH_SIZE = 20   # genomes per batch -- small enough to be reliable, large enough to be efficient


def is_zip_valid(zip_path: Path) -> bool:
    """
    Actually opens the downloaded zip file and checks its internal integrity, rather than just trusting that the download "finished." This is exactly the check that would have caught the corruption we saw -- NCBI's own validator caught it, so we replicate that check here too, explicitly, so our retry logic can react to it.
    """
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            # testzip() reads through every file in the archive and returns the name of the FIRST corrupted file found, or None if everything checks out.
            bad_file = zip_ref.testzip()
            return bad_file is None
    except zipfile.BadZipFile:
        return False


def download_one_batch(accessions_batch: list, out_dir: Path, batch_label: str) -> bool:
    """
    Downloads and validates ONE batch of accessions. Returns True only if the download completes AND the resulting zip file passes a real integrity check.
    """
    accession_file = out_dir / f"_accession_list_{batch_label}.txt"
    with open(accession_file, "w") as f:
        f.write("\n".join(accessions_batch))

    zip_path = out_dir / f"{batch_label}.zip"
    command = [
        "datasets", "download", "genome", "accession",
        "--inputfile", str(accession_file),
        "--include", "genome,gff3",
        "--filename", str(zip_path),
    ]

    for attempt in range(1, MAX_RETRIES + 1):
        if attempt > 1:
            print(f"      Retry {attempt}/{MAX_RETRIES} for batch {batch_label}...")
            time.sleep(RETRY_DELAY_SECONDS)

        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode == 0 and zip_path.exists() and is_zip_valid(zip_path):
            # Real success: the CLI reported success AND the zip file itself genuinely passes integrity validation.
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(out_dir)
            zip_path.unlink()
            accession_file.unlink(missing_ok=True)
            return True

        # Either the command failed, or the zip came back corrupted -- either way, clean up before retrying so we start fresh.
        zip_path.unlink(missing_ok=True)

    accession_file.unlink(missing_ok=True)
    return False


def download_by_accession_list(accessions: list, out_dir: Path, label: str):
    """
    Downloads genome data for a specific, pre-selected list of accessions, split into small batches for reliability. Reports honestly on exactly how many genomes were actually, successfully downloaded -- not just whether the process "ran."
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Split the full accession list into batches of BATCH_SIZE.
    batches = [accessions[i:i + BATCH_SIZE] for i in range(0, len(accessions), BATCH_SIZE)]
    print(f">>> Downloading {len(accessions)} real genome(s) for: {label}")
    print(f"    Split into {len(batches)} batches of up to {BATCH_SIZE} genomes each "
          f"(smaller, more reliable transfers)\n")

    successful_batches = 0
    failed_batch_numbers = []

    for i, batch in enumerate(batches, start=1):
        batch_label = f"{label}_batch{i:03d}"
        print(f"    Batch {i}/{len(batches)} ({len(batch)} genomes)...")

        if download_one_batch(batch, out_dir, batch_label):
            successful_batches += 1
            print(f"      Batch {i} succeeded and passed integrity validation.")
        else:
            failed_batch_numbers.append(i)
            print(f"      [WARNING] Batch {i} failed after {MAX_RETRIES} attempts -- skipping.")

    print(f"\n    Batch summary: {successful_batches}/{len(batches)} batches succeeded.")
    if failed_batch_numbers:
        print(f"    Failed batch numbers: {failed_batch_numbers}")
        print(f"    (Real genomes from successful batches are still saved -- this is a")
        print(f"     partial success, not a total failure. Re-run this script to retry")
        print(f"     just the missing genomes if needed.)")

    print(f"\n    Done. Real genome files now saved under {out_dir}/\n")
    return successful_batches > 0


# ============================================================
# STEP 3: Simpler download path for small/reference-only targets (A. flavus, and negative controls) -- these don't need the accession-sampling approach since they're small to begin with.
# ============================================================
def download_genome_package(taxon_name: str, out_dir: Path, assembly_level: str = None,
                             released_after: str = None, reference_only: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{taxon_name.replace(' ', '_')}.zip"

    print(f">>> Downloading genome(s) for: {taxon_name}")

    command = ["datasets", "download", "genome", "taxon", taxon_name]

    # Only apply an assembly-level filter if one was actually given -- combining --reference with a strict level filter is what broke the Candida albicans download last time.
    if assembly_level:
        command += ["--assembly-level", assembly_level]

    command += ["--assembly-source", "RefSeq", "--include", "genome,gff3", "--filename", str(zip_path)]

    if released_after:
        command += ["--released-after", released_after]
    if reference_only:
        command += ["--reference"]

    result = run_with_retries(command, f"downloading {taxon_name}")

    if result.returncode != 0:
        print(f"[WARNING] Download failed for {taxon_name} after {MAX_RETRIES} attempts:")
        print(result.stderr)
        return False

    print(f"    Download complete: {zip_path.name}")
    print(f"    Unzipping into {out_dir}/ ...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(out_dir)
    zip_path.unlink()

    print(f"    Done. Real genome files now saved under {out_dir}/\n")
    return True


# ============================================================
# MAIN
# ============================================================
def acquire_targets():
    print("Initiating Phase 1 Genomic Acquisition (REAL downloads, v2)...")
    print("-" * 55)

    check_datasets_tool_installed()
    random.seed(RANDOM_SEED)

    # --- A. flavus: previously restricted to RefSeq-only, which left us with just 1 real sample (confirmed: 337 total assemblies actually exist once GenBank-deposited genomes are included, not just the small RefSeq-curated subset). Fixed to use the same accession-sampling + batched download approach already proven reliable for S. aureus, targeting a genuinely usable sample size instead. ---
    MAX_AFLAVUS_GENOMES = 150

    print(">>> Fetching full A. flavus accession list (metadata only, fast)...")
    aflavus_accessions = get_available_accessions_any_source(
        "Aspergillus flavus",
        assembly_level="chromosome,complete,scaffold",
    )
    print(f"    Found {len(aflavus_accessions)} total available accessions "
          f"(RefSeq + GenBank combined).")

    if aflavus_accessions:
        sample_size = min(MAX_AFLAVUS_GENOMES, len(aflavus_accessions))
        sampled_aflavus = random.sample(aflavus_accessions, sample_size)
        print(f"    Randomly sampling {sample_size} accessions (seed={RANDOM_SEED}) "
              f"for a manageable, reproducible, class-balanced dataset.\n")
        download_by_accession_list(sampled_aflavus, RAW_GENOMIC_DIR / "a_flavus", "a_flavus_sample")
    else:
        print("[WARNING] No A. flavus accessions retrieved -- skipping download.\n")

    # --- S. aureus: too large for a direct download -- sample a controlled subset ---
    print(">>> Fetching full S. aureus accession list (metadata only, fast)...")
    all_accessions = get_available_accessions(
        "Staphylococcus aureus",
        assembly_level="chromosome,complete",
        released_after="2018-01-01",
    )
    print(f"    Found {len(all_accessions)} total available accessions.")

    if all_accessions:
        sample_size = min(MAX_SAUREUS_GENOMES, len(all_accessions))
        sampled_accessions = random.sample(all_accessions, sample_size)
        print(f"    Randomly sampling {sample_size} accessions (seed={RANDOM_SEED}) for a manageable, "
              f"reproducible download.\n")
        download_by_accession_list(sampled_accessions, RAW_GENOMIC_DIR / "s_aureus", "s_aureus_sample")
    else:
        print("[WARNING] No S. aureus accessions retrieved -- skipping download.\n")

    # --- Negative controls: --reference only, no assembly-level filter ---
    for species_name, folder_name in [
        ("Escherichia coli", "Escherichia_coli"),
        ("Candida albicans", "Candida_albicans"),
    ]:
        download_genome_package(
            species_name,
            RAW_GENOMIC_DIR / "negative_controls" / folder_name,
            assembly_level=None,   # deliberately omitted this time -- see docstring note above
            reference_only=True,
        )

    print("-" * 55)
    print("Genomic acquisition complete. Check ../data/raw/genomic/ for real files.")


if __name__ == "__main__":
    acquire_targets()
