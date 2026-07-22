# --- Standard library imports ---
import requests
import urllib3
from pathlib import Path
from remotezip import RemoteZip

# Suppress the InsecureRequestWarning that verify=False triggers -- we
# already know why we're doing this (see download_dibas_species below)
# and don't need it repeated on every request.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ============================================================
# CONFIGURATION
# ============================================================
OUT_DIR = Path("../data/images")

DIBAS_BASE_URL = "https://doctoral.matinf.uj.edu.pl/database/dibas"
OPENFUNGI_ZIP_URL = "https://zenodo.org/records/15692070/files/openfungi.zip?download=1"


# ============================================================
# STEP 1: Download DIBaS species (small, direct zips -- no special handling needed, each is only a few MB)
# ============================================================
def download_dibas_species(species_filename: str, out_subdir: str):
    """
    Downloads one species' real DIBaS zip file directly (these are small -- roughly 20 images each -- so no partial-download trick is needed here, unlike OpenFungi).

    Skips the download entirely if this species' images already exist on disk from a previous run -- avoids redundant re-downloads when re-running this script after adding a new organism.
    """
    out_dir = OUT_DIR / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use the species name (without .zip) as a marker to check whether we've already extracted this specific species' images before.
    species_marker = species_filename.replace(".zip", "")
    already_present = any(species_marker.split(".")[0].lower() in f.name.lower()
                           for f in out_dir.glob("*") if f.is_file())

    if already_present:
        print(f"  [SKIPPED] {species_filename} images already present in {out_dir}/ -- not re-downloading.")
        return

    url = f"{DIBAS_BASE_URL}/{species_filename}"
    zip_path = out_dir / species_filename

    print(f"  Fetching real DIBaS images: {species_filename} -> {out_dir}")

    # NOTE: DIBaS's server (a university department host, not a major CDN) has an incomplete SSL certificate chain -- confirmed via curl showing the same "unable to get local issuer certificate" error independent of this script's Python environment. This is a server-side misconfiguration common on smaller academic hosts, not a sign of a genuinely insecure or spoofed connection. Since we're only fetching public, non-sensitive read-only data (not submitting credentials or anything private), verify=False is a reasonable, disclosed tradeoff here -- not something to do routinely or silently.
    response = requests.get(url, timeout=60, verify=False)
    response.raise_for_status()

    with open(zip_path, "wb") as f:
        f.write(response.content)

    # Unzip and clean up
    import zipfile
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(out_dir)
    zip_path.unlink()

    print(f"    Done. Real images saved under {out_dir}/")


# ============================================================
# STEP 2: Selectively download ONLY the "Flavi" subset from the large OpenFungi archive, without downloading the full 8.4GB.
# ============================================================
def download_openfungi_flavi_subset():
    """
    Uses remotezip to read the OpenFungi archive's file listing via HTTP range requests (a small, fast operation), then downloads ONLY the files whose path contains "flavi" -- our actual target -- leaving the other several GB of unrelated genera untouched.

    Skips entirely if A. flavus images already exist on disk from a previous run.
    """
    out_dir = OUT_DIR / "aflavus"
    out_dir.mkdir(parents=True, exist_ok=True)

    existing_files = list(out_dir.rglob("*.*"))
    if existing_files:
        print(f"  [SKIPPED] {len(existing_files)} A. flavus image file(s) already present "
              f"in {out_dir}/ -- not re-downloading.")
        return len(existing_files)

    print("  Reading OpenFungi archive's file listing (fast, no full download)...")

    with RemoteZip(OPENFUNGI_ZIP_URL) as zip_ref:
        all_files = zip_ref.namelist()
        print(f"    Archive contains {len(all_files)} total files across all genera.")

        # Filter to ONLY the microscopic images labeled with "Flavi" in their path -- our real, specific target within this large archive.
        flavi_files = [f for f in all_files if "flavi" in f.lower() and "micro" in f.lower()]
        print(f"    Found {len(flavi_files)} real 'Aspergillus section Flavi' microscopic images.")

        if not flavi_files:
            print("    [WARNING] No matching files found -- archive folder structure may have")
            print("              changed. Inspect `all_files` manually to find the correct path.")
            return 0

        print(f"  Downloading only these {len(flavi_files)} real files (not the full 8.4GB archive)...")
        for i, filename in enumerate(flavi_files, start=1):
            zip_ref.extract(filename, path=out_dir)
            if i % 10 == 0 or i == len(flavi_files):
                print(f"    Extracted {i}/{len(flavi_files)} real images...")

    print(f"    Done. Real A. flavus (section Flavi) images saved under {out_dir}/")
    return len(flavi_files)


# ============================================================
# MAIN
# ============================================================
def acquire_images():
    print("Phase 1 (continued): Real Image Data Acquisition")
    print("=" * 60)

    print("\n[1/4] Downloading real S. aureus microscopy images (DIBaS)...")
    download_dibas_species("Staphylococcus.aureus.zip", "saureus")

    print("\n[2/4] Downloading real Candida albicans images (DIBaS) -- negative control...")
    download_dibas_species("Candida.albicans.zip", "negative_control")

    print("\n[3/4] Downloading real Escherichia coli images (DIBaS) -- our SECOND negative")
    print("       control, matching the two-organism negative control set already used")
    print("       for genomic data in 01_download_genomes.py (this was missed in the")
    print("       first version of this script -- fixed here).")
    download_dibas_species("Escherichia.coli.zip", "negative_control")

    print("\n[4/4] Downloading real A. flavus (section Flavi) images from OpenFungi "
          "(selective, not the full 8.4GB archive)...")
    download_openfungi_flavi_subset()

    print("\n" + "=" * 60)
    print("Real image acquisition complete.")
    print("\nHONEST NOTES FOR THE WRITEUP:")
    print("  - S. aureus: 20 real images, DIBaS (Zielinski et al., 2017)")
    print("  - Negative control (Candida albicans): DIBaS, consistent with our")
    print("    genomic negative control choice")
    print("  - A. flavus: real images labeled 'Aspergillus section Flavi'")
    print("    (OpenFungi, Cighir et al., 2025) -- this section includes A. flavus")
    print("    and close relatives (A. oryzae, A. parasiticus); NOT confirmed as")
    print("    A. flavus specifically at the species level. State this plainly.")
    print("  - Sample sizes here are small (tens of images per class) -- a real")
    print("    limitation for training a deep CNN like ResNet18 without serious")
    print("    data augmentation or transfer learning safeguards against overfitting.")


if __name__ == "__main__":
    acquire_images()
