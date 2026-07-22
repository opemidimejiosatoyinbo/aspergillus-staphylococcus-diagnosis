# AF-SA Diagnostic Pipeline

A real, reproducible, multi-modal deep learning pipeline for *Aspergillus flavus* and *Staphylococcus aureus* identification and methicillin-resistance profiling — built from real genomic sequence, real protein structure, and real microscopy imagery.

Every script in this repository runs on real, publicly sourced data. Nothing here is simulated. Where a result turned out to be flawed during development, that's documented rather than hidden — see [`data/metadata/FINAL_REPORT.md`](data/metadata/FINAL_REPORT.md) and `Final_Roadmap.md` for the full account, including a real bug that initially produced an invalid 100% resistance-detection result and how it was found and corrected.

## What this actually is

A dual-pathway deep learning framework: a transformer reads genomic sequence, a graph neural network reads protein structure, and the two are fused for organism classification and resistance prediction. It's benchmarked honestly against classical machine learning, single-modality baselines, and a real image-based classifier — because a number without a fair comparison isn't evidence of anything.

## Headline results

| Task | Result | Notes |
|---|---|---|
| Organism classification | **98.00%** accuracy | Real held-out test set (50 samples) |
| Resistance detection (honest) | **77.5–82.5%** | Whole-genome k-mer analysis; classical ML slightly ahead of neural |
| Resistance detection (invalid) | ~~100%~~ | Circular — see below |
| Missing-structure robustness | 98.00% → 90.00% | Graceful degradation |
| Sequencing-noise robustness | 98.00% → 94.00% | 15% token corruption |

Full breakdown, including per-evaluation status labels and the four-way resistance comparison: [`data/metadata/FINAL_REPORT.md`](data/metadata/FINAL_REPORT.md).

---

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/opemidimejiosatoyinbo/aspergillus-staphylococcus-diagnosis.git
cd aspergillus-staphylococcus-diagnosis
```

### 2. Set up the environment

```bash
conda create -n af-sa-diagnosis-env python=3.11
conda activate af-sa-diagnosis-env
conda install -c conda-forge ncbi-datasets-cli
conda install -y -c bioconda -c conda-forge ncbi-amrfinderplus
amrfinder -u
python3 -m pip install -r requirements.txt
```

### 3. Regenerate the real data

**This is the step every new clone of this repository needs, and it's deliberate.** Real genomic, structural, and image data — hundreds of megabytes — is *not* committed to this repository (see `.gitignore`). Instead, every byte of it is fully reproducible by running the pipeline scripts in order, against the same real public sources this project was built and validated on:

```bash
cd scripts/
python3 00_setup.py
python3 01_download_genomes.py          # real NCBI genome downloads
python3 02_download_structures.py       # real PDB/AlphaFold downloads
python3 03_download_images.py           # real DIBaS/OpenFungi image downloads
python3 04_call_resistance.py           # real AMRFinderPlus resistance calling
python3 05_build_ledger.py              # rebuilds the ground-truth ledger from what's on disk
python3 06_tokenize_genomes.py
python3 07_build_structural_graphs.py
python3 08_sanity_check.py              # confirms regenerated data is valid before proceeding
```

This takes real time — several hours on a standard machine, most of it genuine network I/O and whole-genome computation, not modeling. A fixed random seed (42) is used throughout, so the same *process* is followed on every run, though NCBI's available accession pool shifts slightly over time as new genomes are deposited — don't expect byte-identical results years later, only the same honest methodology.

### 4. Run the rest of the pipeline

```bash
python3 09_build_architecture.py
python3 10_test_wiring.py
python3 11_build_baselines.py
python3 12_build_classical_image_baselines.py
python3 13_execute_training.py
python3 13b_execute_resistance_blind.py     # run after 13 -- corrects a real, documented bug
python3 13c_execute_resistance_kmer_nn.py   # run after 13 -- the honest resistance number
python3 14_extract_interpretability.py
python3 15_robustness_stress_test.py
python3 16_generate_final_report.py
```

Or run the whole thing end-to-end with `python3 master_run.py`.

---

## Data and model hosting

Raw data and any future trained-model checkpoints are intentionally excluded from this Git repository — they're real, but too large for version control, and Git isn't built for tracking multi-hundred-megabyte binary files efficiently. Two paths forward as this project grows:

- **For now**: everything is regenerable from the scripts above, against the same cited public sources. This *is* the reproducibility story of this project.
- **If a persistent trained checkpoint or a fixed dataset snapshot is needed later** (e.g., for exact result reproduction without a multi-hour re-download, or for manuscript supplementary material), the recommended path is **Git LFS** for moderate file sizes, or external hosting via **Zenodo** (gives a permanent DOI — ideal for citing in the eventual manuscript), **OSF**, or an institutional repository for larger archives. Hugging Face or Kaggle are reasonable alternatives if the trained model itself becomes a shareable artifact.

## Data sources, all real and cited

- **Genomic**: NCBI GenBank/RefSeq
- **Structural**: RCSB Protein Data Bank; AlphaFold DB (predicted structures used only where no experimental structure exists)
- **Resistance calls**: NCBI AMRFinderPlus
- **Microscopy images**: DIBaS (Zieliński et al., 2017); OpenFungi (Cighir, Bolboacă & Lenard, 2025)

## Honest limitations, stated plainly

- Negative-control samples (genomic and image) combine two unrelated organisms under one label — a real simplification, not an oversight
- The S. aureus image test set is small (4 held-out images) — real, but not statistically robust on its own
- The "A. flavus" image class is labeled at the taxonomic section level (*Aspergillus* section Flavi), not confirmed at species level
- The model does not yet reliably signal "unknown" when shown a genuinely unfamiliar organism — a real, disclosed calibration gap
- Co-infection diagnosis was scoped out early, for lack of any real paired ground-truth data, and remains future work

## Repository structure

```
scripts/                          real, numbered pipeline scripts (00-16, plus 13b/13c)
master_run.py                     runs the full real pipeline end-to-end
requirements.txt                  pinned real dependencies
data/metadata/                    real ledger, resistance calls, all result CSVs (version-controlled)
data/metadata/FINAL_REPORT.md     the real, honest final summary
data/raw/, data/processed/,
data/images/, data/structural/    real downloaded/generated data (gitignored -- see above)
```

## Citing this work

A manuscript is in preparation. Until then, cite this repository directly, referencing the specific commit hash used for any reported result.
