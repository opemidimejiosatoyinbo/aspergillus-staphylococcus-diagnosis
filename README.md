# AF-SA Diagnostic Pipeline

A real, reproducible, multi-modal diagnostic pipeline for *Aspergillus flavus* and *Staphylococcus aureus* identification and resistance profiling — built from genomic sequence, protein structure, and microscopy imagery.

Every script in this repository runs on real, publicly sourced data. Nothing here is simulated. Where a result turned out to be flawed during development, that's documented rather than hidden — see `FINAL_REPORT.md` and the roadmap for the full account.

## What this actually is

A dual-pathway deep learning framework: a transformer reads genomic sequence, a graph neural network reads protein structure, and the two are fused for organism classification and resistance prediction. It's benchmarked honestly against classical machine learning, single-modality baselines, and an image-based classifier — because a number without a fair comparison isn't evidence of anything.

## Headline results (see `data/metadata/FINAL_REPORT.md` for the full picture)

- **Organism classification**: 98% accuracy, real held-out test set
- **Resistance detection**: the honest number is 77.5–82.5%, achieved by whole-genome k-mer analysis (classical ML slightly ahead of neural). An earlier 100% result was found to be circular — traced to a preprocessing bug — and is documented, not reported, as a real finding
- **Robustness**: graceful degradation under missing structural input (98% → 90%) and sequencing noise (98% → 94%); appropriately low confidence, but imperfect calibration, on genuinely unseen organisms

## Environment setup

```bash
conda create -n af-sa-diagnosis-env python=3.11
conda activate af-sa-diagnosis-env
conda install -c conda-forge ncbi-datasets-cli
conda install -y -c bioconda -c conda-forge ncbi-amrfinderplus
amrfinder -u
python3 -m pip install biopython requests pandas torch torch_geometric \
    scikit-learn xgboost torchvision pillow remotezip scipy
```

## Running the pipeline

Scripts are numbered in execution order. Run them individually, in sequence — each one depends on real output from the one before it:

```bash
cd scripts/
python3 00_setup.py
python3 01_download_genomes.py
python3 02_download_structures.py
python3 03_download_images.py
python3 04_call_resistance.py
python3 05_build_ledger.py
python3 06_tokenize_genomes.py
python3 07_build_structural_graphs.py
python3 08_sanity_check.py
python3 09_build_architecture.py
python3 10_test_wiring.py
python3 11_build_baselines.py
python3 12_build_classical_image_baselines.py
python3 13_execute_training.py
python3 13b_execute_resistance_blind.py
python3 13c_execute_resistance_kmer_nn.py
python3 14_extract_interpretability.py
python3 15_robustness_stress_test.py
python3 16_generate_final_report.py
```

Scripts `13b` and `13c` exist specifically to honestly re-evaluate resistance detection after the circular result in `13` was identified — run them after `13`, before trusting any resistance number.

A full run, start to finish, takes several hours on a standard CPU-only machine — most of that time is real genome downloading and real k-mer frequency computation across hundreds of full genomes, not training.

## Data sources, all real and cited

- **Genomic**: NCBI GenBank/RefSeq
- **Structural**: RCSB Protein Data Bank; AlphaFold DB (predicted structures where no experimental structure exists)
- **Resistance calls**: NCBI AMRFinderPlus
- **Microscopy images**: DIBaS (Zieliński et al., 2017); OpenFungi (Cighir et al., 2025)

## Honest limitations, stated plainly

- Negative-control samples (in both genomic and image modalities) combine two unrelated organisms under one label — a real simplification, not an oversight
- The S. aureus image test set is small (4 held-out images) — real, but not statistically robust on its own
- The "A. flavus" image class is labeled at the taxonomic section level ("*Aspergillus* section Flavi"), not confirmed at the species level
- The model does not yet reliably signal "unknown" when shown a genuinely unfamiliar organism — a real, disclosed calibration gap
- Co-infection diagnosis was explicitly scoped out early, for lack of any real paired ground-truth data, and remains future work

## Repository structure

```
scripts/          real, numbered pipeline scripts (00-16)
data/raw/         real downloaded genomes, structures, images
data/processed/   real tokenized tensors and structural graphs
data/metadata/    real ledger, resistance calls, and all result CSVs
data/metadata/FINAL_REPORT.md   the real, honest final summary
```
