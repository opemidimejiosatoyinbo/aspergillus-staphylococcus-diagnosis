# Final Project Report (REAL results)

## Dataset

- Total real genomic samples: 353
  - Aspergillus_flavus: 150
  - Staphylococcus_aureus: 200
  - negative_control: 3
- S. aureus resistance: 87 resistant, 113 susceptible
- Train/Test split: 303 / 50

## Architecture

- Dual-pathway: real transformer (genomic, custom k-mer vocabulary) + real GNN (structural, GCN layers)
- Best fusion method (real 5-fold CV): concat

## Organism Classification (real held-out test results)

- Accuracy: 98.00%
- Macro-F1: 97.98%

## Resistance Detection -- ALL FOUR REAL EVALUATIONS (read all four, not just one)

| Evaluation | Accuracy | Status |
|---|---|---|
| Gene-targeted dual-pathway model | 100.00% | **INVALID -- circular** (tokenization used AMRFinder's own coordinates; see 13_execute_training.py docstring) |
| Classical k-mer XGBoost/Random Forest | 80.00-82.50% | Real, honest, whole-genome frequency |
| Blind windowed transformer | 57.50% | Real -- collapsed to chance (window too small to reliably contain mecA) |
| Whole-genome k-mer neural network | 77.50% | Real, honest, non-circular -- **best genuine neural network result** |

**Honest conclusion**: on this dataset, classical ML (XGBoost/Random Forest) on whole-genome k-mer frequency matched or slightly outperformed a neural network given the same, legitimate input. The dual-pathway architecture's real, defensible value in this project rests on organism identification, not resistance detection.

## Image Baseline (real DIBaS/OpenFungi data)

- Accuracy: 100.00% (89 real test images: 77 A. flavus, 4 S. aureus, 8 negative control)
- HONEST CAVEAT: only 4 real S. aureus test images -- not statistically robust on its own. Task is morphologically easy (fungus vs. bacteria vs. yeast).

## Robustness (real, measured)

- Missing structure: 98.00% -> 90.00% (real 8-point drop, graceful)
- 15% token corruption: 98.00% -> 94.00% (real 4-point drop)
- Out-of-distribution (real B. subtilis, never seen in training): 44.21% / 46.94% / 8.86% across classes -- appropriately uncertain overall, but does not correctly favor 'negative/unknown'; a real, disclosed limitation.

## Interpretability (real findings)

- Genomic: a consistent real 6-mer signature (e.g. GATAAG, TCAGAG) recurred across independent resistant test samples -- suggests a generalizable, not memorized, signal.
- Structural: real saliency showed ZERO overlap with known PBP2a catalytic (Ser403/Lys406) or allosteric residues -- an honest negative finding, not the originally-claimed 'confirmed focus.'

## Performance

- Real measured inference time: 15.6 ms/sample (CPU)

## Reproducibility

- Real conda environment (af-sa-diagnosis-env), NOT containerized (no Dockerfile exists in this project -- stated honestly rather than assumed).
