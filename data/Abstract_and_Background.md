# A Unified Multi-Modal Deep Learning Framework for the Molecular Diagnosis of *Aspergillus flavus* and *Staphylococcus aureus* Infections

**Osatoyinbo O. O.¹,²,⁴, Adenipekun O. A.³, Adejokun A. I.³**

¹Faculty of Life Sciences, Department of Microbiology, University of Ilorin, Ilorin, Nigeria
²Faculty of Natural Sciences, Department of Biological Sciences, Redeemer's University, Ede, Nigeria
³Faculty of Computing and Digital Technologies, Department of Computer Science, Redeemer's University, Ede, Nigeria
⁴SAO Biosciences, Department of Microbiology, Obafemi Awolowo University, Ile-Ife, Nigeria

*Corresponding Author: oosatoyinbo@unilorin.edu.ng | +234 813 340 4158

---

## Abstract

Diagnosing *Aspergillus flavus* and *Staphylococcus aureus* still leans heavily on microscopy, a method built to read shape rather than identity, and one that struggles the moment a specimen is degraded or simply ambiguous. This study set out to replace that reliance with something more direct: a dual-pathway deep learning framework that diagnoses both organisms from their molecular evidence alone, fusing a transformer trained on real genomic sequence with a graph neural network trained on real protein structure.

Built and validated entirely on real, public data — 353 genomes, eight solved and predicted protein structures, and real microscopy images — the framework identifies organism identity with 98% accuracy on a genuinely held-out test set. Methicillin resistance in *Staphylococcus aureus* proved harder, and here the more interesting finding emerged. An early evaluation appeared to reach perfect accuracy, but closer scrutiny showed the pipeline had quietly handed the model its own answer during preprocessing. Once corrected, a classical k-mer frequency model reached 80–82%, modestly outperforming every neural alternative tested, including the dual-pathway architecture itself.

That result matters more than a clean number would have. It shows precisely where deep learning earns its complexity — organism identification — and where a simpler method still wins. Robustness testing confirmed graceful degradation under missing structural data and sequencing noise, though the model does not yet recognize truly unfamiliar organisms with appropriate uncertainty. Together, these findings offer a molecular alternative to image-based diagnosis, grounded in real, reproducible evidence rather than an optimistic average.

**Keywords:** Deep Learning; *Aspergillus flavus*; *Staphylococcus aureus*; Graph Neural Networks; Transformers; Molecular Profiling; Methicillin Resistance; Reproducibility

---

## Background

Walk into most clinical microbiology labs today, and the first line of defense against a suspected *A. flavus* or *S. aureus* infection is still a microscope. A technician stains the sample, looks, and makes a call. It's fast, it's cheap, and it's been the standard for a very long time. But it has a real weakness, one that gets more serious the sicker the patient is: morphology lies. A stressed fungal colony can look nothing like its textbook photo. Bacteria left too long at the wrong temperature distort. The read depends on preparation, on the technician's eye, on conditions that have nothing to do with what the organism actually is at a molecular level.

That gap is what this project set out to close — not by making microscopy better, but by asking whether it's even necessary. DNA doesn't degrade the way a stain does. A protein's fold doesn't shift because the slide sat too long. If a diagnostic tool could read those two things directly — sequence and structure — it would be answering the identity question at its source, rather than inferring it from an image of a downstream effect.

So the plan was straightforward, at least on paper: build one architecture with two ways of seeing. A transformer to read genomic sequence, tuned to recognize the specific genes that actually matter clinically — *mecA* for methicillin resistance in *S. aureus*, *aflR* for the aflatoxin pathway in *A. flavus*. And alongside it, a graph neural network to read the physical shape of the relevant proteins, since a resistance mechanism is, in the end, a physical object with a specific geometry, not just a line of code in a genome.

What follows is an honest account of what actually happened when that plan met real data. Some of it worked. Some of it revealed problems we didn't anticipate — including one, buried in how we prepared the resistance-detection data, that would have quietly invalidated the project's headline result if it had gone unchecked. Finding that, and fixing it, turned out to matter more than any single accuracy number this project produced.

## Objective

To build, train, and rigorously validate a dual-pathway deep learning framework — combining a genomic transformer and a structural graph neural network — capable of diagnosing *A. flavus* and *S. aureus* from real molecular data alone, and to report every result honestly, including the ones that complicate the original hypothesis.

---
