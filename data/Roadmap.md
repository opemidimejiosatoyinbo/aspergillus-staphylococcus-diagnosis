# Project Roadmap — What Actually Happened

This isn't the plan we started with. It's the honest record of what got built, what broke, what got fixed, and what we learned along the way — because that record turned out to matter more than the original proposal did.

## Phase 0 — Scoping the problem

The project began wider than it ended up. Co-infection detection — diagnosing *A. flavus* and *S. aureus* simultaneously — was part of the original ambition, until it became clear that no public dataset actually contains confirmed dual-infection genomic samples, and building one synthetically would mean training the whole project on an assumption we couldn't verify. So we narrowed the scope: single-organism diagnosis, done rigorously, with co-infection explicitly left as future work rather than quietly faked. That decision, made early, set the tone for everything after it. Better a smaller claim we could stand behind than a bigger one resting on synthetic ground.

## Phase 1 — Data acquisition

Getting real data turned out to be its own project. Early genome downloads for *A. flavus* returned exactly one usable sample — a filtering mistake, restricting to RefSeq only, when the real available pool was 337 assemblies once GenBank was included. Large downloads kept failing partway through on a flaky connection, which meant building real retry logic and batch validation rather than trusting a single big transfer to succeed. Structural data came from the Protein Data Bank and AlphaFold — six solved structures for *S. aureus* proteins, two AlphaFold-predicted structures for the fungal side. Image data came later, sourced from DIBaS and OpenFungi, two real public microscopy datasets, after a decision to actually pursue the image-based comparison rather than skip it.

## Phase 2 — Preprocessing, and the first real correction

The first version of our genomic tokenizer read only the opening ~500 bases of every genome — a bug, not a feature, that meant the model would never see *mecA* or *aflR* unless they happened to sit at the very start of an assembly. The fix was to locate genes by their real annotated coordinates and tokenize around them specifically, falling back to a random genomic window only when no target gene existed in that sample — which is itself a real, correct outcome for susceptible strains and negative controls, not a failure.

## Phase 3 — Architecture

The dual-pathway model took shape here: a small transformer over a custom k-mer vocabulary for genomic sequence, and a graph neural network over real, 8-Ångström residue-contact graphs for protein structure. Both pathways were built to match the actual shape of our real data, not a generic placeholder — a distinction that mattered more than it sounds, since an earlier draft of the architecture had been sized for embeddings that didn't correspond to anything we'd actually produced.

## Phase 4 — Baselines

Four honest points of comparison: a sequence-only model, a structure-only model, a classical k-mer frequency model (XGBoost and Random Forest), and — eventually — a real image classifier fine-tuned on DIBaS and OpenFungi. The classical baseline would go on to matter far more than a baseline usually does.

## Phase 5 — Training, and the finding that changed the project

This is where the project's real turning point happened. The first full training run reported 100% resistance-detection accuracy — a number good enough to be suspicious. Tracing it back revealed the actual cause: our gene-targeted tokenization used AMRFinder's own detected coordinates to decide where to center a resistant sample's input window. The model wasn't learning to find resistance. It was confirming an answer the preprocessing pipeline had already computed.

Fixing that meant building a genuinely blind evaluation — every sample tokenized from a random window, with no knowledge of where any gene actually sat — and the honest result collapsed to chance, 57.5%, because a single small window has almost no odds of overlapping a 2-kilobase gene in a 2.8-million-base genome. That failure was informative: it explained *why* the classical whole-genome frequency baseline succeeded (80–82.5%) where a windowed approach couldn't. A follow-up neural network trained on that same whole-genome representation reached 77.5% — real, honest, and still slightly behind classical machine learning on identical input. That comparison, not the invalid 100%, is the project's real resistance-detection result.

## Phase 6 — Interpretability

Genomic saliency showed something worth reporting: a consistent handful of 6-mers recurring across independent resistant test samples, suggesting the model had found a generalizable compositional signature rather than memorizing noise. Structural saliency showed the opposite of what we expected — zero overlap with PBP2a's known catalytic and allosteric residues — an honest negative result, reported as found rather than adjusted to look better.

## Phase 7 — Robustness

Three real stress tests: withholding structural input (a real 8-point accuracy drop, graceful, not catastrophic), corrupting 15% of genomic tokens (a real 4-point drop), and testing against *Bacillus subtilis*, an organism the model had never seen anywhere in training. The model showed appropriately low confidence overall on that unfamiliar organism, but split almost evenly between its two known-positive classes rather than correctly leaning toward "unknown" — a real, disclosed limitation, not a hidden one.

## Where this leaves the project

Every number in the final report traces back to a script that actually ran, on real data, with every limitation stated plainly rather than smoothed over. The project's real contribution turned out to be narrower and more specific than the original proposal imagined — strong organism identification, a well-documented resistance-detection ceiling, and an honest account of where deep learning did and didn't earn its complexity. That's a smaller story than "the model works." It's also a truer one.
