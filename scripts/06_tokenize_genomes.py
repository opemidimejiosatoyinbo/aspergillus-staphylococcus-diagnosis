# --- Standard library imports ---
from Bio import SeqIO          # for reading real DNA sequences from FASTA files
import torch                     # for building and saving real numeric tensors

# --- Standard library imports ---
import csv                          # for reading the real ledger CSV
import itertools                       # used to generate the full k-mer vocabulary
import random                             # for random fallback window selection
from pathlib import Path                     # for clean, cross-platform file paths


# ============================================================
# CONFIGURATION
# ============================================================
LEDGER_CSV = Path("../data/metadata/immutable_ledger.csv")
OUT_DIR = Path("../data/processed/genomic")

K = 6                     # k-mer length
MAX_TOKENS = 512            # tokens kept per sample
PAD_TOKEN_ID = 0

# How much real sequence to grab around a located gene, on each side, in bases -- generous enough to comfortably cover MAX_TOKENS worth of k-mers even after the gene itself, while staying focused on the actual region of biological interest rather than the whole genome.
FLANK_BASES = 1000

# Keywords searched (case-insensitive) in each real GFF3 record's attribute field, to actually locate our genes of interest by name.
GENE_SEARCH_KEYWORDS = {
    "Aspergillus_flavus": ["aflr", "aflatoxin", "afld"],
    "Staphylococcus_aureus": ["meca", "mecc"],
}

RANDOM_SEED = 42   # reproducible random fallback window selection
random.seed(RANDOM_SEED)


# ============================================================
# STEP 1: Build the k-mer vocabulary (unchanged from before)
# ============================================================
def build_kmer_vocabulary(k: int) -> dict:
    bases = ["A", "T", "C", "G"]
    all_kmers = ["".join(combo) for combo in itertools.product(bases, repeat=k)]
    return {kmer: idx + 1 for idx, kmer in enumerate(all_kmers)}


# ============================================================
# STEP 2: Read the real ledger
# ============================================================
def load_genomic_samples_from_ledger() -> list[dict]:
    if not LEDGER_CSV.exists():
        raise FileNotFoundError(f"Ledger not found at {LEDGER_CSV}. Run 05_build_ledger.py first.")

    with open(LEDGER_CSV, "r") as f:
        reader = csv.DictReader(f)
        return [row for row in reader if row["data_type"] == "genomic"]


# ============================================================
# STEP 3a: For S. aureus, get REAL gene coordinates from AMRFinder's own output -- not from a separate GFF3 keyword search. 
# This matters: we confirmed empirically that NCBI's automated annotation sometimes labels a gene "mecA" by loose sequence homology even when it's NOT a real, functional resistance determinant (AMRFinder's own curated, identity- scored analysis is what correctly told us a given sample is SUSCEPTIBLE, even though its GFF3 still names a "mecA" gene).
# Using AMRFinder's own coordinates keeps our gene-targeting consistent with the same authoritative source we already trust for the resistance LABEL itself.
# ============================================================
AMR_RESULTS_DIR = Path("../data/metadata/amrfinder_results")


def get_amr_gene_coordinates(sample_id: str, keywords: list[str]) -> tuple | None:
    """
    Reads this sample's REAL AMRFinder output (.tsv) and returns the (contig_id, start, end) of the first row whose Element symbol matches one of our target keywords -- i.e. a gene AMRFinder itself validated as a real resistance determinant, not just a name match.

    Returns None if AMRFinder found no such gene in this sample -- which is the correct, expected outcome for genuinely susceptible samples.
    """
    amr_file = AMR_RESULTS_DIR / f"{sample_id}_amr.tsv"
    if not amr_file.exists():
        return None

    import csv as csv_module   # local import to avoid confusion with the module-level csv import
    with open(amr_file, "r") as f:
        reader = csv_module.DictReader(f, delimiter="\t")
        for row in reader:
            symbol = row.get("Element symbol", "").lower()
            if any(symbol.startswith(kw) for kw in keywords):
                contig_id = row.get("Contig id")
                start = int(row.get("Start"))
                end = int(row.get("Stop"))
                return (contig_id, min(start, end), max(start, end))   # GFF/AMR coords can be given either strand order

    return None   # genuinely no validated resistance gene in this sample


# ============================================================
# STEP 3b: For A. flavus, search the real GFF3 by gene name -- no AMRFinder equivalent exists for fungal biosynthesis genes, so annotation-based name matching is the best available source.
# ============================================================
def find_gff_path_for_sample(fasta_path: Path) -> Path | None:
    gff_path = fasta_path.parent / "genomic.gff"
    return gff_path if gff_path.exists() else None


def search_gff_for_target_gene(gff_path: Path, keywords: list[str]) -> tuple | None:
    with open(gff_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue

            fields = line.strip().split("\t")
            if len(fields) < 9:
                continue

            contig_id, source, feature_type, start, end = fields[0], fields[1], fields[2], fields[3], fields[4]
            attributes = fields[8]

            if feature_type != "gene":
                continue

            attributes_lower = attributes.lower()
            for keyword in keywords:
                if keyword in attributes_lower:
                    return (contig_id, int(start), int(end))

    return None


# ============================================================
# STEP 4: Extract real sequence around a located gene
# ============================================================
def extract_targeted_region(fasta_path: Path, contig_id: str, gene_start: int, gene_end: int) -> str:
    """
    Reads the real FASTA file, finds the specific contig/chromosome the target gene sits on, and extracts real sequence spanning the gene plus FLANK_BASES on either side -- giving the tokenizer actual, biologically relevant sequence to work with.
    """
    for record in SeqIO.parse(fasta_path, "fasta"):
        if record.id == contig_id:
            seq = str(record.seq)
            region_start = max(0, gene_start - FLANK_BASES)
            region_end = min(len(seq), gene_end + FLANK_BASES)
            return seq[region_start:region_end]

    return ""   # contig ID from the GFF3 didn't match any FASTA record -- shouldn't normally happen


# ============================================================
# STEP 5: Fallback -- extract a random window when no target gene is found (expected for negative controls / susceptible samples)
# ============================================================
def extract_random_window(fasta_path: Path, window_size: int = 2500) -> str:
    """
    Picks a random contig and a random starting position within it, and extracts a window of real sequence. Used when no target gene could be located -- still far more representative of the genome as a whole than always reading from position 0.
    """
    records = list(SeqIO.parse(fasta_path, "fasta"))
    if not records:
        return ""

    # Weight contig selection by length, so we don't over-sample tiny plasmid fragments relative to the main chromosome.
    weights = [len(r.seq) for r in records]
    chosen_record = random.choices(records, weights=weights, k=1)[0]
    seq = str(chosen_record.seq)

    if len(seq) <= window_size:
        return seq   # whole contig is shorter than our window -- just use all of it

    start_pos = random.randint(0, len(seq) - window_size)
    return seq[start_pos:start_pos + window_size]


# ============================================================
# STEP 6: Tokenize a real sequence into k-mer IDs (unchanged logic)
# ============================================================
def tokenize_sequence(dna_sequence: str, vocab: dict, k: int, max_tokens: int) -> torch.Tensor:
    dna_sequence = dna_sequence.upper()
    token_ids = []

    for i in range(len(dna_sequence) - k + 1):
        kmer = dna_sequence[i:i + k]
        if kmer in vocab:
            token_ids.append(vocab[kmer])
        if len(token_ids) >= max_tokens:
            break

    if len(token_ids) < max_tokens:
        token_ids += [PAD_TOKEN_ID] * (max_tokens - len(token_ids))

    return torch.tensor(token_ids, dtype=torch.long)


# ============================================================
# STEP 7: Process every real sample, gene-targeted where possible
# ============================================================
def process_all_samples(samples: list[dict], vocab: dict):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_tensors, all_sample_ids, all_labels, all_region_sources = [], [], [], []
    failed_samples = []
    gene_targeted_count = 0
    random_fallback_count = 0

    for sample in samples:
        sample_id = sample["sample_id"]
        organism = sample["organism"]
        fasta_path = Path(sample["file_path"])

        if not fasta_path.exists():
            print(f"  [WARNING] {fasta_path} not found. Skipping.")
            failed_samples.append(sample_id)
            continue

        region_sequence = ""
        region_source = "random_fallback"

        # --- Route to the correct, authoritative source per organism ---
        keywords = GENE_SEARCH_KEYWORDS.get(organism)
        if keywords:
            gene_hit = None

            if organism == "Staphylococcus_aureus":
                # Use AMRFinder's own validated coordinates -- confirmed via real testing that GFF3 gene NAMES alone are not reliable here (some genomes have a gene annotated "mecA" that AMRFinder itself does not count as a real resistance determinant).
                gene_hit = get_amr_gene_coordinates(sample_id, keywords)

            elif organism == "Aspergillus_flavus":
                # No AMRFinder equivalent for fungal biosynthesis genes -- GFF3 annotation search is the best available source here.
                gff_path = find_gff_path_for_sample(fasta_path)
                if gff_path:
                    gene_hit = search_gff_for_target_gene(gff_path, keywords)

            if gene_hit:
                contig_id, gene_start, gene_end = gene_hit
                region_sequence = extract_targeted_region(fasta_path, contig_id, gene_start, gene_end)
                if region_sequence:
                    region_source = "gene_targeted"

        # --- Fall back to a random window if no gene was found/extracted ---
        if not region_sequence:
            region_sequence = extract_random_window(fasta_path)

        if not region_sequence:
            print(f"  [WARNING] Could not extract any sequence for {sample_id}. Skipping.")
            failed_samples.append(sample_id)
            continue

        token_tensor = tokenize_sequence(region_sequence, vocab, K, MAX_TOKENS)

        individual_path = OUT_DIR / f"{sample_id}_tokens.pt"
        torch.save(token_tensor, individual_path)

        all_tensors.append(token_tensor)
        all_sample_ids.append(sample_id)
        all_labels.append(int(sample["label_positive"]) if sample["label_positive"] != "NA" else -1)
        all_region_sources.append(region_source)

        if region_source == "gene_targeted":
            gene_targeted_count += 1
        else:
            random_fallback_count += 1

        print(f"  {sample_id} ({organism}): {region_source}, "
              f"{len(region_sequence):,} bases extracted -> {MAX_TOKENS} tokens")

    if all_tensors:
        combined = {
            "tokens": torch.stack(all_tensors),
            "sample_ids": all_sample_ids,
            "labels": torch.tensor(all_labels, dtype=torch.long),
            "region_sources": all_region_sources,   # kept for later interpretability work
        }
        combined_path = OUT_DIR / "all_genomic_tokens.pt"
        torch.save(combined, combined_path)
        print(f"\nCombined tensor saved: {combined_path} (shape: {combined['tokens'].shape})")

    print(f"\nGene-targeted extractions: {gene_targeted_count}")
    print(f"Random-window fallbacks:    {random_fallback_count}")

    return len(all_tensors), failed_samples


# ============================================================
# MAIN
# ============================================================
def process_sequences():
    print("Building k-mer vocabulary (k={})...".format(K))
    vocab = build_kmer_vocabulary(K)
    print(f"Vocabulary built: {len(vocab)} possible {K}-mers.\n")

    print("Loading real genomic samples from the ledger...")
    samples = load_genomic_samples_from_ledger()
    print(f"Found {len(samples)} real genomic sample(s) to tokenize.\n")

    if len(samples) == 0:
        print("[ERROR] No genomic samples found. Run 01_download_genomes.py and 05_build_ledger.py first.")
        return

    print("-" * 55)
    success_count, failed_samples = process_all_samples(samples, vocab)
    print("-" * 55)

    print(f"Tokenization complete: {success_count}/{len(samples)} samples processed successfully.")
    if failed_samples:
        print(f"Failed/skipped samples: {failed_samples}")
    print(f"\nReal, gene-aware tensors saved under {OUT_DIR}/")


if __name__ == "__main__":
    process_sequences()
