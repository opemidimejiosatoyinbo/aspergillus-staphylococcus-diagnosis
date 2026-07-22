# --- Standard library imports ---
import csv
import itertools
import random
from pathlib import Path

import torch
from Bio import SeqIO


# ============================================================
# CONFIGURATION -- matches 06_tokenize_genomes.py exactly, so
# results are directly comparable
# ============================================================
LEDGER_CSV = Path("../data/metadata/immutable_ledger.csv")
OUT_PATH = Path("../data/processed/genomic/saureus_tokens_blind.pt")

K = 6
MAX_TOKENS = 512
PAD_TOKEN_ID = 0
WINDOW_SIZE = 2500   # matches the random-fallback window size used in 06_tokenize_genomes.py

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


def build_kmer_vocabulary(k: int) -> dict:
    bases = ["A", "T", "C", "G"]
    all_kmers = ["".join(combo) for combo in itertools.product(bases, repeat=k)]
    return {kmer: idx + 1 for idx, kmer in enumerate(all_kmers)}


def extract_random_window(fasta_path: Path, window_size: int = WINDOW_SIZE) -> str:
    """
    Identical logic to 06_tokenize_genomes.py's fallback path -- picks a real, length-weighted random contig and a random starting position, with NO knowledge of where any gene of interest sits.
    """
    records = list(SeqIO.parse(fasta_path, "fasta"))
    if not records:
        return ""

    weights = [len(r.seq) for r in records]
    chosen_record = random.choices(records, weights=weights, k=1)[0]
    seq = str(chosen_record.seq)

    if len(seq) <= window_size:
        return seq

    start_pos = random.randint(0, len(seq) - window_size)
    return seq[start_pos:start_pos + window_size]


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


def main():
    print("Building blind S. aureus tokenization (NO AMRFinder coordinates used)...")
    print("-" * 60)

    vocab = build_kmer_vocabulary(K)

    with open(LEDGER_CSV, "r") as f:
        rows = [
            r for r in csv.DictReader(f)
            if r["organism"] == "Staphylococcus_aureus"
            and r["resistance_status"] in ("RESISTANT", "SUSCEPTIBLE")
        ]

    print(f"Found {len(rows)} real S. aureus samples with confirmed resistance labels.\n")

    all_tokens, all_sample_ids, all_labels = [], [], []

    for i, row in enumerate(rows, start=1):
        fasta_path = Path(row["file_path"])
        if not fasta_path.exists():
            continue

        # Blind, every time -- no branching on resistance_status at all.
        region_sequence = extract_random_window(fasta_path)
        if not region_sequence:
            continue

        token_tensor = tokenize_sequence(region_sequence, vocab, K, MAX_TOKENS)

        all_tokens.append(token_tensor)
        all_sample_ids.append(row["sample_id"])
        all_labels.append(1 if row["resistance_status"] == "RESISTANT" else 0)

        if i % 25 == 0 or i == len(rows):
            print(f"  Blind-tokenized {i}/{len(rows)} samples...")

    combined = {
        "tokens": torch.stack(all_tokens),
        "sample_ids": all_sample_ids,
        "labels": torch.tensor(all_labels, dtype=torch.long),
    }
    torch.save(combined, OUT_PATH)

    n_resistant = sum(all_labels)
    print(f"\nBlind dataset built: {len(all_labels)} samples "
          f"({n_resistant} resistant, {len(all_labels) - n_resistant} susceptible)")
    print(f"Saved to: {OUT_PATH}")
    print("\nEvery sample here was tokenized from a RANDOM genomic window --")
    print("no AMRFinder coordinates, no knowledge of where mecA is, for ANY sample.")


if __name__ == "__main__":
    main()
