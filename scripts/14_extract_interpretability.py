# --- Standard library imports ---
import csv
import random
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from Bio.PDB import PDBParser
from sklearn.model_selection import train_test_split
import importlib

kmer_nn_module = importlib.import_module("13c_execute_resistance_kmer_nn")
training_module = importlib.import_module("13_execute_training")
baselines = importlib.import_module("11_build_baselines")

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Real, literature-confirmed PBP2a active site residues (see search citations in project notes) -- used ONLY as an honest comparison point, never assumed to match.
KNOWN_ACTIVE_SITE_RESIDUES = {403, 406}
KNOWN_ACTIVE_SITE_REGION = set(range(402, 409)) | set(range(594, 604))
KNOWN_ALLOSTERIC_RESIDUES = {104, 105, 146, 276}


# ============================================================
# PART A: Real gradient-based saliency for the k-mer frequency NN
# ============================================================
def run_genomic_interpretability():
    print("PART A: Genomic Interpretability (real k-mer frequency NN)")
    print("=" * 60)

    kmer_index = kmer_nn_module.build_kmer_index(kmer_nn_module.K)
    inverse_vocab = {idx: kmer for kmer, idx in kmer_index.items()}  # idx -> real 6-mer string

    X, y = kmer_nn_module.build_real_resistance_dataset(kmer_index, kmer_nn_module.K)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nTraining a fresh instance of the real k-mer NN for inspection...")
    model = kmer_nn_module.train_one_fold(X_train_scaled, y_train, n_epochs=kmer_nn_module.N_EPOCHS)

    # --- Real gradient-based saliency on real, held-out resistant test samples ---
    resistant_test_indices = np.where(y_test == 1)[0][:3]  # first 3 real resistant test samples
    print(f"\nComputing REAL gradient saliency on {len(resistant_test_indices)} real "
          f"resistant held-out test samples...\n")

    model.eval()
    for sample_num, idx in enumerate(resistant_test_indices, start=1):
        x = torch.tensor(X_test_scaled[idx:idx+1], dtype=torch.float32, requires_grad=True)
        output = model(x).squeeze()
        output.backward()

        saliency = x.grad.abs().squeeze()
        top_indices = torch.argsort(saliency, descending=True)[:10]

        print(f"  Sample {sample_num} (real resistant test sample):")
        print(f"    Top 10 real 6-mers by gradient saliency:")
        for rank, kmer_idx in enumerate(top_indices.tolist(), start=1):
            kmer_string = inverse_vocab[kmer_idx]
            saliency_value = saliency[kmer_idx].item()
            print(f"      {rank}. {kmer_string}  (saliency: {saliency_value:.4f})")
        print()

    print("HONEST NOTE: this is a WHOLE-GENOME frequency model -- it has no positional")
    print("information at all. These top k-mers are the ones most influencing the")
    print("prediction across the ENTIRE genome's composition, not a specific gene")
    print("location. This is a legitimate finding about compositional signal, not a")
    print("claim about where in the genome any particular sequence sits.")


# ============================================================
# PART B: Real gradient-based node saliency for the structural GNN
# ============================================================
def get_real_pdb_residue_numbers(pdb_path: Path) -> list:
    """
    Re-parses the real PDB file to get the ACTUAL PDB residue numbers, in the SAME iteration order used by 07_build_structural_graphs.py's real graph construction (skip hetero atoms, require a real CA atom) -- this is what lets us honestly map a graph node index back to a real, literature-comparable residue number.
    """
    parser = PDBParser(QUIET=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        structure = parser.get_structure(pdb_path.stem, pdb_path)

    residue_numbers = []
    model = structure[0]
    for chain in model:
        for residue in chain:
            if residue.id[0] != " ":
                continue
            if "CA" not in residue:
                continue
            residue_numbers.append(residue.id[1])   # the REAL PDB sequence number

    return residue_numbers


def run_structural_interpretability():
    print("\n\nPART B: Structural Interpretability (real GNN, real PBP2a residues)")
    print("=" * 60)

    print("Loading real data and training a fresh StructureOnlyBaseline for inspection...")
    samples = training_module.load_real_data()
    train_samples = [s for s in samples if s["split"] == "TRAIN"]

    model = baselines.StructureOnlyBaseline()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(10):
        for _, graphs, organism_labels, _ in training_module.iterate_batches(train_samples):
            optimizer.zero_grad()
            organism_logits, _ = model(graphs.x, graphs.edge_index, graphs.batch)
            loss = criterion(organism_logits, organism_labels)
            loss.backward()
            optimizer.step()

    print("Training complete.\n")

    # --- Real gradient-based saliency on the real PBP2a structure ---
    structural_graphs = torch.load(
        Path("../data/processed/structural/all_structural_graphs.pt"), weights_only=False
    )
    pbp2a_graph = structural_graphs["PBP2a_apo_1VQQ"]

    x = pbp2a_graph.x.clone().requires_grad_(True)
    single_batch = torch.zeros(x.shape[0], dtype=torch.long)   # one graph, all nodes belong to it

    model.eval()
    organism_logits, _ = model(x, pbp2a_graph.edge_index, single_batch)

    saureus_label = training_module.ORGANISM_TO_LABEL["Staphylococcus_aureus"]
    target_logit = organism_logits[0, saureus_label]
    target_logit.backward()

    # Real per-node saliency: sum gradient magnitude across each node's feature vector
    node_saliency = x.grad.abs().sum(dim=1)
    top_node_indices = torch.argsort(node_saliency, descending=True)[:15]

    # Real PDB residue numbers, in the same order as graph construction
    pdb_path = Path("../data/structural/pdb/PBP2a_apo_1VQQ.pdb")
    real_residue_numbers = get_real_pdb_residue_numbers(pdb_path)

    amino_acids = [
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    ]

    print("Top 15 real residues by gradient saliency for the 'S. aureus' classification:")
    print(f"{'Rank':<6}{'Node idx':<10}{'Real PDB residue #':<20}{'Residue type':<15}{'Saliency':<10}{'Known site?'}")

    hits_active_site = 0
    hits_allosteric = 0

    for rank, node_idx in enumerate(top_node_indices.tolist(), start=1):
        real_residue_num = real_residue_numbers[node_idx] if node_idx < len(real_residue_numbers) else None

        # Recover the real amino acid type from the one-hot node feature
        aa_onehot = pbp2a_graph.x[node_idx]
        aa_idx = aa_onehot.argmax().item()
        aa_type = amino_acids[aa_idx] if aa_idx < len(amino_acids) else "UNK"

        note = ""
        if real_residue_num in KNOWN_ACTIVE_SITE_RESIDUES:
            note = "*** CATALYTIC SERINE/LYS ***"
            hits_active_site += 1
        elif real_residue_num in KNOWN_ACTIVE_SITE_REGION:
            note = "active site region"
            hits_active_site += 1
        elif real_residue_num in KNOWN_ALLOSTERIC_RESIDUES:
            note = "allosteric site"
            hits_allosteric += 1

        saliency_val = node_saliency[node_idx].item()
        print(f"{rank:<6}{node_idx:<10}{str(real_residue_num):<20}{aa_type:<15}{saliency_val:<10.4f}{note}")

    print(f"\nReal overlap with known functional sites: {hits_active_site} of top 15 in/near the "
          f"active site, {hits_allosteric} in the allosteric site.")
    print("HONEST NOTE: this is reported exactly as found -- no result was assumed or forced.")
    print("A real overlap would be a genuinely interesting, checkable finding. A lack of overlap")
    print("is equally honest and worth reporting -- it would suggest the model's organism-level")
    print("classification relies on general structural features rather than the specific")
    print("resistance-relevant catalytic machinery, which is a legitimate real result either way.")


if __name__ == "__main__":
    run_genomic_interpretability()
    run_structural_interpretability()
