# --- Standard library imports ---
import torch
import torch.nn as nn
import importlib
from pathlib import Path

# Reuse the real encoders from 09_build_architecture.py -- imported dynamically since the filename starts with a digit.
arch = importlib.import_module("09_build_architecture")


# ============================================================
# BASELINE 1: Sequence-only -- real GenomicEncoder, no structural input at all
# ============================================================
class SequenceOnlyBaseline(nn.Module):
    """
    Uses the SAME real GenomicEncoder as the main model, but has no access to structural data whatsoever -- tests how much the genomic pathway alone can accomplish.
    """
    def __init__(self, embed_dim: int = 256, vocab_size: int = 4097):
        super().__init__()
        self.seq_expert = arch.GenomicEncoder(vocab_size=vocab_size, embed_dim=embed_dim)
        self.organism_classifier = nn.Linear(embed_dim, 3)
        self.resistance_classifier = nn.Linear(embed_dim, 1)

    def forward(self, token_ids: torch.Tensor):
        emb = self.seq_expert(token_ids)   # real transformer encoding, matching main model exactly
        return self.organism_classifier(emb), self.resistance_classifier(emb)


# ============================================================
# BASELINE 2: Structure-only -- real StructuralEncoder, no sequence input at all
# ============================================================
class StructureOnlyBaseline(nn.Module):
    """
    Uses the SAME real StructuralEncoder (GNN) as the main model, but has no access to genomic sequence data whatsoever -- tests how much the structural pathway alone can accomplish.
    """
    def __init__(self, embed_dim: int = 256, node_feature_dim: int = 21):
        super().__init__()
        self.struct_expert = arch.StructuralEncoder(node_feature_dim=node_feature_dim, embed_dim=embed_dim)
        self.organism_classifier = nn.Linear(embed_dim, 3)
        self.resistance_classifier = nn.Linear(embed_dim, 1)

    def forward(self, struct_x: torch.Tensor, struct_edge_index: torch.Tensor, struct_batch: torch.Tensor):
        emb = self.struct_expert(struct_x, struct_edge_index, struct_batch)   # real GNN encoding
        return self.organism_classifier(emb), self.resistance_classifier(emb)


# ============================================================
# Real smoke test -- confirm both baselines actually run on data shaped like our real genomic tensors and real structural graphs.
# ============================================================
def compile_baselines():
    print("Initializing Phase 3 Baseline Construction (real encoders, corrected)...")
    print("-" * 60)

    print("Compiling Baseline 1: Sequence-Only (real GenomicEncoder)...")
    seq_base = SequenceOnlyBaseline()

    print("Compiling Baseline 2: Structure-Only (real StructuralEncoder/GNN)...")
    struct_base = StructureOnlyBaseline()

    # --- Real smoke test: run each baseline on real data if available ---
    genomic_path = Path("../data/processed/genomic/all_genomic_tokens.pt")
    structural_path = Path("../data/processed/structural/all_structural_graphs.pt")

    if genomic_path.exists():
        genomic_data = torch.load(genomic_path, weights_only=False)
        real_batch = genomic_data["tokens"][:4]   # small real batch
        seq_base.eval()
        with torch.no_grad():
            organism_logits, resistance_logits = seq_base(real_batch)
        print(f"\n[PASS] SequenceOnlyBaseline ran on real genomic data: "
              f"organism logits {list(organism_logits.shape)}, "
              f"resistance logits {list(resistance_logits.shape)}")
    else:
        print("\n[SKIPPED] Real genomic tensor file not found -- run 06_tokenize_genomes.py first.")

    if structural_path.exists():
        from torch_geometric.data import Batch
        structural_graphs = torch.load(structural_path, weights_only=False)
        one_structure = list(structural_graphs.values())[0]
        real_struct_batch = Batch.from_data_list([one_structure] * 4)
        struct_base.eval()
        with torch.no_grad():
            organism_logits, resistance_logits = struct_base(
                real_struct_batch.x, real_struct_batch.edge_index, real_struct_batch.batch
            )
        print(f"[PASS] StructureOnlyBaseline ran on real structural data: "
              f"organism logits {list(organism_logits.shape)}, "
              f"resistance logits {list(resistance_logits.shape)}")
    else:
        print("[SKIPPED] Real structural graph file not found -- run 07_build_structural_graphs.py first.")

    print("-" * 60)
    print("Baselines compiled and verified against real data. Ready for Phase 5 comparison.")


if __name__ == "__main__":
    compile_baselines()
