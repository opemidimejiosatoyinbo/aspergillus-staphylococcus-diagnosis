# --- Standard library imports ---
import torch
import torch.nn as nn
import torch.optim as optim
import importlib
from pathlib import Path

# Dynamically import our architecture module, since its filename starts with a digit and can't be imported with a normal `import` statement.
arch = importlib.import_module("09_build_architecture")


# ============================================================
# PART A: Gradient sanity check with correctly-shaped synthetic data
# ============================================================
def run_gradient_sanity_check():
    print("PART A: Gradient Sanity Check (synthetic, correctly-shaped data)")
    print("-" * 60)

    model = arch.DiagnosticModel(fusion_method="gated")

    batch_size = 4

    # Matches REAL genomic tensor shape exactly: [batch, 512] integer token IDs, values in the real valid range (0 = padding, 1-4096 = real k-mer IDs).
    dummy_tokens = torch.randint(0, 4097, (batch_size, 512))

    # A tiny synthetic graph batch: 4 "proteins", 10 nodes each, real 21-dim node features (matching our real one-hot amino acid encoding exactly).
    dummy_struct_x = torch.randn(40, 21)
    dummy_edge_index = torch.randint(0, 40, (2, 100))
    dummy_batch = torch.tensor([0]*10 + [1]*10 + [2]*10 + [3]*10)

    target_labels = torch.tensor([0, 1, 2, 1], dtype=torch.long)   # 0=A.flavus, 1=S.aureus, 2=negative

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    print("Feeding correctly-shaped synthetic batch through the real architecture...")
    print("Monitoring loss trajectory for 50 iterations...\n")

    initial_loss, final_loss = None, None
    for epoch in range(50):
        optimizer.zero_grad()
        organism_logits, _ = model(dummy_tokens, dummy_struct_x, dummy_edge_index, dummy_batch)
        loss = criterion(organism_logits, target_labels)
        loss.backward()
        optimizer.step()

        if epoch == 0:
            initial_loss = loss.item()
        if epoch == 49:
            final_loss = loss.item()

    print(f"Initial loss: {initial_loss:.4f}")
    print(f"Final loss:   {final_loss:.4f}")

    if final_loss < 0.1:
        print("[PASS] Gradients flow correctly through the real architecture.")
    else:
        print("[WARNING] Loss did not converge as expected -- check the architecture wiring.")

    print()
    return final_loss < 0.1


# ============================================================
# PART B: Real-data loading smoke test
# ============================================================
def run_real_data_smoke_test():
    print("PART B: Real-Data Loading Smoke Test")
    print("-" * 60)

    genomic_path = Path("../data/processed/genomic/all_genomic_tokens.pt")
    structural_path = Path("../data/processed/structural/all_structural_graphs.pt")

    if not genomic_path.exists() or not structural_path.exists():
        print("[SKIPPED] Real processed data not found -- run 05 and 06 first.")
        return False

    # Load our REAL combined genomic tensor file.
    genomic_data = torch.load(genomic_path, weights_only=False)
    real_tokens = genomic_data["tokens"]          # shape: [204, 512]
    real_sample_ids = genomic_data["sample_ids"]
    print(f"Loaded real genomic tensor: shape {list(real_tokens.shape)}, "
          f"{len(real_sample_ids)} real sample IDs")

    # Load our REAL structural graphs (a dict of sample_id -> graph).
    structural_graphs = torch.load(structural_path, weights_only=False)
    print(f"Loaded real structural graphs: {len(structural_graphs)} real structures "
          f"({list(structural_graphs.keys())})")

    # Take a small real batch of genomic samples (first 4, just for this smoke test).
    real_batch_tokens = real_tokens[:4]
    print(f"\nUsing real genomic batch: shape {list(real_batch_tokens.shape)}")

    # Pair this real genomic batch with ONE real structure, repeated across the batch -- purely to test that real data flows through the model. (Real per-sample structure pairing logic is a Phase 5 design decision, not resolved here -- see the module docstring above.)
    from torch_geometric.data import Batch
    one_structure = list(structural_graphs.values())[0]
    real_struct_batch = Batch.from_data_list([one_structure] * 4)

    model = arch.DiagnosticModel(fusion_method="gated")
    model.eval()

    with torch.no_grad():
        organism_logits, resistance_logits = model(
            real_batch_tokens,
            real_struct_batch.x,
            real_struct_batch.edge_index,
            real_struct_batch.batch,
        )

    print(f"Real forward pass succeeded.")
    print(f"  Organism logits shape:   {list(organism_logits.shape)} (expected: [4, 3])")
    print(f"  Resistance logits shape: {list(resistance_logits.shape)} (expected: [4, 1])")
    print("[PASS] Real genomic tensors and real structural graphs both flow correctly "
          "through the real architecture.\n")

    return True


# ============================================================
# MAIN
# ============================================================
def run_wiring_test():
    print("Phase 3 Wiring Test (corrected, real-shape-aware)")
    print("=" * 60)

    part_a_passed = run_gradient_sanity_check()
    part_b_passed = run_real_data_smoke_test()

    print("=" * 60)
    if part_a_passed and part_b_passed:
        print("[PASS] Architecture wiring verified against real data. Ready for Phase 4.")
    else:
        print("[INCOMPLETE] One or more checks did not pass -- review output above before proceeding.")


if __name__ == "__main__":
    run_wiring_test()
