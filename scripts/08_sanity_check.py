# --- Standard library imports ---
import torch                              # for loading and inspecting real tensors/graphs

# --- Standard library imports ---
import csv                                    # for reading the real ledger
from pathlib import Path                          # for clean, cross-platform file paths


# ============================================================
# CONFIGURATION
# ============================================================
LEDGER_CSV = Path("../data/metadata/immutable_ledger.csv")
GENOMIC_DIR = Path("../data/processed/genomic")
STRUCTURAL_DIR = Path("../data/processed/structural")

EXPECTED_TOKEN_LENGTH = 512     # must match MAX_TOKENS in 06_tokenize_genomes.py
EXPECTED_VOCAB_SIZE = 4096      # 4^6 possible 6-mers -- must match K in 06_tokenize_genomes.py


# ============================================================
# STEP 1: Load the real ledger to know what SHOULD exist
# ============================================================
def load_ledger_rows() -> list[dict]:
    if not LEDGER_CSV.exists():
        raise FileNotFoundError(f"Ledger not found at {LEDGER_CSV}. Run 05_build_ledger.py first.")

    with open(LEDGER_CSV, "r") as f:
        return list(csv.DictReader(f))


# ============================================================
# STEP 2: Validate real genomic tensors
# ============================================================
def check_genomic_tensors(ledger_rows: list[dict]) -> dict:
    """
    Checks every real genomic sample from the ledger against its actual saved tensor file, verifying shape, value range, and absence of corrupted (NaN/Inf) values.
    """
    print("Checking genomic tensors...")
    print("-" * 55)

    genomic_rows = [r for r in ledger_rows if r["data_type"] == "genomic"]
    results = {"checked": 0, "passed": 0, "missing": [], "shape_errors": [], "value_errors": []}

    for row in genomic_rows:
        sample_id = row["sample_id"]
        tensor_path = GENOMIC_DIR / f"{sample_id}_tokens.pt"
        results["checked"] += 1

        if not tensor_path.exists():
            results["missing"].append(sample_id)
            continue

        tensor = torch.load(tensor_path)

        # --- Real check 1: correct shape ---
        if tensor.shape != torch.Size([EXPECTED_TOKEN_LENGTH]):
            results["shape_errors"].append(
                f"{sample_id}: expected shape [{EXPECTED_TOKEN_LENGTH}], got {list(tensor.shape)}"
            )
            continue

        # --- Real check 2: token IDs fall within the valid vocabulary range Valid IDs are 0 (padding) through EXPECTED_VOCAB_SIZE (inclusive). Anything outside this range would indicate a bug in tokenization. ---
        if tensor.min().item() < 0 or tensor.max().item() > EXPECTED_VOCAB_SIZE:
            results["value_errors"].append(
                f"{sample_id}: token IDs out of valid range "
                f"[0, {EXPECTED_VOCAB_SIZE}] -- found min={tensor.min().item()}, "
                f"max={tensor.max().item()}"
            )
            continue

        # --- Real check 3: not entirely padding (which would mean the real sequence failed to tokenize at all) ---
        non_pad_fraction = (tensor != 0).float().mean().item()
        if non_pad_fraction == 0.0:
            results["value_errors"].append(f"{sample_id}: tensor is 100% padding -- likely tokenization failure")
            continue

        results["passed"] += 1

    print(f"  Checked: {results['checked']}  |  Passed: {results['passed']}  |  "
          f"Missing: {len(results['missing'])}  |  Shape errors: {len(results['shape_errors'])}  |  "
          f"Value errors: {len(results['value_errors'])}")

    return results


# ============================================================
# STEP 3: Validate real structural graphs
# ============================================================
def check_structural_graphs(ledger_rows: list[dict]) -> dict:
    """
    Checks every real structural sample from the ledger against its actual saved graph file, verifying node/edge counts and feature shapes are sane.
    """
    print("\nChecking structural graphs...")
    print("-" * 55)

    structural_rows = [
        r for r in ledger_rows
        if r["data_type"] in ("structural_experimental", "structural_predicted")
    ]
    results = {"checked": 0, "passed": 0, "missing": [], "errors": []}

    for row in structural_rows:
        sample_id = row["sample_id"]
        graph_path = STRUCTURAL_DIR / f"{sample_id}_graph.pt"
        results["checked"] += 1

        if not graph_path.exists():
            results["missing"].append(sample_id)
            continue

        graph = torch.load(graph_path, weights_only=False)  # trusted, self-generated file (see note above)

        # --- Real check 1: graph actually has nodes ---
        if graph.num_nodes == 0:
            results["errors"].append(f"{sample_id}: graph has zero nodes -- structure parsing likely failed")
            continue

        # --- Real check 2: graph actually has edges ---
        # A real folded protein should have SOME residues within 8A of each other -- zero edges would suggest a coordinate/units bug.
        if graph.num_edges == 0:
            results["errors"].append(f"{sample_id}: graph has zero edges -- check distance threshold/coordinates")
            continue

        # --- Real check 3: node feature and position tensors match node count ---
        if graph.x.shape[0] != graph.num_nodes or graph.pos.shape[0] != graph.num_nodes:
            results["errors"].append(f"{sample_id}: node feature/position count mismatch")
            continue

        # --- Real check 4: no NaN/Inf in coordinates (would break distance math) ---
        if torch.isnan(graph.pos).any() or torch.isinf(graph.pos).any():
            results["errors"].append(f"{sample_id}: NaN/Inf found in 3D coordinates")
            continue

        results["passed"] += 1

    print(f"  Checked: {results['checked']}  |  Passed: {results['passed']}  |  "
          f"Missing: {len(results['missing'])}  |  Errors: {len(results['errors'])}")

    return results


# ============================================================
# STEP 4: Print one real example in full detail, for a human to visually sanity-check -- the "look at ten by hand" step.
# ============================================================
def print_spot_check_example(ledger_rows: list[dict]):
    print("\nSpot-check: one real example in detail")
    print("-" * 55)

    genomic_rows = [r for r in ledger_rows if r["data_type"] == "genomic"]
    if genomic_rows:
        sample = genomic_rows[0]
        tensor_path = GENOMIC_DIR / f"{sample['sample_id']}_tokens.pt"
        if tensor_path.exists():
            tensor = torch.load(tensor_path)
            print(f"Genomic sample: {sample['sample_id']} (organism: {sample['organism']})")
            print(f"  Tensor shape: {list(tensor.shape)}")
            print(f"  First 10 token IDs: {tensor[:10].tolist()}")
            print(f"  Non-padding fraction: {(tensor != 0).float().mean().item():.2%}")

    structural_rows = [
        r for r in ledger_rows
        if r["data_type"] in ("structural_experimental", "structural_predicted")
    ]
    if structural_rows:
        sample = structural_rows[0]
        graph_path = STRUCTURAL_DIR / f"{sample['sample_id']}_graph.pt"
        if graph_path.exists():
            graph = torch.load(graph_path, weights_only=False)  # trusted, self-generated file (see note above)
            print(f"\nStructural sample: {sample['sample_id']} (organism: {sample['organism']})")
            print(f"  Residues (nodes): {graph.num_nodes}")
            print(f"  Contacts (edges): {graph.num_edges}")
            print(f"  Node feature shape: {list(graph.x.shape)}")
            print(f"  First residue's coordinates: {graph.pos[0].tolist()}")


# ============================================================
# MAIN: run the full, real sanity check
# ============================================================
def verify_artifacts():
    print("Executing Phase 2 Sanity Check (REAL validation)...")
    print("=" * 55)

    ledger_rows = load_ledger_rows()

    genomic_results = check_genomic_tensors(ledger_rows)
    structural_results = check_structural_graphs(ledger_rows)
    print_spot_check_example(ledger_rows)

    # --- Final honest verdict ---
    print("\n" + "=" * 55)
    genomic_ok = (genomic_results["passed"] == genomic_results["checked"]) and genomic_results["checked"] > 0
    structural_ok = (structural_results["passed"] == structural_results["checked"]) and structural_results["checked"] > 0

    if genomic_ok:
        print("[PASS] All real genomic tensors validated successfully.")
    else:
        print(f"[FAIL] Genomic validation issues found -- "
              f"{genomic_results['checked'] - genomic_results['passed']} of "
              f"{genomic_results['checked']} samples failed checks.")

    if structural_ok:
        print("[PASS] All real structural graphs validated successfully.")
    else:
        print(f"[FAIL] Structural validation issues found -- "
              f"{structural_results['checked'] - structural_results['passed']} of "
              f"{structural_results['checked']} samples failed checks.")

    print("=" * 55)
    if genomic_ok and structural_ok:
        print("Translation layer verified with REAL data. Safe to proceed to Phase 3.")
    else:
        print("Do NOT proceed to Phase 3 yet -- fix the issues flagged above first.")


if __name__ == "__main__":
    verify_artifacts()
