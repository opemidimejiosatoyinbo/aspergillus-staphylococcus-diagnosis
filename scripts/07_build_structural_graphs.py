# --- Standard library imports ---
from Bio.PDB import PDBParser              # for reading real 3D coordinates from .pdb files
import torch                                    # for building real tensors
from torch_geometric.data import Data              # the standard graph data format for GNNs

# --- Standard library imports ---
import csv                                             # for reading the real ledger CSV
import warnings                                            # to silence expected, harmless PDB parsing warnings
from pathlib import Path                                       # for clean, cross-platform file paths


# ============================================================
# CONFIGURATION
# ============================================================
LEDGER_CSV = Path("../data/metadata/immutable_ledger.csv")
OUT_DIR = Path("../data/processed/structural")

DISTANCE_THRESHOLD_ANGSTROMS = 8.0   # standard residue-contact cutoff, matches our Phase 2 writeup

# The 20 standard amino acids, used to build a fixed one-hot encoding.
# Any residue type not in this list (e.g. a modified residue) gets encoded as "unknown" rather than crashing the script.
AMINO_ACIDS = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
NUM_AA_TYPES = len(AMINO_ACIDS) + 1   # +1 for "unknown" residue types


# ============================================================
# STEP 1: Read the real ledger and get structural samples
# ============================================================
def load_structural_samples_from_ledger() -> list[dict]:
    """
    Reads the real ledger CSV and returns only rows corresponding to real structural samples (both experimental PDB and AlphaFold-predicted).
    """
    if not LEDGER_CSV.exists():
        raise FileNotFoundError(
            f"Ledger not found at {LEDGER_CSV}. Run 05_build_ledger.py first."
        )

    samples = []
    with open(LEDGER_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["data_type"] in ("structural_experimental", "structural_predicted"):
                samples.append(row)

    return samples


# ============================================================
# STEP 2: Extract real C-alpha coordinates and residue types from one real PDB file.
# ============================================================
def extract_residues(pdb_path: Path) -> tuple[list[str], torch.Tensor]:
    """
    Parses a real .pdb file and extracts, for every standard amino acid
    residue found:
        - its 3-letter residue type (e.g. "ALA", "GLY")
        - the 3D coordinates of its C-alpha (C-alpha) atom -- the atom conventionally used to represent a residue's overall position when building a structural graph like this one.

    Returns:
        residue_types  -- list of 3-letter residue codes, one per residue
        coordinates    -- a real torch.Tensor of shape [num_residues, 3]
    """
    parser = PDBParser(QUIET=True)   # QUIET suppresses routine, harmless format warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # PDB files often trigger benign Biopython warnings
        structure = parser.get_structure(pdb_path.stem, pdb_path)

    residue_types = []
    coordinates = []

    # A PDB structure can contain multiple models/chains -- we take the first model, and iterate over every chain within it, since our targets here are single-chain or simply-structured proteins.
    model = structure[0]
    for chain in model:
        for residue in chain:
            # Skip water molecules, ligands, and other non-amino-acid entries -- we only want real protein backbone residues.
            if residue.id[0] != " ":
                continue
            if "CA" not in residue:
                continue   # skip any residue missing its C-alpha atom

            residue_types.append(residue.get_resname())
            ca_coord = residue["CA"].get_coord()   # real (x, y, z) coordinates, in Angstroms
            coordinates.append(ca_coord)

    coordinates_tensor = torch.tensor(coordinates, dtype=torch.float32)
    return residue_types, coordinates_tensor


# ============================================================
# STEP 3: Build real node features (one-hot amino acid type)
# ============================================================
def build_node_features(residue_types: list[str]) -> torch.Tensor:
    """
    Converts each residue's 3-letter type into a real one-hot encoded feature vector -- e.g. "ALA" becomes a vector with a 1 in the position assigned to alanine, and 0s everywhere else.
    """
    num_residues = len(residue_types)
    features = torch.zeros((num_residues, NUM_AA_TYPES), dtype=torch.float32)

    for i, res_type in enumerate(residue_types):
        idx = AA_TO_INDEX.get(res_type, NUM_AA_TYPES - 1)   # unknown types go in the last slot
        features[i, idx] = 1.0

    return features


# ============================================================
# STEP 4: Build real edges based on real 3D distance
# ============================================================
def build_edges(coordinates: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Computes the real pairwise distance between every pair of residues' C-alpha atoms, and creates an edge between any two residues whose distance falls within the threshold -- this is the actual "8 Angstrom contact" rule described in our Phase 2 writeup.

    Returns edge_index in PyTorch Geometric's expected format: a [2, num_edges] tensor, where each column is one (source, target) pair of connected residue indices.
    """
    # torch.cdist computes the REAL Euclidean distance between every pair of points in 3D space -- this is not an approximation.
    distance_matrix = torch.cdist(coordinates, coordinates)   # shape: [num_residues, num_residues]

    # Find every pair (i, j) where distance is within threshold, excluding a residue being "connected" to itself (i == j).
    within_threshold = (distance_matrix <= threshold) & (distance_matrix > 0)
    edge_indices = within_threshold.nonzero(as_tuple=False)   # shape: [num_edges, 2]

    # PyTorch Geometric expects edges as a [2, num_edges] tensor, so we transpose from [num_edges, 2].
    edge_index = edge_indices.t().contiguous()
    return edge_index


# ============================================================
# STEP 5: Process one real structure end-to-end
# ============================================================
def build_graph_for_structure(pdb_path: Path) -> Data:
    """
    Runs the full real pipeline for one structure: parse -> extract residues -> build node features -> build edges -> package as a PyTorch Geometric Data object.
    """
    residue_types, coordinates = extract_residues(pdb_path)

    if len(residue_types) == 0:
        raise ValueError(f"No valid amino acid residues found in {pdb_path}")

    node_features = build_node_features(residue_types)
    edge_index = build_edges(coordinates, DISTANCE_THRESHOLD_ANGSTROMS)

    graph = Data(
        x=node_features,        # [num_residues, NUM_AA_TYPES] -- real one-hot residue types
        edge_index=edge_index,   # [2, num_edges] -- real 8-Angstrom contact edges
        pos=coordinates,          # [num_residues, 3] -- real 3D coordinates, kept for reference/visualization
    )
    return graph


# ============================================================
# MAIN: run the full, real graph construction process
# ============================================================
def construct_graphs():
    print("Loading real structural samples from the ledger...")
    samples = load_structural_samples_from_ledger()
    print(f"Found {len(samples)} real structural sample(s) to process.\n")

    if len(samples) == 0:
        print("[ERROR] No structural samples found in the ledger. Nothing to build.")
        print("        Run 02_download_structures.py and 05_build_ledger.py first.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("-" * 55)
    success_count = 0
    failed_samples = []
    all_graphs = {}

    for sample in samples:
        sample_id = sample["sample_id"]
        pdb_path = Path(sample["file_path"])

        if not pdb_path.exists():
            print(f"  [WARNING] Ledger references {pdb_path}, but it was not found. Skipping.")
            failed_samples.append(sample_id)
            continue

        try:
            graph = build_graph_for_structure(pdb_path)
        except ValueError as e:
            print(f"  [WARNING] {sample_id}: {e}")
            failed_samples.append(sample_id)
            continue

        # --- Save this sample's graph individually ---
        out_path = OUT_DIR / f"{sample_id}_graph.pt"
        torch.save(graph, out_path)
        all_graphs[sample_id] = graph

        print(f"  Built graph for {sample_id}: "
              f"{graph.num_nodes} residues, {graph.num_edges} edges")
        success_count += 1

    # --- Save one combined file for convenient loading later ---
    if all_graphs:
        combined_path = OUT_DIR / "all_structural_graphs.pt"
        torch.save(all_graphs, combined_path)
        print(f"\nCombined graph dictionary saved: {combined_path}")

    print("-" * 55)
    print(f"Graph construction complete: {success_count}/{len(samples)} structures processed.")
    if failed_samples:
        print(f"Failed/skipped samples: {failed_samples}")
    print(f"\nReal graphs saved under {OUT_DIR}/")


if __name__ == "__main__":
    construct_graphs()
