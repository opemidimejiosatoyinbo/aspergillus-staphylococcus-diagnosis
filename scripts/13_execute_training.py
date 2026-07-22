# --- Standard library imports ---
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Batch
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score
from scipy import stats
import importlib

arch = importlib.import_module("09_build_architecture")
baselines = importlib.import_module("11_build_baselines")


# ============================================================
# CONFIGURATION
# ============================================================
LEDGER_CSV = Path("../data/metadata/immutable_ledger.csv")
GENOMIC_TOKENS_PATH = Path("../data/processed/genomic/all_genomic_tokens.pt")
STRUCTURAL_GRAPHS_PATH = Path("../data/processed/structural/all_structural_graphs.pt")
RESULTS_CSV = Path("../data/metadata/phase5_results.csv")

N_FOLDS = 5
N_EPOCHS = 15
RANDOM_SEED = 42
ORGANISM_TO_LABEL = {"Aspergillus_flavus": 0, "Staphylococcus_aureus": 1, "negative_control": 2}

# FIXED: no longer conditioned on resistance_status for S. aureus -- see the corrected docstring above explaining why the old version leaked the label directly through the structural pathway.
STRUCTURE_PAIRING = {
    "Aspergillus_flavus": "AflR_Aflavus_P41765",
    "Staphylococcus_aureus": "PBP2a_apo_1VQQ",
    "negative_control": "alpha_hemolysin_7AHL",   # disclosed filler -- see docstring
}

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# STEP 1: Load real data and assign real structure pairings
# ============================================================
def load_real_data():
    with open(LEDGER_CSV, "r") as f:
        ledger_rows = [r for r in csv.DictReader(f) if r["data_type"] == "genomic"]

    genomic_data = torch.load(GENOMIC_TOKENS_PATH, weights_only=False)
    token_lookup = {sid: genomic_data["tokens"][i] for i, sid in enumerate(genomic_data["sample_ids"])}

    structural_graphs = torch.load(STRUCTURAL_GRAPHS_PATH, weights_only=False)

    samples = []
    for row in ledger_rows:
        sample_id = row["sample_id"]
        if sample_id not in token_lookup:
            continue   # ledger references a file that wasn't tokenized -- skip honestly

        organism = row["organism"]
        resistance_status = row["resistance_status"] if row["resistance_status"] in ("RESISTANT", "SUSCEPTIBLE") else None

        # FIXED: structure lookup keyed purely by organism now -- resistance status is no longer part of this decision (see corrected STRUCTURE_PAIRING and docstring above for why the old version leaked the label).
        structure_name = STRUCTURE_PAIRING.get(organism)
        if structure_name is None or structure_name not in structural_graphs:
            continue

        samples.append({
            "sample_id": sample_id,
            "tokens": token_lookup[sample_id],
            "structure": structural_graphs[structure_name],
            "organism_label": ORGANISM_TO_LABEL[organism],
            "resistance_label": 1 if resistance_status == "RESISTANT" else (0 if resistance_status == "SUSCEPTIBLE" else -1),
            "split": row["split"],
        })

    return samples


# ============================================================
# STEP 2: Real batching -- combines token tensors and real variable-size graphs into one training batch
# ============================================================
def make_batch(sample_list: list):
    tokens = torch.stack([s["tokens"] for s in sample_list])
    graphs = Batch.from_data_list([s["structure"] for s in sample_list])
    organism_labels = torch.tensor([s["organism_label"] for s in sample_list], dtype=torch.long)
    resistance_labels = torch.tensor([s["resistance_label"] for s in sample_list], dtype=torch.float32)
    return tokens, graphs, organism_labels, resistance_labels


def iterate_batches(samples: list, batch_size: int = 16, shuffle: bool = True):
    indices = list(range(len(samples)))
    if shuffle:
        random.shuffle(indices)
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i + batch_size]
        yield make_batch([samples[j] for j in batch_indices])


# ============================================================
# STEP 3: Real training loop for the dual-pathway model
# ============================================================
def train_dual_pathway_model(train_samples: list, fusion_method: str = "gated", n_epochs: int = N_EPOCHS):
    model = arch.DiagnosticModel(fusion_method=fusion_method)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    organism_criterion = nn.CrossEntropyLoss()
    resistance_criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for tokens, graphs, organism_labels, resistance_labels in iterate_batches(train_samples):
            optimizer.zero_grad()

            organism_logits, resistance_logits = model(
                tokens, graphs.x, graphs.edge_index, graphs.batch
            )

            loss = organism_criterion(organism_logits, organism_labels)

            # Resistance loss ONLY computed on real S. aureus samples with a
            # known real label (-1 marks "not applicable" -- A. flavus/negative).
            valid_resistance_mask = resistance_labels >= 0
            if valid_resistance_mask.any():
                loss += resistance_criterion(
                    resistance_logits.squeeze(-1)[valid_resistance_mask],
                    resistance_labels[valid_resistance_mask],
                )

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 5 == 0 or epoch == n_epochs - 1:
            print(f"    Epoch {epoch + 1}/{n_epochs}: loss = {epoch_loss:.4f}")

    return model


def evaluate_dual_pathway_model(model, eval_samples: list):
    model.eval()
    all_organism_preds, all_organism_true = [], []
    all_resistance_preds, all_resistance_true = [], []

    with torch.no_grad():
        for tokens, graphs, organism_labels, resistance_labels in iterate_batches(eval_samples, shuffle=False):
            organism_logits, resistance_logits = model(tokens, graphs.x, graphs.edge_index, graphs.batch)

            all_organism_preds.extend(organism_logits.argmax(dim=1).tolist())
            all_organism_true.extend(organism_labels.tolist())

            valid_mask = resistance_labels >= 0
            if valid_mask.any():
                preds = (torch.sigmoid(resistance_logits.squeeze(-1)) > 0.5).float()
                all_resistance_preds.extend(preds[valid_mask].tolist())
                all_resistance_true.extend(resistance_labels[valid_mask].tolist())

    organism_acc = accuracy_score(all_organism_true, all_organism_preds)
    organism_f1 = f1_score(all_organism_true, all_organism_preds, average="macro")

    if all_resistance_true:
        resistance_acc = accuracy_score(all_resistance_true, all_resistance_preds)
        resistance_f1 = f1_score(all_resistance_true, all_resistance_preds)
    else:
        resistance_acc, resistance_f1 = None, None

    return {
        "organism_accuracy": organism_acc,
        "organism_macro_f1": organism_f1,
        "resistance_accuracy": resistance_acc,
        "resistance_f1": resistance_f1,
    }


# ============================================================
# STEP 4: Real training/evaluation for single-modality baselines
# ============================================================
def train_sequence_only_baseline(train_samples: list, n_epochs: int = N_EPOCHS):
    model = baselines.SequenceOnlyBaseline()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(n_epochs):
        for tokens, _, organism_labels, _ in iterate_batches(train_samples):
            optimizer.zero_grad()
            organism_logits, _ = model(tokens)
            loss = criterion(organism_logits, organism_labels)
            loss.backward()
            optimizer.step()

    return model


def evaluate_sequence_only_baseline(model, eval_samples: list):
    model.eval()
    preds, true = [], []
    with torch.no_grad():
        for tokens, _, organism_labels, _ in iterate_batches(eval_samples, shuffle=False):
            organism_logits, _ = model(tokens)
            preds.extend(organism_logits.argmax(dim=1).tolist())
            true.extend(organism_labels.tolist())
    return accuracy_score(true, preds), f1_score(true, preds, average="macro")


def train_structure_only_baseline(train_samples: list, n_epochs: int = N_EPOCHS):
    model = baselines.StructureOnlyBaseline()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(n_epochs):
        for _, graphs, organism_labels, _ in iterate_batches(train_samples):
            optimizer.zero_grad()
            organism_logits, _ = model(graphs.x, graphs.edge_index, graphs.batch)
            loss = criterion(organism_logits, organism_labels)
            loss.backward()
            optimizer.step()

    return model


def evaluate_structure_only_baseline(model, eval_samples: list):
    model.eval()
    preds, true = [], []
    with torch.no_grad():
        for _, graphs, organism_labels, _ in iterate_batches(eval_samples, shuffle=False):
            organism_logits, _ = model(graphs.x, graphs.edge_index, graphs.batch)
            preds.extend(organism_logits.argmax(dim=1).tolist())
            true.extend(organism_labels.tolist())
    return accuracy_score(true, preds), f1_score(true, preds, average="macro")


# ============================================================
# STEP 5: Real stratified k-fold cross-validation on TRAIN split
# ============================================================
def run_cross_validation(train_samples: list, fusion_method: str = "gated"):
    labels = [s["organism_label"] for s in train_samples]
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    fold_results_dual, fold_results_seq, fold_results_struct = [], [], []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_samples, labels), start=1):
        print(f"\n  Fold {fold_idx}/{N_FOLDS}...")
        fold_train = [train_samples[i] for i in train_idx]
        fold_val = [train_samples[i] for i in val_idx]

        print(f"    Training dual-pathway model ({fusion_method} fusion)...")
        dual_model = train_dual_pathway_model(fold_train, fusion_method=fusion_method)
        dual_metrics = evaluate_dual_pathway_model(dual_model, fold_val)
        fold_results_dual.append(dual_metrics["organism_accuracy"])
        print(f"    Dual-pathway fold accuracy: {dual_metrics['organism_accuracy']:.2%}")

        print(f"    Training sequence-only baseline...")
        seq_model = train_sequence_only_baseline(fold_train)
        seq_acc, _ = evaluate_sequence_only_baseline(seq_model, fold_val)
        fold_results_seq.append(seq_acc)
        print(f"    Sequence-only fold accuracy: {seq_acc:.2%}")

        print(f"    Training structure-only baseline...")
        struct_model = train_structure_only_baseline(fold_train)
        struct_acc, _ = evaluate_structure_only_baseline(struct_model, fold_val)
        fold_results_struct.append(struct_acc)
        print(f"    Structure-only fold accuracy: {struct_acc:.2%}")

    return fold_results_dual, fold_results_seq, fold_results_struct


# ============================================================
# STEP 6: Real paired statistical testing
# ============================================================
def run_statistical_test(dual_scores: list, baseline_scores: list, baseline_name: str):
    if len(dual_scores) < 2:
        print(f"  [SKIPPED] Not enough folds for a real statistical test against {baseline_name}.")
        return None

    t_stat, p_value = stats.ttest_rel(dual_scores, baseline_scores)
    mean_diff = np.mean(dual_scores) - np.mean(baseline_scores)
    print(f"  Dual-pathway vs. {baseline_name}: mean diff = {mean_diff:+.2%}, "
          f"paired t-test p = {p_value:.4f}")
    return p_value


# ============================================================
# MAIN
# ============================================================
def main():
    print("Phase 5: REAL Training and Evaluation")
    print("=" * 60)

    print("\nLoading real data and applying structure pairing...")
    samples = load_real_data()
    print(f"Loaded {len(samples)} real samples with valid structure pairings.")

    train_samples = [s for s in samples if s["split"] == "TRAIN"]
    test_samples = [s for s in samples if s["split"] == "TEST"]
    print(f"  Train: {len(train_samples)} | Held-out test: {len(test_samples)}")

    # --- Fusion ablation via real cross-validation ---
    print("\n" + "=" * 60)
    print("ABLATION: Fusion method comparison (gated vs. concat)")
    print("=" * 60)

    results = {}
    for fusion_method in ["gated", "concat"]:
        print(f"\nRunning {N_FOLDS}-fold CV with '{fusion_method}' fusion...")
        dual_scores, seq_scores, struct_scores = run_cross_validation(train_samples, fusion_method)
        results[fusion_method] = {
            "dual": dual_scores, "seq": seq_scores, "struct": struct_scores,
        }
        print(f"\n  Mean CV accuracy ({fusion_method}): {np.mean(dual_scores):.2%} "
              f"(+/- {np.std(dual_scores):.2%})")

    best_fusion = max(results, key=lambda k: np.mean(results[k]["dual"]))
    print(f"\n[RESULT] Best fusion method by real CV accuracy: '{best_fusion}'")

    # --- Real statistical comparison against baselines (best fusion method) ---
    print("\n" + "=" * 60)
    print(f"Statistical comparison: dual-pathway ({best_fusion}) vs. baselines")
    print("=" * 60)
    run_statistical_test(results[best_fusion]["dual"], results[best_fusion]["seq"], "sequence-only")
    run_statistical_test(results[best_fusion]["dual"], results[best_fusion]["struct"], "structure-only")

    # --- Final real fit on full TRAIN, evaluated ONCE on real held-out TEST ---
    print("\n" + "=" * 60)
    print(f"FINAL MODEL: fitting on full real TRAIN split ({best_fusion} fusion)...")
    print("=" * 60)
    final_model = train_dual_pathway_model(train_samples, fusion_method=best_fusion, n_epochs=N_EPOCHS)
    final_metrics = evaluate_dual_pathway_model(final_model, test_samples)

    print("\nFINAL REAL HELD-OUT TEST RESULTS:")
    print(f"  Organism accuracy:    {final_metrics['organism_accuracy']:.2%}")
    print(f"  Organism macro-F1:    {final_metrics['organism_macro_f1']:.2%}")
    if final_metrics["resistance_accuracy"] is not None:
        print(f"  Resistance accuracy:  {final_metrics['resistance_accuracy']:.2%}")
        print(f"  Resistance F1:        {final_metrics['resistance_f1']:.2%}")

    # --- Save real results to disk ---
    Path("../data/metadata").mkdir(parents=True, exist_ok=True)
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["best_fusion_method", best_fusion])
        writer.writerow(["test_organism_accuracy", final_metrics["organism_accuracy"]])
        writer.writerow(["test_organism_macro_f1", final_metrics["organism_macro_f1"]])
        writer.writerow(["test_resistance_accuracy", final_metrics["resistance_accuracy"]])
        writer.writerow(["test_resistance_f1", final_metrics["resistance_f1"]])

    print(f"\nReal results saved to: {RESULTS_CSV}")
    print("\nHONEST LIMITATION REMINDER: negative_control samples were paired with")
    print("alpha-hemolysin as a disclosed filler (no true negative structure exists")
    print("in our dataset). Proper missing-modality handling is deferred to")
    print("15_robustness_stress_test.py.")


if __name__ == "__main__":
    main()
