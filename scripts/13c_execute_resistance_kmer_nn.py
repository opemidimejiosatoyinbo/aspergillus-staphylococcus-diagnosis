# --- Standard library imports ---
import csv
import itertools
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from Bio import SeqIO
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler


LEDGER_CSV = Path("../data/metadata/immutable_ledger.csv")
RESULTS_CSV = Path("../data/metadata/phase5_results_kmer_nn.csv")

K = 6
N_FOLDS = 5
N_EPOCHS = 30
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# STEP 1: Real whole-genome k-mer frequency vectors -- identical representation to the classical baseline in script 12, so the comparison isolates model architecture as the only variable.
# ============================================================
def build_kmer_index(k: int) -> dict:
    bases = ["A", "T", "C", "G"]
    all_kmers = ["".join(combo) for combo in itertools.product(bases, repeat=k)]
    return {kmer: idx for idx, kmer in enumerate(all_kmers)}


def compute_kmer_frequency_vector(fasta_path: Path, kmer_index: dict, k: int) -> np.ndarray:
    counts = np.zeros(len(kmer_index), dtype=np.float64)
    for record in SeqIO.parse(fasta_path, "fasta"):
        seq = str(record.seq).upper()
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i + k]
            if kmer in kmer_index:
                counts[kmer_index[kmer]] += 1
    total = counts.sum()
    if total > 0:
        counts = counts / total
    return counts


def build_real_resistance_dataset(kmer_index: dict, k: int):
    with open(LEDGER_CSV, "r") as f:
        rows = [
            r for r in csv.DictReader(f)
            if r["organism"] == "Staphylococcus_aureus"
            and r["resistance_status"] in ("RESISTANT", "SUSCEPTIBLE")
        ]

    X, y = [], []
    print(f"Computing real whole-genome k-mer frequency vectors for {len(rows)} samples...")
    for i, row in enumerate(rows, start=1):
        fasta_path = Path(row["file_path"])
        if not fasta_path.exists():
            continue
        vector = compute_kmer_frequency_vector(fasta_path, kmer_index, k)
        X.append(vector)
        y.append(1 if row["resistance_status"] == "RESISTANT" else 0)
        if i % 25 == 0 or i == len(rows):
            print(f"  Processed {i}/{len(rows)} samples...")

    return np.array(X), np.array(y)


# ============================================================
# STEP 2: A real, deliberately regularized neural network -- not the windowed transformer, a proper MLP over frequency vectors
# ============================================================
class KmerFrequencyMLP(nn.Module):
    """
    A small feedforward network over the real 4096-dim k-mer frequency vector. Dropout and a narrowing architecture are real, deliberate safeguards against overfitting, given how few real training samples exist relative to the feature count.
    """
    def __init__(self, input_dim: int = 4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_one_fold(X_train, y_train, n_epochs=N_EPOCHS):
    model = KmerFrequencyMLP(input_dim=X_train.shape[1])
    # L2 weight decay -- real, explicit regularization given the
    # dimensionality-vs-sample-size imbalance noted in the docstring.
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)

    model.train()
    for epoch in range(n_epochs):
        indices = list(range(len(X_tensor)))
        random.shuffle(indices)
        batch_size = 16
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i + batch_size]
            optimizer.zero_grad()
            logits = model(X_tensor[batch_idx]).squeeze(-1)
            loss = criterion(logits, y_tensor[batch_idx])
            loss.backward()
            optimizer.step()

    return model


def evaluate(model, X_eval, y_eval):
    model.eval()
    X_tensor = torch.tensor(X_eval, dtype=torch.float32)
    with torch.no_grad():
        logits = model(X_tensor).squeeze(-1)
        preds = (torch.sigmoid(logits) > 0.5).float().numpy()
    acc = accuracy_score(y_eval, preds)
    f1 = f1_score(y_eval, preds)
    return acc, f1


# ============================================================
# MAIN
# ============================================================
def main():
    print("Whole-Genome k-mer Frequency Neural Network (real, non-circular)")
    print("=" * 60)

    kmer_index = build_kmer_index(K)
    X, y = build_real_resistance_dataset(kmer_index, K)
    n_resistant = int(y.sum())
    print(f"\nReal dataset built: {len(y)} samples ({n_resistant} resistant, "
          f"{len(y) - n_resistant} susceptible)\n")

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    # Real feature standardization -- fit ONLY on training data, applied
    # to both splits, avoiding any leakage of test-set statistics.
    scaler = StandardScaler()
    X_train_full_scaled = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test)

    print(f"Train: {len(y_train_full)} | Held-out test: {len(y_test)}\n")

    print(f"Running {N_FOLDS}-fold cross-validation...")
    print("-" * 60)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_accuracies = []

    for fold_idx, (fold_train_idx, fold_val_idx) in enumerate(
        skf.split(X_train_full_scaled, y_train_full), start=1
    ):
        print(f"  Fold {fold_idx}/{N_FOLDS}...")
        model = train_one_fold(X_train_full_scaled[fold_train_idx], y_train_full[fold_train_idx])
        acc, f1 = evaluate(model, X_train_full_scaled[fold_val_idx], y_train_full[fold_val_idx])
        fold_accuracies.append(acc)
        print(f"    Fold accuracy: {acc:.2%} (F1: {f1:.2%})")

    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    print(f"\nMean CV accuracy: {mean_acc:.2%} (+/- {std_acc:.2%})")

    print("\nFitting final model on full training data...")
    final_model = train_one_fold(X_train_full_scaled, y_train_full)
    test_acc, test_f1 = evaluate(final_model, X_test_scaled, y_test)

    majority_frac = max(n_resistant, len(y) - n_resistant) / len(y)

    print("\n" + "=" * 60)
    print("FINAL K-MER NEURAL NETWORK HELD-OUT TEST RESULTS")
    print("=" * 60)
    print(f"  Accuracy: {test_acc:.2%}")
    print(f"  F1:       {test_f1:.2%}")
    print(f"  Majority-class reference: {majority_frac:.2%}")

    print("\n" + "=" * 60)
    print("FULL, HONEST COMPARISON -- ALL FOUR RESISTANCE EVALUATIONS:")
    print("=" * 60)
    print("  1. Gene-targeted dual-pathway model:        100.00%  [INVALID -- circular,")
    print("     see 13_execute_training.py docstring for why this number is untrustworthy]")
    print("  2. Classical k-mer XGBoost/RF:               80.00-82.50%  [real, honest,")
    print("     whole-genome frequency, tree-based model]")
    print("  3. Blind windowed transformer:               57.50%  [collapsed to chance --")
    print("     single window too small to reliably contain mecA]")
    print(f"  4. Whole-genome k-mer neural network (this): {test_acc:.2%}  [real, honest,")
    print("     whole-genome frequency, neural network -- same representation as #2,")
    print("     different model class]")
    print("\nCONCLUSION: compare #4 directly against #2 -- this isolates whether a neural")
    print("network adds value over classical ML given IDENTICAL, legitimate input.")

    Path("../data/metadata").mkdir(parents=True, exist_ok=True)
    with open(RESULTS_CSV, "w") as f:
        f.write("metric,value\n")
        f.write(f"kmer_nn_cv_mean_accuracy,{mean_acc}\n")
        f.write(f"kmer_nn_test_accuracy,{test_acc}\n")
        f.write(f"kmer_nn_test_f1,{test_f1}\n")
        f.write(f"majority_class_baseline,{majority_frac}\n")

    print(f"\nReal results saved to: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
