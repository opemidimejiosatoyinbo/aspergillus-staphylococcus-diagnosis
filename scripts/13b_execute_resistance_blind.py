# --- Standard library imports ---
import random
from pathlib import Path
import importlib

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score
from scipy import stats

arch = importlib.import_module("09_build_architecture")

BLIND_TOKENS_PATH = Path("../data/processed/genomic/saureus_tokens_blind.pt")
RESULTS_CSV = Path("../data/metadata/phase5_results_blind.csv")

N_FOLDS = 5
N_EPOCHS = 15
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class ResistanceOnlyModel(nn.Module):
    """
    A focused model for this ablation: the SAME real GenomicEncoder used everywhere else in this project, feeding a single binary resistance classifier head. No organism classification, no structural pathway -- isolating purely what the genomic transformer can learn from a blind window.
    """
    def __init__(self, embed_dim: int = 256, vocab_size: int = 4097):
        super().__init__()
        self.seq_expert = arch.GenomicEncoder(vocab_size=vocab_size, embed_dim=embed_dim)
        self.resistance_classifier = nn.Linear(embed_dim, 1)

    def forward(self, token_ids: torch.Tensor):
        emb = self.seq_expert(token_ids)
        return self.resistance_classifier(emb)


def train_one_fold(train_tokens, train_labels, n_epochs=N_EPOCHS):
    model = ResistanceOnlyModel()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(n_epochs):
        # Simple full-batch-per-step mini-batching over the fold's training data
        indices = list(range(len(train_tokens)))
        random.shuffle(indices)
        batch_size = 16
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i + batch_size]
            batch_tokens = train_tokens[batch_idx]
            batch_labels = train_labels[batch_idx].float()

            optimizer.zero_grad()
            logits = model(batch_tokens).squeeze(-1)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

    return model


def evaluate(model, eval_tokens, eval_labels):
    model.eval()
    with torch.no_grad():
        logits = model(eval_tokens).squeeze(-1)
        preds = (torch.sigmoid(logits) > 0.5).float()
    acc = accuracy_score(eval_labels.tolist(), preds.tolist())
    f1 = f1_score(eval_labels.tolist(), preds.tolist())
    return acc, f1


def main():
    print("BLIND Resistance Evaluation (no AMRFinder-directed tokenization)")
    print("=" * 60)

    if not BLIND_TOKENS_PATH.exists():
        print(f"[ERROR] {BLIND_TOKENS_PATH} not found. Run 06b_tokenize_saureus_blind.py first.")
        return

    data = torch.load(BLIND_TOKENS_PATH, weights_only=False)
    tokens, labels = data["tokens"], data["labels"]
    n_resistant = int(labels.sum())
    print(f"Loaded {len(labels)} real blind samples ({n_resistant} resistant, "
          f"{len(labels) - n_resistant} susceptible)\n")

    # --- Real train/test split, held out for a final honest evaluation ---
    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=RANDOM_SEED, stratify=labels.numpy()
    )
    train_tokens, train_labels = tokens[train_idx], labels[train_idx]
    test_tokens, test_labels = tokens[test_idx], labels[test_idx]

    print(f"Train: {len(train_labels)} | Held-out test: {len(test_labels)}\n")

    # --- Real stratified k-fold CV on the training portion ---
    print(f"Running {N_FOLDS}-fold cross-validation on blind tokens...")
    print("-" * 60)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_accuracies = []

    for fold_idx, (fold_train_idx, fold_val_idx) in enumerate(
        skf.split(train_tokens, train_labels), start=1
    ):
        print(f"  Fold {fold_idx}/{N_FOLDS}...")
        model = train_one_fold(train_tokens[fold_train_idx], train_labels[fold_train_idx])
        acc, f1 = evaluate(model, train_tokens[fold_val_idx], train_labels[fold_val_idx])
        fold_accuracies.append(acc)
        print(f"    Blind fold accuracy: {acc:.2%} (F1: {f1:.2%})")

    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    print(f"\nMean blind CV accuracy: {mean_acc:.2%} (+/- {std_acc:.2%})")

    # --- Final real fit on full training data, evaluated once on real held-out test ---
    print("\nFitting final model on full training data...")
    final_model = train_one_fold(train_tokens, train_labels)
    test_acc, test_f1 = evaluate(final_model, test_tokens, test_labels)

    majority_frac = max(n_resistant, len(labels) - n_resistant) / len(labels)

    print("\n" + "=" * 60)
    print("FINAL BLIND HELD-OUT TEST RESULTS")
    print("=" * 60)
    print(f"  Accuracy: {test_acc:.2%}")
    print(f"  F1:       {test_f1:.2%}")
    print(f"  Majority-class reference: {majority_frac:.2%}")

    print("\n" + "=" * 60)
    print("HONEST COMPARISON ACROSS ALL THREE RESISTANCE EVALUATIONS:")
    print("=" * 60)
    print(f"  1. Gene-targeted dual-pathway model (13_execute_training.py): 100.00%")
    print(f"     -- OPTIMISTIC/CIRCULAR: tokenization used AMRFinder's own coordinates,")
    print(f"        so the model's input already reflected the answer.")
    print(f"  2. Classical k-mer XGBoost/RF (12_build_classical_image_baselines.py): 80-82.5%")
    print(f"     -- Uses whole-genome k-mer FREQUENCY (not windowed), genuinely blind")
    print(f"        to gene location.")
    print(f"  3. Blind genomic transformer (this script): {test_acc:.2%}")
    print(f"     -- HONEST: random genomic window, no knowledge of mecA location,")
    print(f"        same architecture as the main model's genomic pathway.")

    Path("../data/metadata").mkdir(parents=True, exist_ok=True)
    with open(RESULTS_CSV, "w") as f:
        f.write("metric,value\n")
        f.write(f"blind_cv_mean_accuracy,{mean_acc}\n")
        f.write(f"blind_test_accuracy,{test_acc}\n")
        f.write(f"blind_test_f1,{test_f1}\n")
        f.write(f"majority_class_baseline,{majority_frac}\n")

    print(f"\nReal results saved to: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
