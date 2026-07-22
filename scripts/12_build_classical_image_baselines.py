# --- Standard library imports ---
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import csv
import itertools
import numpy as np
from pathlib import Path
from Bio import SeqIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import torch.nn as nn


# ============================================================
# CONFIGURATION
# ============================================================
LEDGER_CSV = Path("../data/metadata/immutable_ledger.csv")
K = 6   # matches the k-mer length used throughout this project

ORGANISM_TO_LABEL = {
    "Aspergillus_flavus": 0,
    "Staphylococcus_aureus": 1,
    "negative_control": 2,
}


# ============================================================
# STEP 1: Build real k-mer FREQUENCY vectors (not token sequences)
# ============================================================
def build_kmer_index(k: int) -> dict:
    """Same real vocabulary construction as 06_tokenize_genomes.py, but here it's used to build a fixed-length COUNT vector, not a token ID sequence."""
    bases = ["A", "T", "C", "G"]
    all_kmers = ["".join(combo) for combo in itertools.product(bases, repeat=k)]
    return {kmer: idx for idx, kmer in enumerate(all_kmers)}


def compute_kmer_frequency_vector(fasta_path: Path, kmer_index: dict, k: int) -> np.ndarray:
    """
    Reads a REAL genome FASTA file and counts how many times each possible k-mer appears across the WHOLE sequence, then normalizes by total count -- giving a real frequency profile, the classic "bag of k-mers" representation used in genomic ML for decades.

    Unlike our transformer's token sequence (which only looks at a gene-targeted window), this deliberately scans the ENTIRE genome -- frequency counting is computationally cheap enough that there's no need to truncate the way we did for the transformer's token limit.
    """
    counts = np.zeros(len(kmer_index), dtype=np.float64)

    for record in SeqIO.parse(fasta_path, "fasta"):
        seq = str(record.seq).upper()
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i + k]
            if kmer in kmer_index:
                counts[kmer_index[kmer]] += 1

    total = counts.sum()
    if total > 0:
        counts = counts / total   # normalize to a real frequency profile

    return counts


# ============================================================
# STEP 2: Build the real, labeled dataset from the ledger
# ============================================================
def build_real_dataset(kmer_index: dict, k: int):
    if not LEDGER_CSV.exists():
        raise FileNotFoundError(f"Ledger not found at {LEDGER_CSV}. Run 05_build_ledger.py first.")

    with open(LEDGER_CSV, "r") as f:
        rows = [r for r in csv.DictReader(f) if r["data_type"] == "genomic"]

    X, y, sample_ids = [], [], []
    print(f"Computing real k-mer frequency vectors for {len(rows)} genomic samples...")
    print("(This scans full genome sequences -- may take a few minutes for real data.)\n")

    for i, row in enumerate(rows, start=1):
        fasta_path = Path(row["file_path"])
        if not fasta_path.exists():
            continue

        vector = compute_kmer_frequency_vector(fasta_path, kmer_index, k)
        X.append(vector)
        y.append(ORGANISM_TO_LABEL[row["organism"]])
        sample_ids.append(row["sample_id"])

        if i % 25 == 0 or i == len(rows):
            print(f"  Processed {i}/{len(rows)} samples...")

    return np.array(X), np.array(y), sample_ids


# ============================================================
# STEP 3: Actually fit the classical models on real data
# ============================================================
def train_and_check_classical_baseline():
    print("Initializing Classical ML Baseline (real k-mer frequency vectors)...")
    print("-" * 60)

    kmer_index = build_kmer_index(K)
    X, y, sample_ids = build_real_dataset(kmer_index, K)

    print(f"\nReal dataset built: {X.shape[0]} samples, {X.shape[1]} features "
          f"(4^{K} = {4**K} possible k-mers)")

    # A quick internal train/test split, JUST to confirm these models can actually learn something above chance from real data -- this is a sanity check, not the real, rigorous cross-validation that belongs in Phase 5.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nFitting XGBoost on {X_train.shape[0]} real training samples...")
    xgb_model = XGBClassifier(n_estimators=100, max_depth=6, objective='multi:softprob', n_jobs=1)
    xgb_model.fit(X_train, y_train)
    xgb_accuracy = xgb_model.score(X_test, y_test)
    print(f"  XGBoost real held-out accuracy (sanity check, not final Phase 5 result): {xgb_accuracy:.2%}")

    print(f"\nFitting Random Forest on {X_train.shape[0]} real training samples...")
    rf_model = RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_accuracy = rf_model.score(X_test, y_test)
    print(f"  Random Forest real held-out accuracy (sanity check, not final Phase 5 result): {rf_accuracy:.2%}")

    print("\n[PASS] Both classical models actually fitted and evaluated on real data.")
    return xgb_model, rf_model


# ============================================================
# STEP 3b: Build the REAL resistance-specific dataset (S. aureus only)

# WHY THIS SECOND BASELINE MATTERS: The organism classifier above (A. flavus / S. aureus / negative) separates a fungus from bacteria -- a genuinely "easy" distinction given how different fungal and bacterial genome composition is at the whole-genome level. That task was never the hard, scientifically interesting question this project is actually about. The real question -- resistant vs. susceptible WITHIN S. aureus, driven by a single gene's presence rather than broad genomic composition -- is a fundamentally harder, more meaningful test, and this baseline hasn't been checked until now.
# ============================================================
def build_real_resistance_dataset(kmer_index: dict, k: int):
    if not LEDGER_CSV.exists():
        raise FileNotFoundError(f"Ledger not found at {LEDGER_CSV}. Run 05_build_ledger.py first.")

    with open(LEDGER_CSV, "r") as f:
        all_rows = list(csv.DictReader(f))

    # Only real S. aureus samples with a confirmed, real resistance call -- excludes anything not actually resolved (e.g. SCAN_FAILED, if any).
    rows = [
        r for r in all_rows
        if r["organism"] == "Staphylococcus_aureus"
        and r["resistance_status"] in ("RESISTANT", "SUSCEPTIBLE")
    ]

    X, y, sample_ids = [], [], []
    print(f"Computing real k-mer frequency vectors for {len(rows)} S. aureus samples "
          f"(resistance-specific dataset)...")

    for i, row in enumerate(rows, start=1):
        fasta_path = Path(row["file_path"])
        if not fasta_path.exists():
            continue

        vector = compute_kmer_frequency_vector(fasta_path, kmer_index, k)
        X.append(vector)
        y.append(1 if row["resistance_status"] == "RESISTANT" else 0)
        sample_ids.append(row["sample_id"])

        if i % 25 == 0 or i == len(rows):
            print(f"  Processed {i}/{len(rows)} samples...")

    return np.array(X), np.array(y), sample_ids


def train_and_check_resistance_baseline(kmer_index: dict):
    print("\n" + "=" * 60)
    print("Resistance-Specific Baseline: RESISTANT vs. SUSCEPTIBLE within S. aureus")
    print("(This is the real, hard question -- not organism identity)")
    print("=" * 60)

    X, y, sample_ids = build_real_resistance_dataset(kmer_index, K)

    n_resistant = int(y.sum())
    n_susceptible = len(y) - n_resistant
    print(f"\nReal resistance dataset built: {len(y)} samples "
          f"({n_resistant} resistant, {n_susceptible} susceptible)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nFitting XGBoost on {X_train.shape[0]} real S. aureus training samples...")
    xgb_model = XGBClassifier(n_estimators=100, max_depth=6, objective='binary:logistic', n_jobs=1)
    xgb_model.fit(X_train, y_train)
    xgb_accuracy = xgb_model.score(X_test, y_test)
    print(f"  XGBoost real held-out accuracy (resistance task, sanity check): {xgb_accuracy:.2%}")

    print(f"\nFitting Random Forest on {X_train.shape[0]} real S. aureus training samples...")
    rf_model = RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_accuracy = rf_model.score(X_test, y_test)
    print(f"  Random Forest real held-out accuracy (resistance task, sanity check): {rf_accuracy:.2%}")

    print(f"\nFor reference: always guessing the majority class "
          f"({'resistant' if n_resistant > n_susceptible else 'susceptible'}) would score "
          f"{max(n_resistant, n_susceptible) / len(y):.2%} -- any real baseline result should "
          f"clearly beat this to mean anything.")

    print("\n[PASS] Resistance-specific baseline actually fitted and evaluated on real data.")
    return xgb_model, rf_model



# ============================================================
# Image baseline -- now REAL: loads, trains, and evaluates on the actual DIBaS/OpenFungi images downloaded in 03_download_images.py
# ============================================================
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score

IMAGES_DIR = Path("../data/images")
IMAGE_CLASS_TO_LABEL = {"aflavus": 0, "saureus": 1, "negative_control": 2}
VALID_EXTENSIONS = (".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp")


class RealMicroscopyImageDataset(Dataset):
    """
    Loads real image files from disk -- whatever formats DIBaS/OpenFungi actually shipped (checked defensively rather than assumed).
    """
    def __init__(self, transform):
        self.transform = transform
        self.samples = []   # list of (real_file_path, label)

        for class_name, label in IMAGE_CLASS_TO_LABEL.items():
            class_dir = IMAGES_DIR / class_name
            if not class_dir.exists():
                continue
            for path in class_dir.rglob("*"):
                if path.suffix.lower() in VALID_EXTENSIONS:
                    self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")   # RGB, since ResNet18 expects 3 channels
        return self.transform(image), label


def build_real_image_dataset():
    # Standard ImageNet normalization -- required since we're using ResNet18's real pretrained weights, which expect inputs preprocessed the same way the original model was trained.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = RealMicroscopyImageDataset(transform)

    if len(dataset) == 0:
        return None, None

    # Report the REAL class distribution -- this is where the severe imbalance (tens of images vs. hundreds) becomes visible and honest.
    class_counts = {name: 0 for name in IMAGE_CLASS_TO_LABEL}
    for _, label in dataset.samples:
        for name, lbl in IMAGE_CLASS_TO_LABEL.items():
            if lbl == label:
                class_counts[name] += 1

    print(f"Real image dataset loaded: {len(dataset)} total images")
    for name, count in class_counts.items():
        print(f"  {name}: {count} real images")

    return dataset, class_counts


def train_and_evaluate_image_baseline():
    print("\nInitializing Image-Based Baseline (Morphological Standard)...")
    print("-" * 60)

    dataset, class_counts = build_real_image_dataset()
    if dataset is None:
        print("[ERROR] No real images found under ../data/images/. "
              "Run 03_download_images.py first.")
        return None

    # HONEST LIMITATION: severe class imbalance (tens vs. hundreds of images) and a very small S. aureus count means even an 80/20 split leaves only a handful of real test images for that class. This is a genuine constraint of the available real data, not a bug.

    # BUG FIX: the previous version used torch's random_split, which does NOT guarantee proportional class representation -- by chance, this left only 2 real S. aureus images in the test set instead of the expected ~4, making an already-small sample even less meaningful. Fixed here with a proper STRATIFIED split, matching the same discipline used for the genomic ledger's train/test split.
    from sklearn.model_selection import train_test_split as sk_train_test_split
    all_labels = [label for _, label in dataset.samples]
    all_indices = list(range(len(dataset)))

    train_indices, test_indices = sk_train_test_split(
        all_indices, test_size=0.2, random_state=42, stratify=all_labels
    )

    train_set = torch.utils.data.Subset(dataset, train_indices)
    test_set = torch.utils.data.Subset(dataset, test_indices)
    print(f"\nStratified split: {len(train_set)} real training images, "
          f"{len(test_set)} real held-out test images")

    # Show the REAL per-class test set sizes up front -- this is exactly the number that determines how much any per-class result can actually be trusted.
    test_label_counts = {name: 0 for name in IMAGE_CLASS_TO_LABEL}
    for idx in test_indices:
        _, label = dataset.samples[idx]
        for name, lbl in IMAGE_CLASS_TO_LABEL.items():
            if lbl == label:
                test_label_counts[name] += 1
    print("Real test set composition (this is what any per-class result is actually based on):")
    for name, count in test_label_counts.items():
        print(f"  {name}: {count} real test images")

    # Real class weighting to counteract the severe imbalance -- without this, the model could just always predict "aflavus" (the majority class) and still look deceptively accurate.
    total = sum(class_counts.values())
    weights = torch.tensor([
        total / (len(class_counts) * class_counts["aflavus"]),
        total / (len(class_counts) * class_counts["saureus"]),
        total / (len(class_counts) * class_counts["negative_control"]),
    ], dtype=torch.float32)

    # Real model: ResNet18 with pretrained ImageNet weights, output head modified for our 3 classes. Given how little real data we have, we FREEZE all convolutional layers and only train the final classifier head -- proper transfer learning, and a real safeguard against overfitting on such a small dataset.
    image_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    for param in image_model.parameters():
        param.requires_grad = False   # freeze the real pretrained feature extractor

    num_ftrs = image_model.fc.in_features
    image_model.fc = nn.Linear(num_ftrs, 3)   # only this new layer will actually train

    optimizer = torch.optim.Adam(image_model.fc.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(weight=weights)

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=8, shuffle=False)

    print("\nFine-tuning ResNet18's final layer on real microscopy images...")
    image_model.train()
    n_epochs = 10
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = image_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 2 == 0 or epoch == n_epochs - 1:
            print(f"  Epoch {epoch + 1}/{n_epochs}: loss = {epoch_loss:.4f}")

    # --- Real evaluation on real held-out test images ---
    image_model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = image_model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.tolist())
            all_true.extend(labels.tolist())

    accuracy = accuracy_score(all_true, all_preds)
    macro_f1 = f1_score(all_true, all_preds, average="macro")

    # Real per-class breakdown -- essential here, since the aggregate macro-F1 can hide a tiny, high-variance result for S. aureus (only ~4 real test images) behind a large, easy A. flavus class.
    from sklearn.metrics import classification_report
    label_names = [name for name, _ in sorted(IMAGE_CLASS_TO_LABEL.items(), key=lambda x: x[1])]
    print("\nPer-class breakdown (READ THIS before trusting the aggregate numbers above):")
    print(classification_report(all_true, all_preds, target_names=label_names, zero_division=0))

    # Majority-class reference, same honest practice as the resistance baseline -- a real number this should clearly beat to mean anything.
    majority_class_frac = max(class_counts.values()) / total

    print(f"\nReal held-out test results ({len(test_set)} real images):")
    print(f"  Accuracy:  {accuracy:.2%}")
    print(f"  Macro-F1:  {macro_f1:.2%}  (more informative than accuracy here, given imbalance)")
    print(f"  For reference, always guessing the majority class would score "
          f"{majority_class_frac:.2%} accuracy")

    print("\n[PASS] Image baseline actually trained and evaluated on real DIBaS/OpenFungi images.")
    print("\nHONEST LIMITATIONS:")
    print("  - S. aureus has only ~20 real images total -- the held-out test set for")
    print("    this class is very small, so its result carries real, high uncertainty.")
    print("  - 'negative_control' combines two organisms (Candida albicans + E. coli)")
    print("    under one label -- morphologically distinct organisms sharing one class.")
    print("  - 'A. flavus' images are labeled 'Aspergillus section Flavi', not confirmed")
    print("    at the species level (see 03_download_images.py for detail).")

    return {"accuracy": accuracy, "macro_f1": macro_f1, "n_test": len(test_set),
             "test_label_counts": test_label_counts}


# ============================================================
# MAIN
# ============================================================
def compile_all_baselines():
    kmer_index = build_kmer_index(K)

    xgb_model, rf_model = train_and_check_classical_baseline()
    xgb_resistance_model, rf_resistance_model = train_and_check_resistance_baseline(kmer_index)
    image_results = train_and_evaluate_image_baseline()

    print("\n" + "-" * 60)
    print("Classical ML baselines (organism ID): real, fitted, and validated on real data.")
    print("Classical ML baselines (resistance): real, fitted, and validated on real data.")
    if image_results:
        print(f"Image baseline: real, trained, and evaluated "
              f"(accuracy={image_results['accuracy']:.2%}, macro-F1={image_results['macro_f1']:.2%}).")
    else:
        print("Image baseline: BLOCKED -- no real images found.")


if __name__ == "__main__":
    compile_all_baselines()
