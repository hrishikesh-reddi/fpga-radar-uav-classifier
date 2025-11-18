#!/usr/bin/env python3
"""
=============================================================================
  FPGA Radar UAV Classifier — Model Training Script
  Lightweight RadarCNN for Range-Doppler Map Classification
=============================================================================

  Architecture (designed for FPGA deployment via hls4ml):
    Input:  1 × 11 × 61  (single-channel Range-Doppler map)
    Conv2d: 1 → 8 filters, 3×3, ReLU, BatchNorm
    MaxPool2d: 2×2
    Conv2d: 8 → 16 filters, 3×3, ReLU, BatchNorm
    MaxPool2d: 2×2
    Flatten → FC(16*h*w → 32) → ReLU → Dropout
    FC(32 → 3)  — output logits for [drone, car, person]

  Target:  >92% test accuracy
  Export:  model.pth (PyTorch state dict) + model_full.pth (full model)

  Defense Context:
    - Real-time UAV/threat classification from FMCW radar returns
    - Minimal latency, low power — suitable for edge FPGA inference
    - Class 0 = drone (primary threat), Class 1 = car, Class 2 = person

  Usage:
    python train_model.py
    python train_model.py --epochs 30 --lr 0.001 --batch_size 64

  Author : BITS Pilani – AMD/Xilinx FPGA Hackathon 2026
  License: MIT
=============================================================================
"""

import os
import sys
import argparse
import time
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for servers/scripts
import matplotlib.pyplot as plt
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR  = os.path.join(os.path.dirname(SCRIPT_DIR), "01_Dataset")
MAPS_PATH    = os.path.join(DATASET_DIR, "rd_maps.npy")
LABELS_PATH  = os.path.join(DATASET_DIR, "labels.npy")

CLASS_NAMES  = ["drone", "car", "person"]
NUM_CLASSES  = 3
INPUT_H, INPUT_W = 11, 61  # Range × Doppler

# Output paths
MODEL_STATE_PATH = os.path.join(SCRIPT_DIR, "model.pth")
MODEL_FULL_PATH  = os.path.join(SCRIPT_DIR, "model_full.pth")
HISTORY_PATH     = os.path.join(SCRIPT_DIR, "training_history.json")
PLOTS_DIR        = os.path.join(SCRIPT_DIR, "plots")


# ─────────────────────────────────────────────────────────────────────────────
# Model Architecture — RadarCNN (FPGA-friendly, quantization-aware design)
# ─────────────────────────────────────────────────────────────────────────────
class RadarCNN(nn.Module):
    """
    Lightweight 2D CNN for Range-Doppler map classification.

    Architecture chosen for FPGA deployment:
      - Small filter counts (8, 16) to minimize DSP usage
      - Standard Conv2d + ReLU + MaxPool — well-supported by hls4ml
      - BatchNorm foldable into Conv weights for inference
      - Compact FC layers for low LUT/BRAM consumption

    Input shape:  (batch, 1, 11, 61)
    Output shape: (batch, 3)  — logits for [drone, car, person]
    """

    def __init__(self, num_classes: int = 3):
        super(RadarCNN, self).__init__()

        # ── Convolutional Feature Extractor ──────────────────────────────────
        # Block 1: 1 → 8 channels
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=8,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1   = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # After pool1: 8 × 5 × 30

        # Block 2: 8 → 16 channels
        self.conv2 = nn.Conv2d(
            in_channels=8, out_channels=16,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2   = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # After pool2: 16 × 2 × 15

        # ── Classifier Head ─────────────────────────────────────────────────
        self.flatten = nn.Flatten()
        self.fc1     = nn.Linear(16 * 2 * 15, 32)
        self.relu3   = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2     = nn.Linear(32, num_classes)

    def forward(self, x):
        """Forward pass: (B, 1, 11, 61) → (B, num_classes)"""
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Classifier
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading & Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
def load_data(test_size: float = 0.2, val_size: float = 0.1, seed: int = 42):
    """
    Load rd_maps.npy / labels.npy, normalize, and split into
    train / validation / test sets.

    Normalization: per-sample standardization (zero mean, unit variance)
    to handle varying radar power levels across scenes.
    """
    print("[INFO] Loading dataset …")
    if not os.path.exists(MAPS_PATH):
        print(f"[ERROR] {MAPS_PATH} not found!")
        print("        Run 01_Dataset/dataset_loader.py first.")
        sys.exit(1)

    rd_maps = np.load(MAPS_PATH)   # (N, 11, 61) float32
    labels  = np.load(LABELS_PATH) # (N,) int64

    print(f"  Loaded {len(rd_maps):,} samples — shape: {rd_maps.shape}")

    # ── Normalize: per-sample zero-mean, unit-variance ──────────────────────
    # This is critical for stable training with varying dBm ranges
    mean = rd_maps.mean(axis=(1, 2), keepdims=True)
    std  = rd_maps.std(axis=(1, 2), keepdims=True) + 1e-8
    rd_maps = (rd_maps - mean) / std

    # ── Add channel dimension: (N, 11, 61) → (N, 1, 11, 61) ───────────────
    rd_maps = rd_maps[:, np.newaxis, :, :]

    # ── Train / Val / Test split ────────────────────────────────────────────
    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        rd_maps, labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    # Second split: train vs val
    relative_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=relative_val,
        random_state=seed,
        stratify=y_trainval,
    )

    print(f"  Train: {len(X_train):>6,}  |  Val: {len(X_val):>5,}  |  Test: {len(X_test):>5,}")

    # ── Convert to PyTorch tensors ──────────────────────────────────────────
    def to_dataset(X, y):
        return TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        )

    return to_dataset(X_train, y_train), to_dataset(X_val, y_val), to_dataset(X_test, y_test)


# ─────────────────────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total   = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total   += targets.size(0)

    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model. Returns (avg_loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total   = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total   += targets.size(0)

    return running_loss / total, 100.0 * correct / total


# ─────────────────────────────────────────────────────────────────────────────
# Full Evaluation with Metrics
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def full_evaluation(model, loader, device):
    """
    Run full evaluation and return predictions + ground truth
    for classification report and confusion matrix.
    """
    model.eval()
    all_preds  = []
    all_labels = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(targets.numpy())

    return np.array(all_preds), np.array(all_labels)


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────
def plot_training_history(history: dict, save_dir: str):
    """Generate and save training plots (loss + accuracy curves)."""
    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Loss ────────────────────────────────────────────────────────────────
    ax1.plot(epochs, history["train_loss"], "b-o", markersize=3, label="Train Loss")
    ax1.plot(epochs, history["val_loss"],   "r-s", markersize=3, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── Accuracy ────────────────────────────────────────────────────────────
    ax2.plot(epochs, history["train_acc"], "b-o", markersize=3, label="Train Acc")
    ax2.plot(epochs, history["val_acc"],   "r-s", markersize=3, label="Val Acc")
    ax2.axhline(y=92, color="green", linestyle="--", alpha=0.7, label="Target: 92%")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {path}")


def plot_confusion_matrix(y_true, y_pred, class_names, save_dir: str):
    """Generate and save confusion matrix heatmap."""
    os.makedirs(save_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True Label",
        xlabel="Predicted Label",
        title="Confusion Matrix — Test Set",
    )

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14, fontweight="bold",
            )

    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Model Summary
# ─────────────────────────────────────────────────────────────────────────────
def print_model_summary(model):
    """Print model architecture and parameter count."""
    print("\n" + "=" * 60)
    print("  RADARCNN ARCHITECTURE")
    print("=" * 60)
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total parameters     : {total_params:,}")
    print(f"  Trainable parameters : {trainable:,}")
    print(f"  Model size (est.)    : {total_params * 4 / 1024:.1f} KB (float32)")
    print(f"  Quantized size (est.): {total_params * 1 / 1024:.1f} KB (int8)")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Main Training Pipeline
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="RadarCNN Training — UAV/Target Classification from Range-Doppler Maps"
    )
    parser.add_argument("--epochs",     type=int,   default=20,    help="Number of training epochs")
    parser.add_argument("--batch_size", type=int,   default=64,    help="Batch size")
    parser.add_argument("--lr",         type=float, default=1e-3,  help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="L2 regularization")
    parser.add_argument("--seed",       type=int,   default=42,    help="Random seed")
    args = parser.parse_args()

    # ── Banner ──────────────────────────────────────────────────────────────
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   FPGA Radar UAV Classifier — Model Training               ║")
    print("║   Defense Systems • BITS Pilani AMD/Xilinx Hackathon 2026  ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # ── Reproducibility ─────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── Device selection ────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[INFO] Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        print("[INFO] Using CPU")

    # ── Load Data ───────────────────────────────────────────────────────────
    train_ds, val_ds, test_ds = load_data(seed=args.seed)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, drop_last=False)

    # ── Model ───────────────────────────────────────────────────────────────
    model = RadarCNN(num_classes=NUM_CLASSES).to(device)
    print_model_summary(model)

    # ── Loss, Optimizer, Scheduler ──────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    # Cosine annealing for smooth LR decay
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Training Loop ───────────────────────────────────────────────────────
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
    }
    best_val_acc = 0.0
    best_epoch   = 0

    print(f"\n{'Epoch':>5s} │ {'Train Loss':>10s} │ {'Train Acc':>9s} │"
          f" {'Val Loss':>10s} │ {'Val Acc':>9s} │ {'LR':>10s} │ Notes")
    print("──────┼────────────┼───────────┼────────────┼───────────┼────────────┼──────")

    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Track best model
        note = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            torch.save(model.state_dict(), MODEL_STATE_PATH)
            torch.save(model, MODEL_FULL_PATH)
            note = "★ best"

        print(f"  {epoch:3d} │ {train_loss:10.4f} │ {train_acc:8.2f}% │"
              f" {val_loss:10.4f} │ {val_acc:8.2f}% │ {current_lr:10.6f} │ {note}")

    elapsed = time.time() - t_start
    print(f"\n  Training time: {elapsed:.1f}s  ({elapsed/args.epochs:.1f}s/epoch)")
    print(f"  Best val accuracy: {best_val_acc:.2f}% (epoch {best_epoch})")

    # ── Load Best Model & Test ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FINAL EVALUATION ON TEST SET")
    print("=" * 60)

    model.load_state_dict(torch.load(MODEL_STATE_PATH, map_location=device, weights_only=True))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\n  Test Loss    : {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.2f}%")

    if test_acc >= 92.0:
        print(f"  ✅ TARGET ACHIEVED: {test_acc:.2f}% ≥ 92%")
    else:
        print(f"  ⚠️  Below target: {test_acc:.2f}% < 92% — consider more epochs or tuning")

    # ── Detailed Classification Report ──────────────────────────────────────
    y_pred, y_true = full_evaluation(model, test_loader, device)
    print("\n  Classification Report:")
    print("  " + "-" * 55)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    for line in report.split("\n"):
        print(f"  {line}")

    # ── Generate Plots ──────────────────────────────────────────────────────
    plot_training_history(history, PLOTS_DIR)
    plot_confusion_matrix(y_true, y_pred, CLASS_NAMES, PLOTS_DIR)

    # ── Save Training History ───────────────────────────────────────────────
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  [SAVED] {HISTORY_PATH}")

    # ── Save Summary ────────────────────────────────────────────────────────
    print(f"\n  [SAVED] {MODEL_STATE_PATH}  (state dict — for hls4ml export)")
    print(f"  [SAVED] {MODEL_FULL_PATH}   (full model — for quick inference)")

    # ── Model Info for Next Stage ───────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  NEXT STEP: hls4ml Conversion")
    print("─" * 60)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters : {total_params:,}")
    print(f"  Input shape      : (1, 1, {INPUT_H}, {INPUT_W})")
    print(f"  Output classes   : {NUM_CLASSES} ({', '.join(CLASS_NAMES)})")
    print(f"  Model file       : {MODEL_STATE_PATH}")
    print(f"  Target FPGA      : xc7z020clg400-1 (ZedBoard/PYNQ)")
    print("─" * 60)
    print("\n  ✅ Training complete! Ready for quantization & FPGA synthesis.\n")


if __name__ == "__main__":
    main()
