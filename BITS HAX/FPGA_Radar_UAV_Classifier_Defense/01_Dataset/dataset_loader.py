#!/usr/bin/env python3
"""
=============================================================================
  FPGA Radar UAV Classifier — Dataset Loader
  Real Doppler RAD-DAR Kaggle Dataset (drone / car / person)
=============================================================================

  Purpose:
    - Download or locate the RAD-DAR Kaggle dataset zip file
    - Extract class-level CSV files (trimmed 11×61 Range-Doppler maps)
    - Concatenate into NumPy arrays: rd_maps.npy (N,11,61) & labels.npy (N,)
    - Print summary statistics and sample shapes

  Dataset:
    https://www.kaggle.com/datasets/iroldan/real-doppler-rad-dar-database

  Class Mapping:
    0 = drone   (UAV — primary threat target)
    1 = car     (ground vehicle)
    2 = person  (human)

  Input matrix shape:  11 rows (range bins) × 61 columns (Doppler bins)
  Value type:          float32  (dBm power levels)

  Usage:
    python dataset_loader.py                          # interactive mode
    python dataset_loader.py --zip /path/to/archive.zip  # specify zip path

  Author : BITS Pilani – AMD/Xilinx FPGA Hackathon 2026
  License: MIT
=============================================================================
"""

import os
import sys
import glob
import zipfile
import argparse
import shutil

import numpy as np
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Class label mapping (defense-oriented — drones first)
CLASS_MAP = {
    "drones": 0,
    "cars":   1,
    "people": 2,
}

# Expected shape of each trimmed Range-Doppler matrix
EXPECTED_ROWS = 11
EXPECTED_COLS = 61

# Output file paths (saved alongside this script)
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_MAPS  = os.path.join(SCRIPT_DIR, "rd_maps.npy")
OUTPUT_LABELS = os.path.join(SCRIPT_DIR, "labels.npy")

# Kaggle dataset slug (for CLI download)
KAGGLE_SLUG = "iroldan/real-doppler-rad-dar-database"


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Attempt Kaggle API download
# ─────────────────────────────────────────────────────────────────────────────
def try_kaggle_download(dest_dir: str) -> str | None:
    """
    Attempt to download the dataset via the Kaggle CLI.
    Returns the path to the downloaded zip, or None on failure.
    """
    try:
        import subprocess
        print("[INFO] Attempting Kaggle CLI download …")
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_SLUG, "-p", dest_dir],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode == 0:
            # Find the downloaded zip
            zips = glob.glob(os.path.join(dest_dir, "*.zip"))
            if zips:
                print(f"[OK]   Downloaded → {zips[0]}")
                return zips[0]
        else:
            print(f"[WARN] Kaggle CLI error: {result.stderr.strip()}")
    except FileNotFoundError:
        print("[WARN] Kaggle CLI not found (pip install kaggle).")
    except Exception as e:
        print(f"[WARN] Kaggle download failed: {e}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Locate or obtain the dataset
# ─────────────────────────────────────────────────────────────────────────────
def locate_dataset(zip_path: str | None = None) -> str:
    """
    Returns the path to the extracted dataset root containing class folders.
    Strategy:
      1. If class folders already exist in SCRIPT_DIR → use them directly
      2. If --zip was provided → extract it
      3. Try Kaggle CLI download
      4. Prompt user to manually place the zip
    """
    # ── 1. Class folders already extracted? ──────────────────────────────────
    dataset_root = find_class_root(SCRIPT_DIR)
    if dataset_root:
        print(f"[OK]   Class folders found in {dataset_root}. Skipping extraction.")
        return dataset_root

    # ── 2. Zip provided via CLI? ────────────────────────────────────────────
    if zip_path and os.path.isfile(zip_path):
        return extract_zip(zip_path)

    # ── 3. Check for any zip in SCRIPT_DIR ──────────────────────────────────
    existing_zips = glob.glob(os.path.join(SCRIPT_DIR, "*.zip"))
    if existing_zips:
        print(f"[OK]   Found existing zip: {existing_zips[0]}")
        return extract_zip(existing_zips[0])

    # ── 4. Kaggle CLI ───────────────────────────────────────────────────────
    kaggle_zip = try_kaggle_download(SCRIPT_DIR)
    if kaggle_zip:
        return extract_zip(kaggle_zip)

    # ── 5. Manual guidance ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  DATASET NOT FOUND — Manual Download Required")
    print("=" * 70)
    print(f"""
  1. Go to: https://www.kaggle.com/datasets/iroldan/real-doppler-rad-dar-database
  2. Click "Download" and save the zip file.
  3. Place the zip in:
     {SCRIPT_DIR}/
  4. Re-run this script.

  Or run with:  python dataset_loader.py --zip /path/to/downloaded.zip
""")
    sys.exit(1)


def extract_zip(zip_path: str) -> str:
    """Extract a zip file and return the path containing class folders."""
    print(f"[INFO] Extracting {os.path.basename(zip_path)} …")
    extract_dir = os.path.join(SCRIPT_DIR, "_extracted")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        for member in tqdm(members, desc="Extracting", unit="file"):
            zf.extract(member, extract_dir)

    # Walk extracted tree to find the directory containing class folders
    dataset_root = find_class_root(extract_dir)
    if dataset_root is None:
        print("[ERROR] Could not find drone/car/person folders in the zip.")
        sys.exit(1)

    # Move class folders up to SCRIPT_DIR for clean structure
    for cls in CLASS_MAP.keys():
        src = os.path.join(dataset_root, cls)
        dst = os.path.join(SCRIPT_DIR, cls)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.move(src, dst)

    # Cleanup
    shutil.rmtree(extract_dir, ignore_errors=True)
    print("[OK]   Extraction complete.")
    return SCRIPT_DIR


def find_class_root(base: str) -> str | None:
    """Recursively search for a directory containing all class sub-folders."""
    for root, dirs, _ in os.walk(base):
        lower_dirs = {d.lower() for d in dirs}
        if all(cls in lower_dirs for cls in CLASS_MAP.keys()):
            # Return with correct casing
            return root
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Core: Load CSV files into NumPy arrays
# ─────────────────────────────────────────────────────────────────────────────
def load_range_doppler_maps(dataset_root: str):
    """
    Load all CSV Range-Doppler matrices from class folders.

    Each CSV file contains an 11×61 matrix (rows=range, cols=Doppler).
    Values are floating-point power levels in dBm.

    Returns:
        rd_maps : np.ndarray, shape (N, 11, 61), dtype float32
        labels  : np.ndarray, shape (N,),         dtype int64
    """
    all_maps  = []
    all_labels = []
    skipped   = 0
    class_counts = {}

    for class_name, class_label in CLASS_MAP.items():
        # Locate the class folder (case-insensitive search)
        class_dir = None
        for entry in os.listdir(dataset_root):
            if entry.lower() == class_name.lower() and os.path.isdir(
                os.path.join(dataset_root, entry)
            ):
                class_dir = os.path.join(dataset_root, entry)
                break

        if class_dir is None:
            print(f"[WARN] Class folder '{class_name}' not found. Skipping.")
            continue

        # Gather CSV files (recursively — some datasets nest inside subfolders)
        csv_files = sorted(glob.glob(os.path.join(class_dir, "**", "*.csv"), recursive=True))
        class_counts[class_name] = 0

        for csv_path in tqdm(
            csv_files,
            desc=f"  Loading {class_name:>6s} (label={class_label})",
            unit="file",
            leave=True,
        ):
            try:
                # Load the 11×61 Range-Doppler matrix
                rd_matrix = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)

                # Validate shape
                if rd_matrix.shape == (EXPECTED_ROWS, EXPECTED_COLS):
                    all_maps.append(rd_matrix)
                    all_labels.append(class_label)
                    class_counts[class_name] += 1
                else:
                    # Try to trim/pad if close to expected shape
                    skipped += 1
            except Exception:
                skipped += 1

    if len(all_maps) == 0:
        print("[ERROR] No valid Range-Doppler maps loaded!")
        sys.exit(1)

    # Stack into single arrays
    rd_maps = np.stack(all_maps, axis=0)   # (N, 11, 61)
    labels  = np.array(all_labels, dtype=np.int64)  # (N,)

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  DATASET LOADING SUMMARY")
    print("=" * 60)
    print(f"  Total samples loaded : {len(rd_maps):,}")
    print(f"  Samples skipped      : {skipped:,}")
    print(f"  RD map shape         : {rd_maps.shape}  (N × Range × Doppler)")
    print(f"  Labels shape         : {labels.shape}")
    print(f"  Data type            : {rd_maps.dtype}")
    print(f"  Value range          : [{rd_maps.min():.2f}, {rd_maps.max():.2f}] dBm")
    print()
    for cls_name, cls_label in CLASS_MAP.items():
        count = class_counts.get(cls_name, 0)
        pct = 100.0 * count / len(labels) if len(labels) > 0 else 0
        print(f"    {cls_name:>6s} (label {cls_label}) : {count:>5,} samples ({pct:5.1f}%)")
    print("=" * 60)

    return rd_maps, labels


# ─────────────────────────────────────────────────────────────────────────────
# Save to disk
# ─────────────────────────────────────────────────────────────────────────────
def save_arrays(rd_maps: np.ndarray, labels: np.ndarray):
    """Persist arrays as .npy files for downstream training."""
    np.save(OUTPUT_MAPS, rd_maps)
    np.save(OUTPUT_LABELS, labels)
    maps_mb   = os.path.getsize(OUTPUT_MAPS) / (1024 * 1024)
    labels_kb = os.path.getsize(OUTPUT_LABELS) / 1024
    print(f"\n  [SAVED] {OUTPUT_MAPS}  ({maps_mb:.1f} MB)")
    print(f"  [SAVED] {OUTPUT_LABELS}  ({labels_kb:.1f} KB)")


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check — print sample info
# ─────────────────────────────────────────────────────────────────────────────
def show_sample_info(rd_maps: np.ndarray, labels: np.ndarray):
    """Display shape and statistics for quick verification."""
    print("\n" + "─" * 60)
    print("  SAMPLE VERIFICATION")
    print("─" * 60)
    print(f"  rd_maps.shape  = {rd_maps.shape}")
    print(f"  rd_maps.dtype  = {rd_maps.dtype}")
    print(f"  labels.shape   = {labels.shape}")
    print(f"  labels.dtype   = {labels.dtype}")
    print(f"  Unique labels  = {np.unique(labels).tolist()}")
    print()
    print(f"  Sample rd_maps[0] — shape: {rd_maps[0].shape}")
    print(f"    min={rd_maps[0].min():.4f}  max={rd_maps[0].max():.4f}"
          f"  mean={rd_maps[0].mean():.4f}  std={rd_maps[0].std():.4f}")
    print(f"    label = {labels[0]}  ({list(CLASS_MAP.keys())[labels[0]]})")
    print("─" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="RAD-DAR Dataset Loader — Range-Doppler maps for UAV classification"
    )
    parser.add_argument(
        "--zip", type=str, default=None,
        help="Path to the downloaded Kaggle dataset zip file"
    )
    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   FPGA Radar UAV Classifier — Dataset Loader               ║")
    print("║   Defense Systems • BITS Pilani AMD/Xilinx Hackathon 2026  ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # Step 1: Locate / download / extract the dataset
    dataset_root = locate_dataset(zip_path=args.zip)

    # Step 2: Load all CSVs into NumPy arrays
    rd_maps, labels = load_range_doppler_maps(dataset_root)

    # Step 3: Save to disk
    save_arrays(rd_maps, labels)

    # Step 4: Sanity check
    show_sample_info(rd_maps, labels)

    print("\n  ✅ Dataset ready for training!\n")


if __name__ == "__main__":
    main()
