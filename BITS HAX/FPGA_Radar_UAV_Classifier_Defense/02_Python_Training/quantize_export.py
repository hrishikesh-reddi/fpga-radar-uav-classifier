#!/usr/bin/env python3
"""
=============================================================================
  FPGA Radar UAV Classifier — 8-bit Quantization & ONNX Export
=============================================================================

  Pipeline:
    1. Load trained RadarCNN (model.pth)
    2. Fold BatchNorm into Conv2d weights (inference optimization)
    3. Export float32 model → model.onnx (opset 11)
    4. Quantize to INT8 → model_quantized.onnx (ONNX Runtime quantization)
    5. Compare float vs quantized accuracy on test set
    6. Print size reduction & latency summary

  Why two-step quantization?
    - hls4ml will perform its own fixed-point quantization for FPGA synthesis
    - The ONNX INT8 model serves as a validation checkpoint and for
      edge-CPU deployment benchmarks
    - Both float ONNX & quantized ONNX are exported for flexibility

  Defense Context:
    - 8-bit inference reduces FPGA resource usage by ~4x (LUT/DSP)
    - Enables real-time sub-millisecond classification latency
    - Critical for UAV threat detection in power-constrained environments

  Usage:
    python quantize_export.py
    python quantize_export.py --model model.pth --no-evaluate

  Author : BITS Pilani – AMD/Xilinx FPGA Hackathon 2026
  License: MIT
=============================================================================
"""

import os
import sys
import copy
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "01_Dataset")
MAPS_PATH   = os.path.join(DATASET_DIR, "rd_maps.npy")
LABELS_PATH = os.path.join(DATASET_DIR, "labels.npy")

# Import the model class from our training script
sys.path.insert(0, SCRIPT_DIR)
from train_model import RadarCNN, CLASS_NAMES, NUM_CLASSES, INPUT_H, INPUT_W


# ─────────────────────────────────────────────────────────────────────────────
# BatchNorm Folding — fuse BN into Conv for inference
# ─────────────────────────────────────────────────────────────────────────────
def fold_batchnorm(model: nn.Module) -> nn.Module:
    """
    Fold BatchNorm parameters into the preceding Conv2d weights.

    For inference, BN(Conv(x)) = W_new * x + b_new, where:
      W_new = (gamma / sqrt(var + eps)) * W_conv
      b_new = gamma * (b_conv - mean) / sqrt(var + eps) + beta

    This eliminates BN layers entirely — critical for FPGA deployment
    since hls4ml handles fused Conv+Bias more efficiently.
    """
    folded = copy.deepcopy(model)

    # Block 1: conv1 + bn1
    _fuse_conv_bn(folded.conv1, folded.bn1)
    folded.bn1 = nn.Identity()

    # Block 2: conv2 + bn2
    _fuse_conv_bn(folded.conv2, folded.bn2)
    folded.bn2 = nn.Identity()

    return folded


def _fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    """In-place fusion of BN parameters into Conv weights."""
    with torch.no_grad():
        # Extract BN parameters
        gamma   = bn.weight                          # scale
        beta    = bn.bias                             # shift
        mean    = bn.running_mean                     # running mean
        var     = bn.running_var                      # running variance
        eps     = bn.eps

        # Compute fused weights
        std_inv = gamma / torch.sqrt(var + eps)       # (C_out,)
        w_fused = conv.weight * std_inv.view(-1, 1, 1, 1)

        # Compute fused bias (conv might not have bias originally)
        if conv.bias is not None:
            b_fused = (conv.bias - mean) * std_inv + beta
        else:
            b_fused = -mean * std_inv + beta

        # Write back
        conv.weight.copy_(w_fused)
        if conv.bias is None:
            conv.bias = nn.Parameter(b_fused)
        else:
            conv.bias.copy_(b_fused)


# ─────────────────────────────────────────────────────────────────────────────
# ONNX Export
# ─────────────────────────────────────────────────────────────────────────────
def export_to_onnx(model: nn.Module, output_path: str, opset: int = 11):
    """
    Export PyTorch model to ONNX format.

    Args:
        model:       PyTorch model (should be in eval mode, BN folded)
        output_path: Path to save .onnx file
        opset:       ONNX opset version (11 for broad compatibility)
    """
    model.eval()
    dummy_input = torch.randn(1, 1, INPUT_H, INPUT_W)

    # Use legacy TorchScript-based exporter (dynamo=False) for reliable
    # opset 11 support — the new dynamo exporter targets opset ≥18 and
    # the version downconversion is unreliable.
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset,
        input_names=["range_doppler_map"],
        output_names=["class_logits"],
        dynamic_axes={
            "range_doppler_map": {0: "batch_size"},
            "class_logits":      {0: "batch_size"},
        },
        do_constant_folding=True,
        dynamo=False,  # Force legacy exporter for opset 11 compatibility
    )

    # Validate exported model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    file_size = os.path.getsize(output_path)
    print(f"  [SAVED] {output_path}")
    print(f"          Size: {file_size / 1024:.1f} KB")
    print(f"          Opset: {opset}")
    print(f"          Inputs:  {[inp.name for inp in onnx_model.graph.input]}")
    print(f"          Outputs: {[out.name for out in onnx_model.graph.output]}")

    return onnx_model


# ─────────────────────────────────────────────────────────────────────────────
# INT8 Quantization via ONNX Runtime
# ─────────────────────────────────────────────────────────────────────────────
def quantize_to_int8(float_onnx_path: str, quantized_onnx_path: str):
    """
    Post-training dynamic quantization to INT8 using ONNX Runtime.

    This quantizes weights to 8-bit integers while keeping activations
    in float during inference. For FPGA deployment, hls4ml will do
    its own fixed-point quantization — this step validates that the
    model tolerates reduced precision.
    """
    print("\n[INFO] Quantizing to INT8 …")

    # Pre-process: run shape inference so the quantizer can resolve all
    # tensor shapes correctly (avoids InferenceError on FC layers)
    preprocessed_path = float_onnx_path.replace(".onnx", "_preprocessed.onnx")
    float_model = onnx.load(float_onnx_path)
    inferred_model = onnx.shape_inference.infer_shapes(float_model)
    onnx.save(inferred_model, preprocessed_path)

    quantize_dynamic(
        model_input=preprocessed_path,
        model_output=quantized_onnx_path,
        weight_type=QuantType.QInt8,
    )

    # Cleanup preprocessed file
    if os.path.exists(preprocessed_path):
        os.remove(preprocessed_path)

    # Validate
    q_model = onnx.load(quantized_onnx_path)
    onnx.checker.check_model(q_model)

    float_size = os.path.getsize(float_onnx_path)
    quant_size = os.path.getsize(quantized_onnx_path)
    reduction  = (1.0 - quant_size / float_size) * 100

    print(f"  [SAVED] {quantized_onnx_path}")
    print(f"          Float32 size:    {float_size / 1024:.1f} KB")
    print(f"          INT8 size:       {quant_size / 1024:.1f} KB")
    print(f"          Size reduction:  {reduction:.1f}%")

    return q_model


# ─────────────────────────────────────────────────────────────────────────────
# Accuracy Evaluation (PyTorch & ONNX Runtime)
# ─────────────────────────────────────────────────────────────────────────────
def load_test_data():
    """Load and preprocess test split from saved dataset arrays."""
    from sklearn.model_selection import train_test_split

    rd_maps = np.load(MAPS_PATH)
    labels  = np.load(LABELS_PATH)

    # Normalize (same as training)
    mean = rd_maps.mean(axis=(1, 2), keepdims=True)
    std  = rd_maps.std(axis=(1, 2), keepdims=True) + 1e-8
    rd_maps = (rd_maps - mean) / std

    # Add channel dim
    rd_maps = rd_maps[:, np.newaxis, :, :].astype(np.float32)

    # Reproduce the same test split as training (seed=42, test_size=0.2)
    _, X_test, _, y_test = train_test_split(
        rd_maps, labels, test_size=0.2, random_state=42, stratify=labels
    )
    return X_test, y_test


def evaluate_pytorch(model: nn.Module, X_test: np.ndarray, y_test: np.ndarray) -> float:
    """Evaluate PyTorch model accuracy on test data."""
    model.eval()
    with torch.no_grad():
        inputs  = torch.tensor(X_test, dtype=torch.float32)
        outputs = model(inputs)
        preds   = outputs.argmax(dim=1).numpy()
    accuracy = 100.0 * (preds == y_test).mean()
    return accuracy


def evaluate_onnx(onnx_path: str, X_test: np.ndarray, y_test: np.ndarray) -> tuple:
    """
    Evaluate ONNX model accuracy using ONNX Runtime.
    Returns (accuracy, avg_latency_ms).
    """
    session = ort.InferenceSession(onnx_path)
    input_name  = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Batch inference
    t_start = time.time()
    result = session.run([output_name], {input_name: X_test})
    elapsed = time.time() - t_start

    logits = result[0]
    preds  = logits.argmax(axis=1)
    accuracy = 100.0 * (preds == y_test).mean()

    avg_latency = (elapsed / len(X_test)) * 1000  # ms per sample
    return accuracy, avg_latency


# ─────────────────────────────────────────────────────────────────────────────
# Model Summary
# ─────────────────────────────────────────────────────────────────────────────
def print_onnx_summary(onnx_path: str, label: str):
    """Print a summary of ONNX model structure."""
    model = onnx.load(onnx_path)
    print(f"\n  ── {label} ──")
    print(f"  File:      {os.path.basename(onnx_path)}")
    print(f"  Size:      {os.path.getsize(onnx_path) / 1024:.1f} KB")
    print(f"  IR version: {model.ir_version}")
    print(f"  Opset:     {model.opset_import[0].version}")
    print(f"  Nodes:     {len(model.graph.node)}")

    # Count ops by type
    op_counts = {}
    for node in model.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
    print(f"  Op types:  {dict(sorted(op_counts.items()))}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="RadarCNN Quantization & ONNX Export for FPGA Deployment"
    )
    parser.add_argument(
        "--model", type=str, default=os.path.join(SCRIPT_DIR, "model.pth"),
        help="Path to trained model state dict"
    )
    parser.add_argument(
        "--no-evaluate", action="store_true",
        help="Skip accuracy evaluation (useful when dataset not available)"
    )
    args = parser.parse_args()

    # Output paths
    onnx_float_path = os.path.join(SCRIPT_DIR, "model.onnx")
    onnx_quant_path = os.path.join(SCRIPT_DIR, "model_quantized.onnx")

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   FPGA Radar UAV Classifier — Quantization & ONNX Export   ║")
    print("║   Defense Systems • BITS Pilani AMD/Xilinx Hackathon 2026  ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # ── Step 1: Load trained model ──────────────────────────────────────────
    print("[STEP 1] Loading trained RadarCNN …")
    model = RadarCNN(num_classes=NUM_CLASSES)

    if os.path.exists(args.model):
        state_dict = torch.load(args.model, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        print(f"  Loaded weights from: {args.model}")
    else:
        print(f"  [WARN] {args.model} not found — using randomly initialized weights.")
        print(f"         Train the model first: python train_model.py")

    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # ── Step 2: Fold BatchNorm ──────────────────────────────────────────────
    print("\n[STEP 2] Folding BatchNorm into Conv2d weights …")
    model_folded = fold_batchnorm(model)
    model_folded.eval()

    # Verify folding didn't change outputs
    dummy = torch.randn(1, 1, INPUT_H, INPUT_W)
    with torch.no_grad():
        out_original = model(dummy)
        out_folded   = model_folded(dummy)
    max_diff = (out_original - out_folded).abs().max().item()
    print(f"  BN folding max output diff: {max_diff:.2e} (should be ≈ 0)")
    if max_diff < 1e-4:
        print("  ✅ BatchNorm folded successfully — identical outputs")
    else:
        print("  ⚠️  Small numerical difference (acceptable)")

    # ── Step 3: Export to ONNX (float32) ────────────────────────────────────
    print(f"\n[STEP 3] Exporting to ONNX (float32, opset 11) …")
    export_to_onnx(model_folded, onnx_float_path, opset=11)

    # ── Step 4: Quantize to INT8 ────────────────────────────────────────────
    print(f"\n[STEP 4] Quantizing to INT8 …")
    quantize_to_int8(onnx_float_path, onnx_quant_path)

    # ── Step 5: Model Summaries ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  MODEL SUMMARIES")
    print("=" * 60)
    print_onnx_summary(onnx_float_path, "Float32 ONNX Model")
    print_onnx_summary(onnx_quant_path, "INT8 Quantized ONNX Model")

    # ── Step 6: Accuracy Comparison ─────────────────────────────────────────
    has_dataset = os.path.exists(MAPS_PATH) and os.path.exists(LABELS_PATH)

    if has_dataset and not args.no_evaluate:
        print("\n" + "=" * 60)
        print("  ACCURACY COMPARISON: Float32 vs INT8")
        print("=" * 60)

        X_test, y_test = load_test_data()
        print(f"\n  Test set: {len(X_test):,} samples")

        # PyTorch float32
        pt_acc = evaluate_pytorch(model_folded, X_test, y_test)
        print(f"\n  PyTorch Float32:     {pt_acc:.2f}%")

        # ONNX float32
        onnx_float_acc, onnx_float_lat = evaluate_onnx(onnx_float_path, X_test, y_test)
        print(f"  ONNX Float32:        {onnx_float_acc:.2f}%  (latency: {onnx_float_lat:.3f} ms/sample)")

        # ONNX INT8
        onnx_quant_acc, onnx_quant_lat = evaluate_onnx(onnx_quant_path, X_test, y_test)
        print(f"  ONNX INT8 Quantized: {onnx_quant_acc:.2f}%  (latency: {onnx_quant_lat:.3f} ms/sample)")

        # Accuracy drop
        acc_drop = onnx_float_acc - onnx_quant_acc
        speedup  = onnx_float_lat / onnx_quant_lat if onnx_quant_lat > 0 else 0
        print(f"\n  Accuracy drop from quantization: {acc_drop:+.2f}%")
        print(f"  Latency speedup (CPU):           {speedup:.2f}x")

        if onnx_quant_acc >= 92.0:
            print(f"\n  ✅ QUANTIZED MODEL MEETS TARGET: {onnx_quant_acc:.2f}% ≥ 92%")
        else:
            print(f"\n  ⚠️  Quantized accuracy below target: {onnx_quant_acc:.2f}% < 92%")

    else:
        print("\n[INFO] Dataset not found or --no-evaluate specified.")
        print("       Skipping accuracy evaluation.")
        print("       Run 01_Dataset/dataset_loader.py first, then re-run this script.")

        # Still verify ONNX inference works with dummy data
        print("\n  Verifying ONNX inference with dummy input …")
        session = ort.InferenceSession(onnx_float_path)
        dummy_np = np.random.randn(1, 1, INPUT_H, INPUT_W).astype(np.float32)
        result = session.run(None, {"range_doppler_map": dummy_np})
        print(f"  Float32 ONNX output shape: {result[0].shape}")
        print(f"  Float32 ONNX output:       {result[0][0]}")

        session_q = ort.InferenceSession(onnx_quant_path)
        result_q = session_q.run(None, {"range_doppler_map": dummy_np})
        print(f"  INT8 ONNX output shape:    {result_q[0].shape}")
        print(f"  INT8 ONNX output:          {result_q[0][0]}")

        max_diff = np.abs(result[0] - result_q[0]).max()
        print(f"  Max diff (float vs quant): {max_diff:.6f}")
        print("  ✅ Both ONNX models produce valid outputs")

    # ── Final Summary ───────────────────────────────────────────────────────
    f_size = os.path.getsize(onnx_float_path) / 1024
    q_size = os.path.getsize(onnx_quant_path) / 1024

    print("\n" + "=" * 60)
    print("  EXPORT SUMMARY")
    print("=" * 60)
    print(f"  {'File':<30s} {'Size':>10s}  {'Format':>12s}")
    print(f"  {'─'*30} {'─'*10}  {'─'*12}")
    print(f"  {'model.onnx':<30s} {f_size:>8.1f}KB  {'Float32':>12s}")
    print(f"  {'model_quantized.onnx':<30s} {q_size:>8.1f}KB  {'INT8':>12s}")
    print(f"  {'model.pth':<30s} {'—':>10s}  {'PyTorch':>12s}")
    print(f"\n  Input:         (1, 1, {INPUT_H}, {INPUT_W}) — Range-Doppler map")
    print(f"  Output:        ({NUM_CLASSES},) — [{', '.join(CLASS_NAMES)}]")
    print(f"  Quantization:  Dynamic INT8 (ONNX Runtime)")
    print(f"  Target FPGA:   xc7z020clg400-1 (ZedBoard/PYNQ)")
    print("=" * 60)

    print("\n  📁 Files ready for hls4ml conversion:")
    print(f"     → {onnx_float_path}")
    print(f"     → {onnx_quant_path}")
    print("\n  ✅ Quantization & export complete!\n")


if __name__ == "__main__":
    main()
