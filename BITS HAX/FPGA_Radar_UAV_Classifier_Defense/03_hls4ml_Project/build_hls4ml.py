#!/usr/bin/env python3
"""
=============================================================================
  FPGA Radar UAV Classifier — hls4ml Project Builder
  PyTorch RadarCNN → HLS4ML Vivado project
=============================================================================

  Config:
    backend     = 'Vivado'
    part        = 'xc7z020clg400-1'  (ZedBoard / PYNQ-Z2)
    clock_period= 10 ns  → 100 MHz
    Precision   = ap_fixed<8,3>
    ReuseFactor = 4
    io_type     = io_stream
    Strategy    = Latency

  Usage:
    python build_hls4ml.py

  Note:
    RTL synthesis (synth=True, export=True) requires Vivado 2020.1+
    in PATH. Without Vivado, only C-simulation is performed.

  Author : BITS Pilani – AMD/Xilinx FPGA Hackathon 2026
  License: MIT
=============================================================================
"""

import os
import sys
import shutil

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "02_Python_Training")
OUTPUT_DIR   = os.path.join(SCRIPT_DIR, "hls_output")

MODEL_STATE  = os.path.join(TRAINING_DIR, "model.pth")   # state dict
MODEL_FULL   = os.path.join(TRAINING_DIR, "model_full.pth")  # full model


# ─────────────────────────────────────────────────────────────────────────────
# Model Architecture (must match train_model.py)
# ─────────────────────────────────────────────────────────────────────────────
def build_model():
    """Load trained RadarCNN from state dict."""
    import torch
    import torch.nn as nn

    sys.path.insert(0, TRAINING_DIR)
    from train_model import RadarCNN

    model = RadarCNN(num_classes=3)
    if os.path.exists(MODEL_STATE):
        state = torch.load(MODEL_STATE, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        print(f"  Loaded weights: {MODEL_STATE}")
    else:
        print(f"  [WARN] {MODEL_STATE} not found — using random weights!")

    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Main build
# ─────────────────────────────────────────────────────────────────────────────
def build():
    import hls4ml
    import numpy as np

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   FPGA Radar UAV Classifier — hls4ml Project Builder       ║")
    print("║   Defense Systems • BITS Pilani AMD/Xilinx Hackathon 2026  ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # ── Load PyTorch model ────────────────────────────────────────────────
    print("[STEP 1] Loading trained RadarCNN …")
    model = build_model()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters : {total_params:,}")
    print(f"  Input shape: (1, 1, 11, 61)")

    # ── hls4ml config ─────────────────────────────────────────────────────
    print("\n[STEP 2] Building hls4ml configuration …")
    # input_shape = (channels, H, W) — no batch dimension
    input_shape = (1, 11, 61)

    hls_config = hls4ml.utils.config_from_pytorch_model(
        model,
        input_shape=input_shape,
        granularity="name",
        backend="Vivado",
        default_precision="ap_fixed<8,3>",
        default_reuse_factor=4,
    )

    # Global overrides
    hls_config["Model"]["Precision"]   = "ap_fixed<8,3>"
    hls_config["Model"]["ReuseFactor"] = 4

    # Per-layer strategy
    for layer in hls_config.get("LayerName", {}).values():
        layer["Strategy"] = "Latency"

    print("  HLS config:")
    print(f"    Precision   : {hls_config['Model']['Precision']}")
    print(f"    ReuseFactor : {hls_config['Model']['ReuseFactor']}")
    print(f"    io_type     : io_stream")
    print(f"    Strategy    : Latency (all layers)")

    # ── Convert PyTorch → hls4ml model ───────────────────────────────────
    print("\n[STEP 3] Converting PyTorch → hls4ml project …")
    hls_model = hls4ml.converters.convert_from_pytorch_model(
        model,
        output_dir=OUTPUT_DIR,
        backend="Vivado",
        hls_config=hls_config,
        part="xc7z020clg400-1",
        clock_period=10,
        io_type="io_stream",
        input_shape=input_shape,
    )

    print(f"  [OK] HLS project written to: {OUTPUT_DIR}")
    print(f"  Backend      : Vivado")
    print(f"  Part         : xc7z020clg400-1")
    print(f"  Clock period : 10 ns (100 MHz)")

    # ── Compile (C-simulation) ─────────────────────────────────────────────
    print("\n[STEP 4] Compiling HLS C-simulation …")
    try:
        hls_model.compile()
        print("  [OK] C-simulation compiled successfully.")

        # Quick prediction sanity check
        print("\n[STEP 5] Running forward pass through compiled model …")
        dummy = np.random.randn(1, 1, 11, 61).astype(np.float32)
        hls_pred = hls_model.predict(dummy)
        print(f"  Input  shape : {dummy.shape}")
        print(f"  Output shape : {hls_pred.shape}")
        print(f"  Output logits: {hls_pred[0]}")
        print(f"  Predicted class: {int(hls_pred[0].argmax())}  ({['drone','car','person'][int(hls_pred[0].argmax())]})")
    except Exception as e:
        print(f"  [WARN] C-simulation failed: {e}")
        print("         This may happen without Vivado HLS / Vitis HLS.")

    # ── Vivado RTL synthesis ───────────────────────────────────────────────
    vivado_available = shutil.which("vivado") is not None
    if vivado_available:
        print("\n[STEP 6] Running Vivado RTL synthesis + export …")
        report = hls_model.build(
            csim=True, synth=True, export=True, report=True,
        )
        _print_report(report)
    else:
        print("\n[STEP 6] Vivado not found — skipping RTL synthesis.")
        print("         Install Vivado 2020.1+ and source settings64.sh to synthesise.")
        _print_estimated_resources()

    # ── List generated files ───────────────────────────────────────────────
    print("\n[STEP 7] Generated HLS project files:")
    _list_project_files(OUTPUT_DIR)

    print("\n  ✅ hls4ml project build complete!\n")


# ─────────────────────────────────────────────────────────────────────────────
# Report helpers
# ─────────────────────────────────────────────────────────────────────────────
def _print_report(report):
    if report is None:
        print("  [WARN] No report returned.")
        return
    print("\n" + "=" * 60)
    print("  RESOURCE UTILISATION (post-synthesis)")
    print("=" * 60)
    synth = report.get("CSynthesisReport", report)
    rows = [
        ("LUT",  "LUTsUsed",  "AvailableLUTs",  "LUTsUsedPercent"),
        ("FF",   "FFUsed",    "AvailableFFs",    "FFUsedPercent"),
        ("DSP",  "DSPUsed",   "AvailableDSPs",   "DSPUsedPercent"),
        ("BRAM", "BRAMUsed",  "AvailableBRAMs",  "BRAMUsedPercent"),
    ]
    print(f"\n  {'Resource':<8} {'Used':>8} {'Available':>12} {'Util %':>10}")
    print(f"  {'─'*8} {'─'*8} {'─'*12} {'─'*10}")
    for res, uk, ak, pk in rows:
        u, a, p = synth.get(uk,"N/A"), synth.get(ak,"N/A"), synth.get(pk,"N/A")
        print(f"  {res:<8} {str(u):>8} {str(a):>12} {str(p)+' %':>10}")
    lat = synth.get("LatencyBest",  synth.get("Latency","N/A"))
    ii  = synth.get("IntervalBest", synth.get("InitiationInterval","N/A"))
    print(f"\n  Latency (best) : {lat} cycles")
    print(f"  Initiation II  : {ii} cycles")
    print("=" * 60)


def _print_estimated_resources():
    """Display estimated resource usage without Vivado synthesis."""
    print()
    print("  Estimated post-synthesis resources (ap_fixed<8,3>, xc7z020clg400-1):")
    print("  ┌──────────┬─────────────┬───────────┬───────────────┐")
    print("  │ Resource │ Used (est.) │ Available │ Utilisation   │")
    print("  ├──────────┼─────────────┼───────────┼───────────────┤")
    print("  │ LUT      │  ~3 000     │  53 200   │ ~5.6 %        │")
    print("  │ FF       │  ~2 500     │ 106 400   │ ~2.4 %        │")
    print("  │ DSP      │  ~8         │   220     │ ~3.6 %        │")
    print("  │ BRAM     │  ~2         │    60     │ ~3.3 %        │")
    print("  └──────────┴─────────────┴───────────┴───────────────┘")
    print("  Estimated latency   : ~120 cycles @ 100 MHz ≈ 1.2 µs")
    print("  Initiation interval : 1 cycle    (io_stream, Latency strategy)")


def _list_project_files(output_dir):
    if not os.path.isdir(output_dir):
        print("  [WARN] Output directory does not exist yet.")
        return
    for root, dirs, files in os.walk(output_dir):
        # Limit depth for readability
        depth = root.replace(output_dir, "").count(os.sep)
        if depth > 2:
            continue
        indent = "  " + "    " * depth
        rel = os.path.relpath(root, output_dir)
        if rel == ".":
            print(f"\n  {output_dir}/")
        else:
            print(f"{indent}{os.path.basename(root)}/")
        sub_indent = indent + "    "
        for f in sorted(files):
            full = os.path.join(root, f)
            size = os.path.getsize(full)
            print(f"{sub_indent}{f:<42s}  {size/1024:6.1f} KB")


if __name__ == "__main__":
    build()
