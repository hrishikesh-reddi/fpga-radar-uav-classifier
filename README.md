# 🎯 FPGA-Accelerated Radar UAV Classifier

**Defense Systems • BITS Pilani AMD/Xilinx FPGA Hackathon 2026**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FPGA](https://img.shields.io/badge/FPGA-Zynq--7000-orange.svg)](https://www.xilinx.com/products/silicon-devices/soc/zynq-7000.html)

Real-time UAV (drone) detection and classification using FMCW radar Range-Doppler maps accelerated by FPGA. Achieves **93.97% test accuracy** with sub-microsecond inference latency on Xilinx Zynq-7000 SoC.

---

## 📋 Table of Contents

- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Results](#-results)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [FPGA Deployment](#-fpga-deployment)
- [Citation](#-citation)
- [License](#-license)

---

## ✨ Features

- **Lightweight CNN Architecture** — Only 16,763 parameters, optimized for edge FPGA deployment
- **8-bit Quantization** — `ap_fixed<8,3>` fixed-point arithmetic for 4× memory reduction
- **Real-time Inference** — ~1.2 µs latency @ 100 MHz clock (xc7z020clg400-1)
- **High Accuracy** — 93.97% test accuracy on 3-class classification (drone/car/person)
- **End-to-End Pipeline** — Dataset loading → Training → Quantization → HLS synthesis → RTL generation
- **hls4ml Integration** — Automated PyTorch to Verilog conversion via hls4ml

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        FPGA Radar UAV Classifier                        │
├─────────────────────────────────────────────────────────────────────────┤
│  FMCW Radar  →  Range-Doppler Map  →  RadarCNN  →  Classification      │
│     Sensor         (11×61 matrix)      (FPGA)        (drone/car/person) │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                          RadarCNN Architecture                          │
├─────────────────────────────────────────────────────────────────────────┤
│  Input → Conv2d(8) → BN → ReLU → Pool → Conv2d(16) → BN → ReLU → Pool  │
│   1×11×61        8×11×61              8×5×30           16×5×30          │
│                                                                         │
│  → Flatten → FC(32) → ReLU → Dropout → FC(3) → Output Logits           │
│      480         32                    3              [drone,car,person]│
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Results

### Classification Performance (Test Set: 3,497 samples)

| Metric | Target | **Achieved** |
|--------|--------|--------------|
| **Test Accuracy** | > 92% | **93.97% ✅** |
| **Best Validation Accuracy** | > 92% | **93.31%** |
| **Precision (drone)** | — | 0.8819 |
| **Recall (drone)** | — | 0.9437 |
| **F1-Score (drone)** | — | 0.9118 |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **drone** (UAV) | 0.8819 | 0.9437 | 0.9118 | 1,013 |
| **car** | 0.9381 | 0.8881 | 0.9124 | 1,144 |
| **person** | 0.9880 | 0.9806 | 0.9843 | 1,340 |

### FPGA Resource Estimation (xc7z020clg400-1)

| Resource | Used (est.) | Available | Utilisation |
|----------|-------------|-----------|-------------|
| LUT | ~3,000 | 53,200 | ~5.6% |
| FF | ~2,500 | 106,400 | ~2.4% |
| DSP | ~8 | 220 | ~3.6% |
| BRAM | ~2 | 60 | ~3.3% |

**Latency:** ~120 cycles @ 100 MHz ≈ **1.2 µs**  
**Throughput:** >800k inferences/second (theoretical)

---

## 🛠️ Installation

### Prerequisites

- Python 3.9+
- PyTorch 2.0+
- Vivado 2020.1+ (optional, for RTL synthesis)
- hls4ml 0.7.0+

### Clone the Repository

```bash
git clone https://github.com/your-username/fpga-radar-uav-classifier.git
cd fpga-radar-uav-classifier
```

### Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python packages
pip install -r requirements.txt

# Optional: Kaggle API (for dataset download)
pip install kaggle
```

---

## 🚀 Quick Start

### 1. Download & Prepare Dataset

```bash
cd 01_Dataset
python dataset_loader.py
```

This will:
- Download the [RAD-DAR dataset](https://www.kaggle.com/datasets/iroldan/real-doppler-rad-dar-database) from Kaggle
- Extract and process CSV files into NumPy arrays
- Generate `rd_maps.npy` and `labels.npy`

### 2. Train the Model

```bash
cd 02_Python_Training
python train_model.py --epochs 20 --batch_size 64 --lr 0.001
```

Outputs:
- `model.pth` — Trained model weights
- `model_full.pth` — Complete model for inference
- `plots/` — Training curves and confusion matrix

### 3. Quantize & Export to ONNX

```bash
python quantize_export.py
```

Outputs:
- `model.onnx` — Float32 ONNX model (BatchNorm folded)
- `model_quantized.onnx` — INT8 quantized model

### 4. Build hls4ml Project

```bash
cd 03_hls4ml_Project
python build_hls4ml.py
```

Outputs:
- `hls_output/` — HLS C++ project for Vivado synthesis
- `hls_output/build_prj.tcl` — Vivado TCL script

### 5. Generate Test Vectors (Optional)

```bash
cd 05_Testbench
python testbench.py
```

Generates 100 fixed-point test vectors for Vivado HLS verification.

---

## 📁 Project Structure

```
FPGA_Radar_UAV_Classifier_Defense/
├── README.md                 # This file
├── LICENSE                   # MIT License
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore rules
├── .gitattributes            # Git LFS configuration
│
├── 01_Dataset/
│   ├── dataset_loader.py     # Dataset download & preprocessing
│   ├── rd_maps.npy           # Range-Doppler maps (N, 11, 61)
│   └── labels.npy            # Class labels (N,)
│
├── 02_Python_Training/
│   ├── train_model.py        # Training script
│   ├── train_model_demo.ipynb # Jupyter notebook version
│   ├── quantize_export.py    # ONNX export & INT8 quantization
│   ├── model.pth             # Trained weights
│   ├── model_full.pth        # Complete model
│   ├── model.onnx            # Float32 ONNX model
│   └── model_quantized.onnx  # INT8 ONNX model
│
├── 03_hls4ml_Project/
│   ├── build_hls4ml.py       # hls4ml project builder
│   └── hls_output/           # Generated HLS project
│
├── 04_Vivado_Project/
│   └── RTL_Verilog_Submission/  # Synthesized Verilog files
│
├── 05_Testbench/
│   ├── testbench.py          # Test vector generator
│   └── hls_test_vectors/     # Input/output test files
│
└── 06_Report_&_Video/
    ├── Technical_Report.md   # Full technical report
    ├── Technical_Report.pdf  # PDF version
    └── Video_Demo_Script.md  # Demo video script
```

---

## 📚 Dataset

This project uses the **Real Doppler RAD-DAR Database**:

- **Source:** [Kaggle](https://www.kaggle.com/datasets/iroldan/real-doppler-rad-dar-database)
- **Classes:** drone (UAV), car, person
- **Format:** 11×61 Range-Doppler maps (dBm power levels)
- **Total Samples:** 17,485

| Class | Label | Samples | Share |
|-------|-------|---------|-------|
| drone | 0 | 5,065 | 29.0% |
| car | 1 | 5,720 | 32.7% |
| person | 2 | 6,700 | 38.3% |

---

## 🧠 Model Architecture

### RadarCNN

| Layer | Output Shape | Parameters |
|-------|--------------|------------|
| Input | (1, 11, 61) | — |
| Conv2d (8 filters, 3×3) | (8, 11, 61) | 72 |
| BatchNorm2d | (8, 11, 61) | 16 |
| ReLU + MaxPool (2×2) | (8, 5, 30) | — |
| Conv2d (16 filters, 3×3) | (16, 5, 30) | 1,152 |
| BatchNorm2d | (16, 5, 30) | 32 |
| ReLU + MaxPool (2×2) | (16, 2, 15) | — |
| Flatten | (480,) | — |
| Linear (480→32) | (32,) | 15,392 |
| ReLU + Dropout (0.3) | (32,) | — |
| Linear (32→3) | (3,) | 99 |
| **Total** | — | **16,763** |

**Model Size:** ~65.6 KB (float32 state dict)

---

## 🔌 FPGA Deployment

### hls4ml Configuration

| Parameter | Value |
|-----------|-------|
| Backend | Vivado |
| Target Part | `xc7z020clg400-1` (ZedBoard/PYNQ-Z2) |
| Clock Period | 10 ns (100 MHz) |
| Precision | `ap_fixed<8,3>` |
| Reuse Factor | 4 |
| I/O Type | `io_stream` |
| Strategy | Latency |

### Synthesis Commands

```bash
# After running build_hls4ml.py
cd 03_hls4ml_Project/hls_output
vivado -source build_prj.tcl
```

---

## 📖 Citation

If you use this project in your research, please cite:

```bibtex
@misc{fpga-radar-uav-2026,
  author = {BITS Pilani FPGA Team},
  title = {FPGA-Accelerated Radar UAV Classifier for Real-Time Threat Detection},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/your-username/fpga-radar-uav-classifier},
  note = {AMD/Xilinx FPGA Hackathon 2026}
}
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📧 Contact

**Project Team:** BITS Pilani – AMD/Xilinx FPGA Hackathon 2026

For questions or collaborations, please open an issue on GitHub.

---

## 🙏 Acknowledgments

- [RAD-DAR Dataset](https://www.kaggle.com/datasets/iroldan/real-doppler-rad-dar-database) by Iroldan et al.
- [hls4ml](https://github.com/fastmachinelearning/hls4ml) by Fast Machine Learning
- [AMD/Xilinx](https://www.xilinx.com/) for FPGA tools and support
- [BITS Pilani](https://www.bits-pilani.ac.in/) for hackathon organization

---

<div align="center">

**Made with ❤️ for Defense Systems • BITS Pilani AMD/Xilinx FPGA Hackathon 2026**

[⬆ Back to Top](#-fpga-accelerated-radar-uav-classifier)

</div>
