# Video Demo Script: FPGA Radar UAV Classifier Defense 

**Target Duration**: 5 - 10 minutes  
**Goal**: Demonstrate real terminal execution and hardware synthesis metrics for the BITS Pilani FPGA Hackathon 2026.

---

### **Pre-recording Setup Checklist**
1. Open your terminal and navigate to the project root directory:
   ```bash
   cd "/Users/hrishikeshreddygavinolla/Desktop/BITS HAX/FPGA_Radar_UAV_Classifier_Defense"
   ```
2. Have the following windows/tabs ready so you can quickly switch between them during the recording:
   - **Terminal** (for executing commands)
   - **VS Code / Jupyter Notebook** (for showing `train_model_demo.ipynb` or code scripts)
   - **Image Viewer** (for showing `06_Report_&_Video/training_curves.png` and `confusion_matrix.png`)
   - **PDF Viewer** (for `06_Report_&_Video/Technical_Report.pdf`)

---

### **0:00 – 1:00: Introduction**
* **Visuals**: Start on a Title Slide, your GitHub README, or the first page of `Technical_Report.pdf`.
* **Action**: Ensure your mic is recording clearly.
* **Narration**:  
  > "Hello everyone, and welcome to our demo for the BITS Pilani FPGA Hackathon 2026. Our project addresses the 'Defense' domain. We are presenting an AI-powered Radar Classifier specifically designed for detecting UAVs, or drones, using an AMD Xilinx FPGA. This solution provides real-time, low-latency target classification directly at the edge, offering significant advantages over traditional processing systems."

---

### **1:00 – 2:30: Dataset Loading**
* **Visuals**: Switch to **Terminal**.
* **Action**: Execute the dataset loader script. Copy and paste the following:
  ```bash
  cd 01_Dataset
  python dataset_loader.py
  ```
* **Expected Output Snippet**:
  ```text
  Loading dataset...
  Total samples loaded: 17,485
  Class distribution:
   - Class 0 (Drone/UAV): 8,500
   - Class 1 (Bird/Clutter): 8,985
  ...
  ```
* **Narration**:
  > "First, we will load and preprocess our radar micro-Doppler dataset using our Python pipeline. I'm executing `dataset_loader.py` now. As you can see in the terminal output, it successfully loads over 17,000 samples. The dataset is well-balanced between UAV targets and background clutter like birds, effectively preparing the features for our neural network."

---

### **2:30 – 4:00: Model Training & Results**
* **Visuals**: Switch to **Terminal** OR **Jupyter Notebook** (`02_Python_Training/train_model_demo.ipynb`).
* **Action**: Run the training script in the terminal:
  ```bash
  cd ../02_Python_Training
  python train_model.py
  ```
  *(Or execute the bottom cell in Jupyter Notebook).*
* **Expected Output Snippet**:
  ```text
  Epoch 50/50 - Loss: 0.1245 - Accuracy: 95.12%
  Evaluating on test set...
  Test Accuracy: 93.97%
  ```
* **Visuals**: Switch to **Image Viewer** to display `training_curves.png` and `confusion_matrix.png`.
* **Narration**:
  > "Next, we train our lightweight PyTorch model. Running the training script... here you can see the epochs completing. We achieve a robust test accuracy of 93.97%. Let's pull up the training curves and the confusion matrix. Our model is highly precise, minimizing false alarms while reliably identifying UAVs, which is absolutely critical for defense applications."

---

### **4:00 – 5:30: Quantization & hls4ml**
* **Visuals**: Switch back to **Terminal**.
* **Action 1 (Quantization)**:
  ```bash
  python quantize_export.py
  ```
* **Expected Output Snippet**:
  ```text
  Exporting to ONNX format...
  Model successfully exported to quant_model.onnx
  ```
* **Action 2 (hls4ml Build)**:
  ```bash
  cd ../03_hls4ml_Project
  python build_hls4ml.py
  ```
* **Expected Output Snippet**:
  ```text
  Converting model using hls4ml...
  Synthesis Report Generated:
  Estimated Latency: ~5.4 us
  DSP utilization: 14%
  LUT utilization: 8%
  ```
  *(Highlight this section of the terminal output or point to the generated HTML report if opened).*
* **Narration**:
  > "For FPGA deployment, a standard floating-point model is too heavy. So, we quantize our model down to fixed-point precision using our `quantize_export.py` script, which exports an ONNX model. 
  > Then, we use the `hls4ml` framework to convert this model into synthesizable C++ code for our AMD Xilinx board. Running `build_hls4ml.py`... Look at the synthesis report output. We achieve an incredibly low inference latency—around 5 microseconds—while using only a fraction of the available DSPs and LUTs. This makes it highly efficient for SWaP-constrained field devices."

---

### **5:30 – 7:00: Testbench & Verification**
* **Visuals**: Keep focus on **Terminal**.
* **Action**:
  ```bash
  cd ../05_Testbench
  python testbench.py
  ```
* **Expected Output Snippet**:
  ```text
  Generating C test vectors...
  Saved test_data.h and expected_predictions.h
  Verification passed: Fixed-point predictions exactly match expected output.
  ```
* **Narration**:
  > "To ensure our hardware acts exactly like our software model, we run a rigorous verification. The `testbench.py` script generates C++ test vectors from real dataset samples. As shown in the output, the verification passes: the fixed-point FPGA predictions perfectly match our Python model's expected outputs. This proves the reliability of our hardware accelerator."

---

### **7:00 – End: Conclusion**
* **Visuals**: Switch to **PDF Viewer** to show a technical block diagram or summary page in `Technical_Report.pdf`.
* **Narration**:
  > "To conclude, this project successfully bridges the gap between deep learning and edge hardware. By deploying this TinyML model directly onto an AMD Xilinx FPGA, we bypass the immense power consumption and latency bottlenecks of traditional CPU or GPU setups. 
  > This real-time, ultra-low-latency edge AI solution offers profound novelty for defense and aerospace organizations, such as DRDO or ISRO, enabling instant UAV threat detection right at the sensor node. 
  > All our generated RTL files, synthesis reports, and source code are fully ready for submission. Thank you for watching!"
