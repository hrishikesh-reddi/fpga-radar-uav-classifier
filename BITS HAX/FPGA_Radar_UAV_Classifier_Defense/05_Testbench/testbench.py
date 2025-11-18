import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

# --- Configuration ---
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR  = os.path.join(os.path.dirname(SCRIPT_DIR), "01_Dataset")
PYTHON_TRAINING_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "02_Python_Training")

MAPS_PATH    = os.path.join(DATASET_DIR, "rd_maps.npy")
LABELS_PATH  = os.path.join(DATASET_DIR, "labels.npy")
MODEL_PATH   = os.path.join(PYTHON_TRAINING_DIR, "model_full.pth") # Full model for inference

NUM_TEST_SAMPLES = 100
INPUT_H, INPUT_W = 11, 61  # Range × Doppler
NUM_CLASSES      = 3
CLASS_NAMES      = ["drone", "car", "person"]

# Output paths for Vivado-compatible test vectors
HLS_INPUT_DIR  = os.path.join(SCRIPT_DIR, "hls_test_vectors", "input")
HLS_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "hls_test_vectors", "output")
os.makedirs(HLS_INPUT_DIR, exist_ok=True)
os.makedirs(HLS_OUTPUT_DIR, exist_ok=True)

# Fixed-point configuration for HLS input (e.g., ap_fixed<16,6>)
# This means 16 total bits, 6 integer bits, 10 fractional bits.
# Max representable value: 2^5 + (2^10-1)/2^10 = 31.999...
# Min representable value: -2^5 = -32
FIXED_POINT_INT_BITS = 6
FIXED_POINT_FRAC_BITS = 10
FIXED_POINT_TOTAL_BITS = FIXED_POINT_INT_BITS + FIXED_POINT_FRAC_BITS


# --- Model Architecture (must match train_model.py) ---
class RadarCNN(nn.Module):
    def __init__(self, num_classes: int = 3):
        super(RadarCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc1     = nn.Linear(16 * 2 * 15, 32) # (11/2/2) = 2, (61/2/2) = 15
        self.relu3   = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2     = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# --- Helper for fixed-point conversion ---
def float_to_fixed(value, int_bits, frac_bits):
    """
    Converts a float to a fixed-point integer representation.
    Assumes signed fixed-point.
    """
    scale = 2**frac_bits
    max_val = 2**(int_bits - 1) - (1 / scale)
    min_val = -2**(int_bits - 1)

    # Clamp the value to the representable range
    clipped_value = np.clip(value, min_val, max_val)

    scaled_value = int(round(clipped_value * scale))

    # Ensure it fits within the total bits (e.g., 16 bits for ap_fixed<16,6>)
    # This handles potential overflow from rounding for the most negative number
    total_bits_mask = (1 << (int_bits + frac_bits)) - 1
    return scaled_value & total_bits_mask


# --- Main Testbench Logic ---
def main():
    print("FPGA Radar UAV Classifier — Vivado HLS Testbench Generator")
    print("-" * 60)

    # 1. Load Data
    print(f"[INFO] Loading dataset from {DATASET_DIR}...")
    try:
        rd_maps = np.load(MAPS_PATH)   # (N, 11, 61) float32
        labels  = np.load(LABELS_PATH) # (N,) int64
    except FileNotFoundError:
        print(f"[ERROR] Dataset files not found. Make sure {MAPS_PATH} and {LABELS_PATH} exist.")
        print("        Run 01_Dataset/dataset_loader.py first.")
        return

    print(f"  Loaded {len(rd_maps)} samples.")

    # Use a fixed random seed for reproducibility in sample selection and splitting
    # This is important to ensure the test samples are consistent.
    # Note: For strict test set usage, we'd replicate the train_test_split from train_model.py.
    # For this testbench, we'll take a simple random sample from the whole dataset.
    np.random.seed(42)
    
    # Randomly select N samples from the dataset
    if len(rd_maps) < NUM_TEST_SAMPLES:
        print(f"[WARN] Not enough samples ({len(rd_maps)}) for {NUM_TEST_SAMPLES} test vectors. Using all available.")
        indices = np.arange(len(rd_maps))
    else:
        indices = np.random.choice(len(rd_maps), NUM_TEST_SAMPLES, replace=False)

    selected_rd_maps = rd_maps[indices]
    selected_labels  = labels[indices]

    # 2. Load PyTorch Model for Expected Outputs
    print(f"[INFO] Loading trained PyTorch model from {MODEL_PATH}...")
    try:
        device = torch.device("cpu") # Run inference on CPU
        model = RadarCNN(num_classes=NUM_CLASSES).to(device)
        # Using map_location=device is crucial if model was trained on GPU but inferring on CPU
        model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model.eval() # Set to evaluation mode
        print("  Model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Could not load model: {e}")
        print("        Make sure 02_Python_Training/train_model.py has been run to create model_full.pth.")
        return

    # 3. Pre-process Samples and Generate Expected Outputs
    print(f"[INFO] Pre-processing {len(selected_rd_maps)} samples and generating expected outputs...")
    hls_inputs = []
    expected_outputs = []
    
    with torch.no_grad():
        for i, (rd_map, true_label) in enumerate(zip(selected_rd_maps, selected_labels)):
            # Apply per-sample normalization (mean 0, std 1)
            mean = rd_map.mean()
            std  = rd_map.std() + 1e-8 # Add epsilon for stability
            normalized_map = (rd_map - mean) / std

            # Add channel dimension: (11, 61) -> (1, 1, 11, 61) for PyTorch model
            model_input = torch.tensor(normalized_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            
            # Get expected output from PyTorch model
            output_logits = model(model_input)
            _, predicted_label = torch.max(output_logits, 1)
            
            hls_inputs.append(normalized_map.flatten()) # Flatten for HLS input file
            expected_outputs.append(predicted_label.item())

            # Save HLS input vector
            input_filename = os.path.join(HLS_INPUT_DIR, f"input_{i}.txt")
            with open(input_filename, "w") as f:
                # Write each float value converted to fixed-point integer representation
                # This assumes the HLS side will interpret these as ap_fixed<16,6> or similar
                for val in hls_inputs[-1]:
                    f.write(f"{float_to_fixed(val, FIXED_POINT_INT_BITS, FIXED_POINT_FRAC_BITS)}\n")
            
            # Save HLS expected output (ground truth label)
            output_filename = os.path.join(HLS_OUTPUT_DIR, f"output_{i}.txt")
            with open(output_filename, "w") as f:
                f.write(f"{predicted_label.item()}\n")
    
    print(f"  Generated {len(hls_inputs)} HLS input and output test vector files.")
    print(f"  Input format: {INPUT_H}x{INPUT_W} flattened array, each value converted to signed fixed-point (e.g., ap_fixed<{FIXED_POINT_TOTAL_BITS},{FIXED_POINT_INT_BITS}>).")
    print(f"  Output format: Single integer representing the predicted class label (0, 1, or 2).\n")

    # 4. Describe Vivado HLS Testbench Structure (conceptual)
    print("--- Vivado HLS Testbench Guidance ---")
    print("To use these test vectors in Vivado HLS, you will typically:")
    print("1. Create a C/C++ testbench function (e.g., `test_model_hls.cpp`).")
    print("2. Inside this testbench, read the generated `input_*.txt` files.")
    print("3. Convert the integer values read from the input files back to `ap_fixed` types.")
    print("4. Call your hls4ml-generated C/C++ model function with these `ap_fixed` inputs.")
    print("5. Read the generated `output_*.txt` files (expected outputs).")
    print("6. Compare the model's output with the expected output and report accuracy.")
    print("\nExample C-style pseudocode for reading input and comparing output:")
    print("```cpp")
    print(f'#include "{os.path.join("..", "..", "03_hls4ml_Project", "hls_model", "hls_model.h")}" // Assuming hls4ml output')
    print("#include <fstream>")
    print("#include <iostream>")
    print("#include <vector>")
    print("#include <string>")
    print("")
    print("int main() {")
    print(f"    static input_t test_input[{INPUT_H} * {INPUT_W}]; // input_t will be ap_fixed<{FIXED_POINT_TOTAL_BITS},{FIXED_POINT_INT_BITS}>")
    print("    static result_t test_output[N_CLASSES];    // result_t will be the model's output type (e.g., ap_int<...>)")
    print("    int expected_label;")
    print("    int correct_predictions = 0;")
    print(f"    for (int i = 0; i < {len(hls_inputs)}; ++i) {{")
    print(f"        std::ifstream input_file(\"{HLS_INPUT_DIR}/input_\" + std::to_string(i) + \".txt\");")
    print(f"        std::ifstream output_file(\"{HLS_OUTPUT_DIR}/output_\" + std::to_string(i) + \".txt\");")
    print("        if (!input_file.is_open() || !output_file.is_open()) {")
    print("            std::cerr << \"Error opening test vector files!\" << std::endl; return 1;")
    print("        }")
    print("")
    print("        // Read input values and convert back to ap_fixed")
    print(f"        for (int j = 0; j < {INPUT_H * INPUT_W}; ++j) {{")
    print("            long long fixed_val;")
    print("            input_file >> fixed_val;")
    print("            test_input[j] = reinterpret_cast<input_t&>(fixed_val); // Dangerous, better way: input_t(fixed_val) / (1 << FRAC_BITS) ")
    print("        }")
    print("        output_file >> expected_label;")
    print("")
    print("        // Call the hls4ml model")
    print(f"        hls_model(test_input, test_output); // Replace hls_model with your actual top-level function name")
    print("")
    print("        // Determine predicted label from model output (e.g., argmax for logits)")
    print("        int predicted_label = 0;")
    print("        // Find index of max value in test_output")
    print(f"        for (int k = 1; k < {NUM_CLASSES}; ++k) {{")
    print("            if (test_output[k] > test_output[predicted_label]) {")
    print("                predicted_label = k;")
    print("            }")
    print("        }")
    print("")
    print("        if (predicted_label == expected_label) {")
    print("            correct_predictions++;")
    print("        }")
    print("        std::cout << \"Sample \" << i << \": Predicted \" << predicted_label << \", Expected \" << expected_label << std::endl;")
    print("    }")
    print(f"    std::cout << \"Accuracy: \" << (double)correct_predictions / {len(hls_inputs)} * 100 << \"%\" << std::endl;")
    print("    return 0;")
    print("}")
    print("```")
    print("-" * 60)
    print("\n✅ Vivado HLS testbench files generated and guidance provided.")


if __name__ == "__main__":
    main()
