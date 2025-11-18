# Quantum Network Anomaly Detection

This project demonstrates the application of Quantum Machine Learning (QML) techniques for detecting anomalies in telecom networks.

## 📁 Project Structure

```
telecom_anomaly_detection/
├── data/
│   └── quantum_data.npz          # Quantum-ready dataset
├── models/
│   ├── hybrid_qnn_model.pth      # Trained Hybrid QNN model
│   ├── qsvm_model.pkl            # Trained QSVM model
│   └── qsvm_model_complete.pkl   # Complete QSVM model with support vectors
├── results/
│   └── data_overview.png         # Data visualization
├── web_app/
│   ├── app.py                    # Streamlit web application
│   └── requirements.txt          # Python dependencies
├── data_generator.py             # Data generation script
├── hybrid_qnn.py                 # Hybrid Quantum-Classical Neural Network
├── qsvm.py                       # Quantum Support Vector Machine
└── qae.py                        # Quantum Autoencoder
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Navigate to the project directory:
   ```bash
   cd telecom_anomaly_detection
   ```

2. Install the required dependencies:
   ```bash
   pip install -r web_app/requirements.txt
   ```

### Running the Web Application

1. Start the Streamlit application:
   ```bash
   streamlit run web_app/app.py
   ```

2. The application will open in your default web browser at `http://localhost:8501`

## 🧠 Models Overview

### Hybrid Quantum-Classical Neural Network (Hybrid QNN)
A neural network that combines classical preprocessing with quantum circuits for enhanced feature processing.

### Quantum Support Vector Machine (QSVM)
A support vector machine that uses a quantum kernel to map data into high-dimensional Hilbert space.

### Quantum Autoencoder (QAE)
An autoencoder that uses quantum circuits for efficient data compression and reconstruction.

## 📊 Web Application Features

The web application provides:

1. **Dashboard View**: Real-time network traffic visualization
2. **Model Information**: Detailed descriptions of each quantum model
3. **Performance Metrics**: Accuracy, precision, recall, and F1-score
4. **Interactive Controls**: Model selection and parameter tuning
5. **Data Visualization**: Feature distribution and anomaly detection results

## 🏆 Hackathon Goals

This project aims to demonstrate:

- Practical quantum advantage in network security
- Real-world applicability of Quantum Machine Learning
- A foundation for future quantum network monitoring systems

## 🚀 Future Enhancements

Planned improvements include:

- Integration with real-time network data streams
- Advanced quantum circuits with more qubits
- Ensemble methods combining all three approaches
- Deployment on actual quantum hardware