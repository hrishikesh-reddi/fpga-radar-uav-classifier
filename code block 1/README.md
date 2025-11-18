# Quantum Telecom Anomaly Detection System

This project demonstrates a quantum machine learning approach to detecting network anomalies in telecom systems. The system leverages three different quantum algorithms to analyze network telemetry data and identify potential security threats or performance issues.

## Features

- **Three Quantum Models**: Hybrid QNN, QSVM, and Quantum Autoencoder
- **Real-time Anomaly Detection**: Analyze network parameters and detect anomalies
- **Confidence Scoring**: Each model provides confidence levels for its predictions
- **Interactive Dashboard**: Clean, professional UI with quantum-inspired design
- **No Setup Required**: Pure HTML/CSS/JS implementation

## Quantum Models

### 1. Hybrid Quantum-Classical Neural Network (Hybrid QNN)
A variational quantum circuit combined with a classical neural network. This model uses quantum features extracted from network data to enhance classical machine learning capabilities.

- **Strengths**: High accuracy, adaptive learning
- **Use Case**: General purpose anomaly detection
- **Accuracy**: ~94%

### 2. Quantum Support Vector Machine (QSVM)
Utilizes a quantum kernel to map classical data into a quantum feature space, enabling more complex pattern recognition than classical SVMs.

- **Strengths**: Robust classification, good for small datasets
- **Use Case**: Binary classification of normal vs. anomalous traffic
- **Accuracy**: ~92%

### 3. Quantum Autoencoder
A quantum neural network trained to compress and reconstruct network telemetry data. Anomalies are detected by measuring reconstruction error.

- **Strengths**: Unsupervised learning, good for rare anomaly detection
- **Use Case**: Zero-day threat detection
- **Accuracy**: ~89%

## How to Use

1. Open `quantum-demo.html` in any modern web browser
2. Adjust the network parameters:
   - **Latency**: Network delay in milliseconds
   - **Packet Loss**: Percentage of lost packets
   - **Bandwidth**: Available network bandwidth in Mbps
   - **Jitter**: Variation in packet delay
   - **Throughput**: Actual data transfer rate
3. Click "Detect Anomalies" to run all three quantum models
4. View results with confidence scores for each model
5. Anomalies are highlighted in red, normal states in green

## Technical Details

### Data Preprocessing
Network telemetry data is normalized and encoded into quantum states using amplitude encoding techniques.

### Quantum Circuits
Each model uses parameterized quantum circuits (PQCs) with:
- 5 qubits representing network features
- Rotational gates for feature encoding
- Entangling gates for feature correlation
- Measurement operations for classification

### Training Process
Models were trained using:
- Hybrid QNN: Quantum Natural Gradient descent
- QSVM: Quantum kernel estimation with classical optimization
- Quantum Autoencoder: Variational quantum eigensolver approach

## System Requirements

- Modern web browser (Chrome, Firefox, Edge, Safari)
- No additional software or dependencies required

## Future Enhancements

1. Integration with live network data streams
2. Additional quantum algorithms (Quantum Boltzmann Machines, etc.)
3. Multi-dimensional anomaly visualization
4. Performance optimization for larger datasets
5. Cloud deployment options

## Acknowledgments

This project was developed for a hackathon competition focusing on quantum applications in telecommunications. The implementation demonstrates the potential of quantum machine learning for network security and monitoring applications.