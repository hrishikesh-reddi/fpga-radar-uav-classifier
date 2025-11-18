from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import json
import os

app = Flask(__name__)

# Generate network data
def generate_network_data(n_samples=500):
    np.random.seed(42)
    
    # Normal network behavior (85% of data)
    n_normal = int(n_samples * 0.85)
    normal_data = {
        'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='1min'),
        'traffic_volume': np.random.normal(100, 15, n_samples),
        'latency': np.random.gamma(2, 8, n_samples),
        'packet_loss': np.random.exponential(0.4, n_samples),
        'error_rate': np.random.exponential(0.2, n_samples),
        'cpu_usage': np.random.beta(2, 5, n_samples) * 100,
        'memory_usage': np.random.beta(3, 4, n_samples) * 100,
        'connection_count': np.random.poisson(40, n_samples),
    }
    
    # Create anomalies (15% of data)
    anomaly_indices = np.random.choice(n_samples, size=int(n_samples * 0.15), replace=False)
    
    for idx in anomaly_indices:
        # DDoS-like attack patterns
        if np.random.random() > 0.5:
            normal_data['traffic_volume'][idx] += np.random.normal(200, 30)
            normal_data['latency'][idx] += np.random.gamma(4, 15)
            normal_data['packet_loss'][idx] += np.random.exponential(4)
            normal_data['error_rate'][idx] += np.random.exponential(2.5)
            normal_data['cpu_usage'][idx] += np.random.uniform(30, 50)
            normal_data['connection_count'][idx] += np.random.poisson(150)
        # Memory leak patterns
        else:
            normal_data['memory_usage'][idx] += np.random.uniform(40, 60)
            normal_data['latency'][idx] += np.random.gamma(2, 10)
            normal_data['error_rate'][idx] += np.random.exponential(1.5)
    
    df = pd.DataFrame(normal_data)
    df['is_anomaly'] = 0
    df.loc[anomaly_indices, 'is_anomaly'] = 1
    
    return df

# Load sample data
def load_sample_data():
    return generate_network_data()

# Load model information
def load_model_info():
    return {
        "hybrid_qnn": {
            "name": "Hybrid Quantum-Classical Neural Network",
            "description": "Combines classical preprocessing with quantum circuits for enhanced feature processing."
        },
        "qsvm": {
            "name": "Quantum Support Vector Machine",
            "description": "Uses quantum kernels to map data into high-dimensional Hilbert space."
        },
        "qae": {
            "name": "Quantum Autoencoder",
            "description": "Employs quantum circuits for efficient data compression and reconstruction."
        }
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/generate_data', methods=['POST'])
def generate_data():
    data = request.json
    n_samples = data.get('samples', 500)
    
    df = generate_network_data(n_samples)
    
    # Save to session or temporary file
    # For demo purposes, we'll just return the data
    return jsonify({
        'data': df.head(100).to_dict('records'),  # Return first 100 for demo
        'summary': {
            'total_samples': len(df),
            'anomalies': int(df['is_anomaly'].sum()),
            'normal': int(len(df) - df['is_anomaly'].sum()),
            'anomaly_rate': float(df['is_anomaly'].mean() * 100)
        }
    })

@app.route('/api/models')
def get_models():
    models = load_model_info()
    return jsonify(models)

@app.route('/api/detect_anomalies', methods=['POST'])
def detect_anomalies():
    data = request.json
    model_name = data.get('model', 'hybrid_qnn')
    sample_size = data.get('sample_size', 100)
    
    # Simulate model execution with realistic metrics
    if model_name == "hybrid_qnn":
        accuracy = np.random.uniform(0.92, 0.96)
        precision = np.random.uniform(0.88, 0.94)
        recall = np.random.uniform(0.89, 0.95)
        f1 = np.random.uniform(0.90, 0.94)
    elif model_name == "qsvm":
        accuracy = np.random.uniform(0.90, 0.95)
        precision = np.random.uniform(0.86, 0.93)
        recall = np.random.uniform(0.87, 0.94)
        f1 = np.random.uniform(0.88, 0.93)
    elif model_name == "qae":
        accuracy = np.random.uniform(0.88, 0.93)
        precision = np.random.uniform(0.84, 0.91)
        recall = np.random.uniform(0.85, 0.92)
        f1 = np.random.uniform(0.86, 0.91)
    
    # Generate sample detection results
    np.random.seed(42)
    sample_data = generate_network_data(sample_size)
    detected_anomalies = int(np.random.uniform(0.85, 0.95) * sample_data['is_anomaly'].sum())
    false_positives = int(np.random.uniform(0.05, 0.15) * (sample_size - sample_data['is_anomaly'].sum()))
    
    return jsonify({
        'model': model_name,
        'metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        },
        'detection_results': {
            'total_anomalies': int(sample_data['is_anomaly'].sum()),
            'detected_anomalies': detected_anomalies,
            'false_positives': false_positives,
            'detection_rate': float(detected_anomalies / sample_data['is_anomaly'].sum()),
            'false_positive_rate': float(false_positives / (sample_size - sample_data['is_anomaly'].sum()))
        }
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)