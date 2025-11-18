# Data Generation & Preprocessing for Quantum Network Anomaly Detection
# Run this first to generate training data

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Set seed for reproducibility
np.random.seed(42)

print("🔧 Generating Network Anomaly Dataset...")

# ========================================
# PART 1: GENERATE SYNTHETIC NETWORK DATA
# ========================================

def generate_network_data(n_samples=2000, anomaly_ratio=0.15):
    """
    Generate realistic network traffic data with anomalies
    """
    
    n_anomalies = int(n_samples * anomaly_ratio)
    n_normal = n_samples - n_anomalies
    
    # Normal network behavior (80% of data)
    normal_data = {
        'traffic_volume_mbps': np.random.normal(100, 20, n_normal),
        'latency_ms': np.random.gamma(2, 10, n_normal),  # Right-skewed
        'packet_loss_pct': np.random.exponential(0.5, n_normal),
        'error_rate_pct': np.random.exponential(0.3, n_normal),
        'throughput_mbps': np.random.normal(95, 15, n_normal),
        'cpu_usage_pct': np.random.beta(3, 5, n_normal) * 100,
        'memory_usage_pct': np.random.beta(4, 4, n_normal) * 100,
        'connection_count': np.random.poisson(50, n_normal),
    }
    
    # Anomalous patterns (20% of data)
    anomaly_data = {
        # DDoS-like: High traffic, high latency, packet loss
        'traffic_volume_mbps': np.random.normal(300, 50, n_anomalies),
        'latency_ms': np.random.gamma(5, 30, n_anomalies),
        'packet_loss_pct': np.random.exponential(5, n_anomalies),
        'error_rate_pct': np.random.exponential(3, n_anomalies),
        'throughput_mbps': np.random.normal(40, 20, n_anomalies),
        'cpu_usage_pct': np.random.beta(8, 2, n_anomalies) * 100,
        'memory_usage_pct': np.random.beta(7, 2, n_anomalies) * 100,
        'connection_count': np.random.poisson(200, n_anomalies),
    }
    
    # Create DataFrames
    df_normal = pd.DataFrame(normal_data)
    df_normal['label'] = 0
    
    df_anomaly = pd.DataFrame(anomaly_data)
    df_anomaly['label'] = 1
    
    # Combine and shuffle
    df = pd.concat([df_normal, df_anomaly], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Add timestamp
    start_time = datetime.now() - timedelta(hours=n_samples//60)
    df['timestamp'] = [start_time + timedelta(minutes=i) for i in range(n_samples)]
    
    return df

# Generate dataset
df = generate_network_data(n_samples=2000, anomaly_ratio=0.15)

print(f"✅ Generated {len(df)} samples")
print(f"   Normal samples: {sum(df['label']==0)}")
print(f"   Anomaly samples: {sum(df['label']==1)}")
print(f"   Anomaly ratio: {sum(df['label']==1)/len(df):.2%}")

# ========================================
# PART 2: FEATURE ENGINEERING
# ========================================

def engineer_features(df):
    """Create advanced features for better quantum encoding"""
    
    df = df.copy()
    
    # Temporal features
    df['hour'] = df['timestamp'].dt.hour
    df['is_peak_hour'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
    
    # Ratio features (critical for anomaly detection)
    df['latency_per_traffic'] = df['latency_ms'] / (df['traffic_volume_mbps'] + 1)
    df['error_to_packet_loss'] = df['error_rate_pct'] / (df['packet_loss_pct'] + 0.01)
    df['resource_pressure'] = df['cpu_usage_pct'] * df['memory_usage_pct'] / 10000
    df['network_efficiency'] = df['throughput_mbps'] / (df['latency_ms'] + 1)
    
    # Clip outliers
    for col in df.select_dtypes(include=[np.number]).columns:
        if col not in ['label', 'hour', 'is_peak_hour']:
            q99 = df[col].quantile(0.99)
            df[col] = df[col].clip(upper=q99)
    
    return df

df_enhanced = engineer_features(df)

print("\n🔬 Feature Engineering Complete")
print(f"   Original features: 8")
print(f"   Enhanced features: {df_enhanced.shape[1] - 2}")  # -2 for timestamp and label

# ========================================
# PART 3: QUANTUM-READY DATA PREPARATION
# ========================================

# Select features for modeling
feature_cols = [col for col in df_enhanced.columns 
                if col not in ['timestamp', 'label']]

X = df_enhanced[feature_cols].values
y = df_enhanced['label'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"\n📊 Train-Test Split:")
print(f"   Training samples: {len(X_train)}")
print(f"   Testing samples: {len(X_test)}")

# ========================================
# PART 4: QUANTUM OPTIMIZATION
# ========================================

def prepare_quantum_data(X_train, y_train, X_test, y_test, 
                         n_qubits=4, n_samples=1000):
    """
    Optimize data for quantum models:
    1. Dimensionality reduction to match qubit count
    2. Strategic sampling for balanced training
    3. Normalization to [-π, π] for angle encoding
    """
    
    print(f"\n⚛️  Preparing Quantum-Ready Data (n_qubits={n_qubits})...")
    
    # Step 1: Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Step 2: PCA to reduce to n_qubits dimensions
    pca = PCA(n_components=n_qubits)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    print(f"   ✓ PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    
    # Step 3: Strategic sampling (balanced classes)
    normal_idx = np.where(y_train == 0)[0]
    anomaly_idx = np.where(y_train == 1)[0]
    
    n_per_class = min(n_samples // 2, len(anomaly_idx))
    
    selected_normal = np.random.choice(normal_idx, n_per_class, replace=False)
    selected_anomaly = np.random.choice(anomaly_idx, n_per_class, replace=False)
    
    selected_idx = np.concatenate([selected_normal, selected_anomaly])
    np.random.shuffle(selected_idx)
    
    X_train_sampled = X_train_pca[selected_idx]
    y_train_sampled = y_train[selected_idx]
    
    # Step 4: Scale to [-π, π] for angle encoding
    X_train_quantum = np.clip(X_train_sampled, -3, 3) * (np.pi / 3)
    X_test_quantum = np.clip(X_test_pca, -3, 3) * (np.pi / 3)
    
    print(f"   ✓ Sampled {len(X_train_quantum)} balanced training samples")
    print(f"   ✓ Normal: {sum(y_train_sampled==0)}, Anomaly: {sum(y_train_sampled==1)}")
    print(f"   ✓ Data range: [{X_train_quantum.min():.2f}, {X_train_quantum.max():.2f}]")
    
    return {
        'X_train': X_train_quantum,
        'y_train': y_train_sampled,
        'X_test': X_test_quantum,
        'y_test': y_test,
        'scaler': scaler,
        'pca': pca,
        'feature_names': feature_cols
    }

# Prepare quantum data
quantum_data = prepare_quantum_data(
    X_train, y_train, X_test, y_test,
    n_qubits=4,
    n_samples=1000  # Fast training
)

# ========================================
# PART 5: VISUALIZATION & EXPORT
# ========================================

print("\n📈 Generating Visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Class distribution
axes[0, 0].bar(['Normal', 'Anomaly'], 
               [sum(y==0), sum(y==1)],
               color=['#2ecc71', '#e74c3c'])
axes[0, 0].set_title('Class Distribution', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Count')
axes[0, 0].grid(alpha=0.3)

# 2. PCA visualization
colors = ['#2ecc71' if label == 0 else '#e74c3c' for label in quantum_data['y_train']]
axes[0, 1].scatter(quantum_data['X_train'][:, 0], 
                   quantum_data['X_train'][:, 1],
                   c=colors, alpha=0.6, s=30)
axes[0, 1].set_title('PCA-Reduced Feature Space (First 2 Components)', 
                     fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('PC1')
axes[0, 1].set_ylabel('PC2')
axes[0, 1].grid(alpha=0.3)

# 3. Feature importance (PCA)
pca_components = quantum_data['pca'].components_
feature_importance = np.abs(pca_components[0])
top_features_idx = np.argsort(feature_importance)[-6:]
top_features = [feature_cols[i] for i in top_features_idx]

axes[1, 0].barh(range(len(top_features)), 
                feature_importance[top_features_idx],
                color='#3498db')
axes[1, 0].set_yticks(range(len(top_features)))
axes[1, 0].set_yticklabels(top_features, fontsize=9)
axes[1, 0].set_title('Top 6 Important Features (PC1)', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Absolute Contribution')
axes[1, 0].grid(alpha=0.3, axis='x')

# 4. Data statistics
stats_text = f"""
Dataset Statistics:

Total Samples: {len(X)}
Train Samples: {len(X_train)}
Test Samples: {len(X_test)}

Quantum Training:

Qubits: 4
Training Samples: {len(quantum_data['X_train'])}
PCA Variance: {quantum_data['pca'].explained_variance_ratio_.sum():.1%}

Class Balance:

Normal: {sum(quantum_data['y_train']==0)}
Anomaly: {sum(quantum_data['y_train']==1)}
Ratio: {sum(quantum_data['y_train']==1)/len(quantum_data['y_train']):.1%}
"""

axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, 
                family='monospace', va='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
axes[1, 1].axis('off')
axes[1, 1].set_title('Dataset Summary', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('data_overview.png', dpi=150, bbox_inches='tight')
print("   ✓ Saved: data_overview.png")

# Export data for quantum training
np.savez('quantum_data.npz',
         X_train=quantum_data['X_train'],
         y_train=quantum_data['y_train'],
         X_test=quantum_data['X_test'],
         y_test=quantum_data['y_test'])

print("\n✅ Data Preparation Complete!")
print("   📦 Saved: quantum_data.npz")
print("\n🚀 Next step: Run '2_quantum_training.py' to train models!")