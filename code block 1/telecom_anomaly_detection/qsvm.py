# ============================================================
# Quantum Kernel SVM (Model 2)
# ============================================================

import numpy as np
import pennylane as qml
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle  # Add pickle for model saving

# ------------------------
# Load Data
# ------------------------
data = np.load("quantum_data.npz")
X_train = data["X_train"]
y_train = data["y_train"]
X_test = data["X_test"]
y_test = data["y_test"]

print("Train:", X_train.shape, " Test:", X_test.shape)

# ------------------------
# Kernel Circuit
# ------------------------
dev = qml.device("default.qubit", wires=4)

@qml.qnode(dev)
def kernel_qnode(x1, x2):
    for i in range(4):
        qml.Hadamard(wires=i)
        qml.RZ(x1[i], wires=i)

    for i in range(3):
        qml.CZ(wires=[i, i+1])
    qml.CZ(wires=[3, 0])

    for i in range(4):
        qml.RZ(-x2[i], wires=i)
        qml.Hadamard(wires=i)

    return qml.probs(wires=range(4))

def compute_kernel(A, B):
    K = np.zeros((len(A), len(B)))
    total = len(A) * len(B)
    count = 0

    for i in range(len(A)):
        for j in range(len(B)):
            probs = kernel_qnode(A[i], B[j])
            K[i, j] = probs[0]
            count += 1
        print(f"Progress {count}/{total}")

    return K

# ------------------------
# Training QSVM
# ------------------------
def train_qsvm(n_samples=100):  # Increased from 20 to 100 for better accuracy
    idx = np.random.choice(len(X_train), n_samples, replace=False)
    XS = X_train[idx]
    yS = y_train[idx]

    print("Computing K_train...")
    K_train = compute_kernel(XS, XS)

    print("Computing K_test...")
    K_test = compute_kernel(X_test[:100], XS)  # Use 100 test samples

    svm = SVC(kernel="precomputed", C=1)
    svm.fit(K_train, yS)

    pred = svm.predict(K_test)
    acc = accuracy_score(y_test[:100], pred)

    # Save the trained model
    with open('qsvm_model.pkl', 'wb') as f:
        pickle.dump(svm, f)
    
    # Save the support vectors for inference
    model_data = {
        'model': svm,
        'support_vectors': XS,
        'y_support': yS
    }
    
    with open('qsvm_model_complete.pkl', 'wb') as f:
        pickle.dump(model_data, f)

    print("\nQSVM Accuracy:", acc)
    print("Model saved as 'qsvm_model.pkl' and 'qsvm_model_complete.pkl'")
    return svm, pred

model2, pred2 = train_qsvm()