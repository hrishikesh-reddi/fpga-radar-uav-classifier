# ============================================================
# Hybrid Quantum-Classical Neural Network (Model 1)
# ============================================================

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import pennylane as qml
from sklearn.metrics import accuracy_score
import pickle  # Add pickle for model saving

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------

data = np.load("quantum_data.npz")
X_train = data["X_train"]
y_train = data["y_train"]
X_test = data["X_test"]
y_test = data["y_test"]

print("Train:", X_train.shape, " Test:", X_test.shape)

# ------------------------------------------------------------
# Hybrid QNN Circuit
# ------------------------------------------------------------

dev = qml.device("default.qubit", wires=4)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(x, w):
    for i in range(4):
        qml.RY(x[i], wires=i)

    for i in range(4):
        qml.RY(w[0, i], wires=i)
        qml.RZ(w[1, i], wires=i)

    for i in range(3):
        qml.CNOT(wires=[i, i+1])
    qml.CNOT(wires=[3, 0])

    for i in range(4):
        qml.RY(w[2, i], wires=i)

    return qml.expval(qml.PauliZ(0))

class HybridQNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Linear(4, 8),
            nn.Tanh()
        )
        self.q_weights = nn.Parameter(torch.randn(3, 4) * 0.1)
        self.post = nn.Sequential(
            nn.Linear(1, 4),
            nn.ReLU(),
            nn.Linear(4, 2)
        )

    def forward(self, x):
        x = self.pre(x)
        batch = x.shape[0]
        out = torch.zeros(batch, 1).to(device)

        for i in range(batch):
            out[i] = quantum_circuit(x[i], self.q_weights)

        return self.post(out)

def train_qnn(epochs=20):  # Increased from 10 to 20 for better accuracy
    model = HybridQNN().to(device)
    opt = Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    # Use more training data for better accuracy
    Xt = torch.tensor(X_train[:300], dtype=torch.float32).to(device)
    yt = torch.tensor(y_train[:300], dtype=torch.long).to(device)
    Xv = torch.tensor(X_test[:150], dtype=torch.float32).to(device)  # Use 150 test samples

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()

        out = model(Xt)
        loss = loss_fn(out, yt)
        loss.backward()
        opt.step()

        print(f"Epoch {epoch}: loss={loss.item():.4f}")

    preds = model(Xv).argmax(dim=1).cpu().numpy()
    acc = accuracy_score(y_test[:150], preds)  # Match test sample size
    print("\nFinal Accuracy:", acc)

    # Save the trained model
    torch.save(model.state_dict(), 'hybrid_qnn_model.pth')
    print("Model saved as 'hybrid_qnn_model.pth'")

    return model, preds

model1, pred1 = train_qnn()