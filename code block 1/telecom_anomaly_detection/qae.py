# ============================================================
# Quantum Autoencoder (Model 3)
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

# -----------------------
# Load data
# -----------------------
data = np.load("quantum_data.npz")
X_train = data["X_train"]
y_train = data["y_train"]
X_test = data["X_test"]
y_test = data["y_test"]

print("Train:", X_train.shape, " Test:", X_test.shape)

# -----------------------
# QAE Circuit
# -----------------------
dev = qml.device("default.qubit", wires=6)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def qae(inputs, w_enc, w_dec):
    for i in range(4):
        qml.RY(inputs[i], wires=i)

    # Compress → latent
    for i in range(2):
        qml.RY(w_enc[i,0], wires=i)
        qml.RZ(w_enc[i,1], wires=i)

    qml.CNOT(wires=[0,4])
    qml.CNOT(wires=[1,5])

    # Decompress
    for i in range(4):
        qml.RY(w_dec[i,0], wires=i)
        qml.RZ(w_dec[i,1], wires=i)

    return [qml.expval(qml.PauliZ(i)) for i in range(4)]

class QAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_enc = nn.Parameter(torch.randn(2,2)*0.1)
        self.w_dec = nn.Parameter(torch.randn(4,2)*0.1)

    def forward(self, x):
        out = torch.zeros_like(x)
        for i in range(len(x)):
            out[i] = torch.tensor(qae(x[i], self.w_enc, self.w_dec))
        return out

    def score(self, x):
        recon = self.forward(x)
        return torch.mean((x-recon)**2, dim=1)

def train_qae(epochs=20):  # Increased from 10 to 20 for better accuracy
    normal = X_train[y_train==0][:200]  # Increased from 100 to 200
    Xt = torch.tensor(normal, dtype=torch.float32).to(device)
    Xtest = torch.tensor(X_test[:150], dtype=torch.float32).to(device)  # Use 150 test samples

    model = QAE().to(device)
    opt = Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for ep in range(epochs):
        opt.zero_grad()
        recon = model(Xt)
        loss = loss_fn(recon, Xt)
        loss.backward()
        opt.step()
        print(f"Epoch {ep}: Loss={loss.item():.4f}")

    scores = model.score(Xtest).cpu().numpy()
    threshold = np.median(scores)
    preds = (scores > threshold).astype(int)

    acc = accuracy_score(y_test[:150], preds)  # Match test sample size
    print("\nQAE Accuracy:", acc)

    # Save the trained model
    torch.save(model.state_dict(), 'qae_model.pth')
    print("Model saved as 'qae_model.pth'")

    return model, preds

model3, pred3 = train_qae()