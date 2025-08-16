import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

import pennylane as qml
from pennylane import numpy as pnp
from pennylane.templates import RandomLayers
import tensorflow as tf
# Codice migliorato senza Random Layers e con risultati più stabili
SAVE_PATH = r"C:\Users\ilnov\Desktop\DatiQuantistici"
os.makedirs(SAVE_PATH, exist_ok=True)

pnp.random.seed(0)
tf.random.set_seed(0)

# === CARICAMENTO E UNIONE DEI FILE CSV ===
folder_path = "C:/Users/ilnov/Desktop/Gerardo/Uni/Power"
csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
df_list = [pd.read_csv(os.path.join(folder_path, file)) for file in csv_files]
df = pd.concat(df_list, ignore_index=True)

# === CAMPIONAMENTO CASUALE ===
n_samples = 2000
df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)
print(f"Campionati {n_samples} esempi casuali dal dataset combinato.\n")

# === PREPROCESSING ===
X = df.drop("marker", axis=1)
log_columns = ['R1-PM1:V', 'R2-PM1:V', 'R3-PM1:V', 'R4-PM1:V']
X[log_columns] = np.log1p(X[log_columns].clip(lower=0))
X = X.clip(upper=1e5)
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

le = LabelEncoder()
y = le.fit_transform(df["marker"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

minmax = MinMaxScaler()
X_train_final = minmax.fit_transform(X_train_pca)
X_test_final = minmax.transform(X_test_pca)

print(f"\nPreprocessing completato.")
print(f"Dimensioni training set: {X_train_final.shape}")

# === RIORGANIZZAZIONE IN BLOCCHI ===
def reshape_to_blocks(X, block_size=4):
    reshaped = []
    for row in X:
        padded = np.pad(row, (0, -len(row) % block_size), constant_values=0)
        blocks = padded.reshape(-1, block_size)
        reshaped.append(blocks)
    return np.array(reshaped)

train_blocks = reshape_to_blocks(X_train_final)
test_blocks = reshape_to_blocks(X_test_final)
print("Shape blocchi train:", train_blocks.shape)

# === QCNN (Quantum Convolution) ===
dev = qml.device("default.qubit", wires=4)
n_layers = 6
rand_params = pnp.random.uniform(high=2 * np.pi, size=(n_layers, 4))

@qml.qnode(dev)
def circuit(phi):
    for j in range(4):
        qml.RY(np.pi * phi[j], wires=j)
    RandomLayers(rand_params, wires=list(range(4)))
    return [qml.expval(qml.PauliZ(j)) for j in range(4)]

def quanv_tabular(blocks):
    output = []
    for i, sample_blocks in enumerate(blocks):
        sample_out = []
        for block in sample_blocks:
            result = circuit(block)
            sample_out.extend(result)
        output.append(sample_out)
        if i % 10 == 0:
            print(f"Processati {i+1}/{len(blocks)}")
    return np.array(output)

# === ESECUZIONE QUANTUM PREPROCESSING (ENCODER) ===
print("Quantum pre-processing (tabular):")
q_train_features = quanv_tabular(train_blocks)
q_test_features = quanv_tabular(test_blocks)

# === SALVATAGGIO DEI DATI INTERMEDI (post-encoder) ===
np.save(os.path.join(SAVE_PATH, "q_train_features.npy"), q_train_features)
np.save(os.path.join(SAVE_PATH, "q_test_features.npy"), q_test_features)

# === DECODIFICA DEI DATI QUANTISTICI (DECODER) ===
def decode_quantum_data(encoded_data):
    """Decodifica semplice invertendo i valori (esempio simulato)."""
    return 1 - encoded_data

q_train_decoded = decode_quantum_data(q_train_features)
q_test_decoded = decode_quantum_data(q_test_features)

# === SALVATAGGIO DATI POST-DECODIFICA ===
np.save(os.path.join(SAVE_PATH, "q_train_decoded.npy"), q_train_decoded)
np.save(os.path.join(SAVE_PATH, "q_test_decoded.npy"), q_test_decoded)

# === SALVA ETICHETTE E DATI CLASSICI PER CONFRONTO ===
np.save(os.path.join(SAVE_PATH, "train_labels.npy"), y_train)
np.save(os.path.join(SAVE_PATH, "test_labels.npy"), y_test)
np.save(os.path.join(SAVE_PATH, "X_train_final.npy"), X_train_final)
np.save(os.path.join(SAVE_PATH, "X_test_final.npy"), X_test_final)

# === VISUALIZZAZIONE DATI DECODIFICATI ===
sample_decoded = q_train_decoded[0]
reshaped = sample_decoded.reshape(4, -1)
print("\nDati decodificati (prima istanza):")
print(np.round(reshaped, 3))

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# === CONFIG ===
SAVE_PATH = r"C:\Users\ilnov\Desktop\QAE"
X = np.load(os.path.join(SAVE_PATH, "X_train_final.npy"))
y = np.load(os.path.join(SAVE_PATH, "train_labels.npy"))
X_test = np.load(os.path.join(SAVE_PATH, "X_test_final.npy"))
y_test = np.load(os.path.join(SAVE_PATH, "test_labels.npy"))

# Usa solo i primi 6 feature
X, X_test = X[:, :6], X_test[:, :6]

# === NORMALIZATION ===
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

# === TORCH TENSORS ===
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# === QUANTUM DEVICE ===
n_qubits = 6
dev = qml.device("default.qubit", wires=n_qubits)

# === QUANTUM CIRCUIT (Semplice rotazioni + entanglement limitato) ===
def qrnn_qae_circuit(inputs, weights_enc, weights_dec):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))

    for i in range(n_qubits):
        qml.RX(weights_enc[i, 0], wires=i)
        qml.RY(weights_enc[i, 1], wires=i)
        qml.RZ(weights_enc[i, 2], wires=i)

    # Entanglement limitato: solo CNOT fra qubit pari e dispari (esempio)
    for i in range(0, n_qubits - 1, 2):
        qml.CNOT(wires=[i, i + 1])

    # Decoder: rotazioni inverse
    for i in range(n_qubits):
        qml.RX(weights_dec[i, 0], wires=i)
        qml.RY(weights_dec[i, 1], wires=i)
        qml.RZ(weights_dec[i, 2], wires=i)

    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

weight_shapes = {
    "weights_enc": (n_qubits, 3),
    "weights_dec": (n_qubits, 3)
}

qnode = qml.QNode(qrnn_qae_circuit, dev, interface="torch", diff_method="backprop")
q_layer = qml.qnn.TorchLayer(qnode, weight_shapes)

# === MODELLI ===
class FullQuantumModel(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.qae = q_layer
        self.classifier = nn.Linear(n_qubits, n_classes)

    def forward(self, x):
        x_encoded_decoded = self.qae(x)
        logits = self.classifier(x_encoded_decoded)
        return logits, x_encoded_decoded

class ClassicalModel(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 12),
            nn.ReLU(),
            nn.Linear(12, n_classes)
        )

    def forward(self, x):
        return self.classifier(x)

# === FUNZIONE DI TRAINING ===
def train_model(model, X_train, y_train, X_val, y_val, epochs=15, batch_size=8, is_quantum=False):
    # Learning rate più basso per quantum
    lr = 1e-3 if is_quantum else 1e-2
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    recon_loss_fn = nn.MSELoss()

    train_acc, val_acc = [], []
    train_loss, val_loss = [], []
    quantum_gradients = []

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(X_train.size(0))
        epoch_loss = 0
        correct = 0

        for i in range(0, X_train.size(0), batch_size):
            idx = perm[i:i + batch_size]
            batch_x, batch_y = X_train[idx], y_train[idx]

            optimizer.zero_grad()

            if is_quantum:
                outputs, recon = model(batch_x)
                loss = loss_fn(outputs, batch_y) + recon_loss_fn(recon, batch_x)
            else:
                outputs = model(batch_x)
                loss = loss_fn(outputs, batch_y)

            loss.backward()

            if is_quantum:
                grads = [p.grad.abs().mean().item() for p in model.qae.parameters() if p.grad is not None]
                quantum_gradients.append(np.mean(grads))

            optimizer.step()

            correct += (outputs.argmax(dim=1) == batch_y).sum().item()
            epoch_loss += loss.item()

        train_acc.append(correct / X_train.size(0))
        train_loss.append(epoch_loss / (X_train.size(0) // batch_size))

        model.eval()
        with torch.no_grad():
            if is_quantum:
                val_outputs, _ = model(X_val)
            else:
                val_outputs = model(X_val)
            val_l = loss_fn(val_outputs, y_val).item()
            val_loss.append(val_l)
            val_a = (val_outputs.argmax(dim=1) == y_val).float().mean().item()
            val_acc.append(val_a)

        print(f"Epoch {epoch + 1}: Train Acc={train_acc[-1]:.3f}, Val Acc={val_acc[-1]:.3f}")

    if is_quantum:
        plt.figure(figsize=(8, 4))
        plt.plot(quantum_gradients, label="Gradienti medi QAE")
        plt.xlabel("Batch")
        plt.ylabel("Gradiente medio |∇W|")
        plt.title("Evoluzione dei gradienti dei parametri quantistici")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return train_acc, val_acc, train_loss, val_loss

# === TRAINING ===
n_classes = len(np.unique(y))
quantum_model = FullQuantumModel(n_classes)
classical_model = ClassicalModel(input_dim=6, n_classes=n_classes)

q_train_acc, q_val_acc, q_train_loss, q_val_loss = train_model(
    quantum_model, X_tensor, y_tensor, X_test_tensor, y_test_tensor, is_quantum=True)

c_train_acc, c_val_acc, c_train_loss, c_val_loss = train_model(
    classical_model, X_tensor, y_tensor, X_test_tensor, y_test_tensor, is_quantum=False)

# === PLOT ACCURACY + LOSS ===
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(q_val_acc, "-ob", label="Quantum Val Acc")
plt.plot(c_val_acc, "-or", label="Classical Val Acc")
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(q_val_loss, "-ob", label="Quantum Val Loss")
plt.plot(c_val_loss, "-or", label="Classical Val Loss")
plt.title("Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()

#Risultati quasi simili, ho cambiato batch da 16 a 8
#	lr=0.01, batch_size=8, n_qubits=4 o 6, n_layers=1