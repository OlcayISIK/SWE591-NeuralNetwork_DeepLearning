import os
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import itertools
import time

def load_mnist_images(path):
    with open(path, 'rb') as f:
        _, n, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
    return images.reshape(n, rows, cols).astype(np.float32) / 255.0

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
print(f"Using device: {device}")

# Load dataset
DATA_DIR = os.path.join(os.getcwd(), 'MNIST')
X_train = load_mnist_images(os.path.join(DATA_DIR, 'train-images.idx3-ubyte'))
X_test  = load_mnist_images(os.path.join(DATA_DIR, 't10k-images.idx3-ubyte'))

BATCH_SIZE = 128
train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train)), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test)), batch_size=BATCH_SIZE, shuffle=False)

# Models
class LSTMEncoder(nn.Module):
    def __init__(self, hidden_size, latent_dim):
        super().__init__()
        self.lstm = nn.LSTM(28, hidden_size, batch_first=True)
        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        self.fc_logv = nn.Linear(hidden_size, latent_dim)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h_last = h_n.squeeze(0)
        return self.fc_mu(h_last), self.fc_logv(h_last)

class FCDecoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z).view(-1, 28, 28)

class VAE(nn.Module):
    def __init__(self, hidden_size, latent_dim):
        super().__init__()
        self.encoder = LSTMEncoder(hidden_size, latent_dim)
        self.decoder = FCDecoder(latent_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        return x_recon, mu, log_var

def vae_loss(x_recon, x, mu, log_var):
    recon = F.binary_cross_entropy(x_recon, x, reduction='sum') / x.size(0)
    kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.size(0)
    return recon + kl, recon, kl

def train_and_evaluate(latent_dim, hidden_size, lr, epochs=10):
    model = VAE(hidden_size, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Train
    for epoch in range(epochs):
        model.train()
        for (Xb,) in train_loader:
            Xb = Xb.to(device)
            x_recon, mu, log_var = model(Xb)
            loss, _, _ = vae_loss(x_recon, Xb, mu, log_var)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    # Evaluate
    model.eval()
    total_loss, total_recon, total_kl = 0.0, 0.0, 0.0
    N = len(test_loader.dataset)
    with torch.no_grad():
        for (Xb,) in test_loader:
            Xb = Xb.to(device)
            x_recon, mu, log_var = model(Xb)
            loss, recon, kl = vae_loss(x_recon, Xb, mu, log_var)
            n = Xb.size(0)
            total_loss += loss.item() * n
            total_recon += recon.item() * n
            total_kl += kl.item() * n
            
    return total_loss / N, total_recon / N, total_kl / N

# Grid search parameters
latent_dims = [20, 10]
hidden_sizes = [128, 64]
learning_rates = [1e-3, 5e-4]

results = []

print("Starting VAE Grid Search...")
start_time = time.time()

for ld, hs, lr in itertools.product(latent_dims, hidden_sizes, learning_rates):
    print(f"Training -> Latent Dim: {ld}, Hidden Size: {hs}, LR: {lr}")
    avg_loss, avg_recon, avg_kl = train_and_evaluate(ld, hs, lr, epochs=10)
    print(f"Result -> Loss: {avg_loss:.2f}, Recon: {avg_recon:.2f}, KL: {avg_kl:.2f}")
    results.append({
        'Latent Dim': ld,
        'Hidden Size': hs,
        'Learning Rate': lr,
        'Total Loss': avg_loss,
        'Recon Loss': avg_recon,
        'KL Divergence': avg_kl
    })

print(f"Grid search completed in {(time.time() - start_time)/60:.2f} minutes.")

# Save results
df = pd.DataFrame(results)
df = df.sort_values(by='Total Loss')
df.to_csv('vae_grid_results.csv', index=False)
print("Results saved to vae_grid_results.csv")
