import time
import torch
import torch.nn as nn
import os, struct
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

def load_mnist_images(path):
    with open(path, 'rb') as f:
        _, n, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
    return images.reshape(n, rows, cols).astype(np.float32) / 255.0

DATA_DIR = os.path.join(os.getcwd(), 'MNIST')
X_train = load_mnist_images(os.path.join(DATA_DIR, 'train-images.idx3-ubyte'))
train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train)), batch_size=128, shuffle=True)

# Test ConvAutoencoder
class ConvEncoder(nn.Module):
    def __init__(self, latent_dim=64):
        super(ConvEncoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU()
        )
        self.fc = nn.Linear(32 * 7 * 7, latent_dim)
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return torch.relu(self.fc(x))

class FCDecoder(nn.Module):
    def __init__(self, latent_dim=64, hidden_dim=256, output_dim=784):
        super(FCDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim), nn.Sigmoid()
        )
    def forward(self, z):
        return self.decoder(z)

class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super(ConvAutoEncoder, self).__init__()
        self.encoder = ConvEncoder(64)
        self.decoder = FCDecoder(64, 256, 784)
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat.view(x.size())

model = ConvAutoEncoder()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

start = time.time()
for i, (images,) in enumerate(train_loader):
    images = images.unsqueeze(1)
    x_hat = model(images)
    loss = criterion(x_hat, images)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i == 50: break
print("ConvAE 50 batches time:", time.time() - start)

# Test VAE LSTM
class LSTMEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(28, 128, batch_first=True)
        self.fc_mu = nn.Linear(128, 20)
        self.fc_logv = nn.Linear(128, 20)
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h_last = h_n.squeeze(0)
        return self.fc_mu(h_last), self.fc_logv(h_last)

class FCDecoderVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 784), nn.Sigmoid()
        )
    def forward(self, z):
        return self.net(z).view(-1, 28, 28)

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = LSTMEncoder()
        self.decoder = FCDecoderVAE()
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        return mu + torch.randn_like(std) * std
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

model_vae = VAE()
optimizer_vae = torch.optim.Adam(model_vae.parameters(), lr=1e-3)
def vae_loss(x_recon, x, mu, log_var):
    recon = torch.nn.functional.binary_cross_entropy(x_recon, x, reduction='sum') / x.size(0)
    kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.size(0)
    return recon + kl

start = time.time()
for i, (images,) in enumerate(train_loader):
    x_recon, mu, log_var = model_vae(images)
    loss = vae_loss(x_recon, images, mu, log_var)
    optimizer_vae.zero_grad()
    loss.backward()
    optimizer_vae.step()
    if i == 50: break
print("VAE LSTM 50 batches time:", time.time() - start)
