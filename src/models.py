import torch
import torch.nn as nn

# --- 1. BASIC VAE (Easy Task) ---
class BasicVAE(nn.Module):
    def __init__(self):
        super().__init__()
        # Simple Linear Layers
        self.encoder = nn.Sequential(nn.Linear(128*640, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())
        self.fc_mu = nn.Linear(256, 128)
        self.fc_logvar = nn.Linear(256, 128)
        self.decoder = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 128*640))

    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, -10, 10)
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def forward(self, x):
        h = self.encoder(x.view(-1, 128*640))
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

# --- 2. CONVOLUTIONAL VAE (Medium Task) ---
class ConvVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(128*16*80, 128)
        self.fc_logvar = nn.Linear(128*16*80, 128)
        self.decoder_fc = nn.Linear(128, 128*16*80)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 16, 80)),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, 2, 1, 1)
        )
    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
    def forward(self, x):
        mu, logvar = self.fc_mu(self.encoder(x)), self.fc_logvar(self.encoder(x))
        return self.decoder(self.decoder_fc(self.reparameterize(mu, logvar))), mu, logvar

# --- 3. BETA-CVAE (Hard Task) ---
class BetaCVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
            nn.Flatten()
        )
        # Condition (Genre) added to Latent
        self.fc_mu = nn.Linear(128*16*80 + 8, 128) 
        self.fc_logvar = nn.Linear(128*16*80 + 8, 128)
        self.decoder_fc = nn.Linear(128 + 8, 128*16*80)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 16, 80)),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, 2, 1, 1)
        )
    def encode(self, x, c):
        h = self.encoder(x)
        h_cond = torch.cat([h, c], dim=1)
        return self.fc_mu(h_cond), self.fc_logvar(h_cond)
    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, -10, 10)
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
    def decode(self, z, c):
        z_cond = torch.cat([z, c], dim=1)
        return self.decoder(self.decoder_fc(z_cond))
    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar