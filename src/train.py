import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
from dataset import MusicDataset
from vae import LinearVAE

def loss_function(recon_x, x, mu, logvar):
    # MSE: Accuracy
    MSE = torch.nn.functional.mse_loss(recon_x, x, reduction='sum')
    # KLD: Structure
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD

def train():
    # --- CONFIGURATION ---
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    EPOCHS = 20
    LATENT_DIM = 4
    
    # NEW PATH: Point to the calculated features, not audio
    FEATURE_DIR = "../data/processed_features" 
    MODEL_SAVE_PATH = "../results/vae_model.pth"
    
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    print(f"Loading features from {FEATURE_DIR}...")
    dataset = MusicDataset(feature_dir=FEATURE_DIR)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Initialize Model
    model = LinearVAE(input_dim=26, hidden_dim=64, latent_dim=LATENT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n🔥 Starting Training (Lightning Fast Mode)...")
    model.train()
    
    for epoch in range(1, EPOCHS + 1):
        train_loss = 0
        for data in train_loader:
            data = data.to(device)
            
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_loss = train_loss / len(dataset)
        print(f"Epoch: {epoch}/{EPOCHS} | Average Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n🎉 Training Complete. Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()