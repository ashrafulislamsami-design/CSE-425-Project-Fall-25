import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score # <--- NEW METRIC
from torch.utils.data import DataLoader
import os

# Import your code
from dataset import MusicDataset
from vae import LinearVAE

def extract_features(model, dataset, device):
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    latent_vectors = []
    raw_data = []
    model.eval()
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            h = torch.relu(model.encoder_h(batch))
            mu = model.fc_mu(h)
            latent_vectors.append(mu.cpu().numpy())
            raw_data.append(batch.cpu().numpy())
            
    return np.concatenate(latent_vectors), np.concatenate(raw_data)

def perform_clustering_analysis():
    # --- CONFIG ---
    MODEL_PATH = "../results/vae_model.pth"
    FEATURE_DIR = "../data/processed_features" 
    RESULTS_DIR = "../results"
    LATENT_DIM = 4
    N_CLUSTERS = 5
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running analysis on: {device}")

    # 1. Load Data
    print(f"Loading features from {FEATURE_DIR}...")
    dataset = MusicDataset(feature_dir=FEATURE_DIR) 
    
    # 2. Load Model
    print("Loading VAE model...")
    model = LinearVAE(input_dim=26, hidden_dim=64, latent_dim=LATENT_DIM).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    
    # 3. Extract Features
    print("Extracting features...")
    X_vae, X_raw = extract_features(model, dataset, device)
    
    # 4. Clustering (VAE)
    print(f"Clustering VAE features (k={N_CLUSTERS})...")
    kmeans_vae = KMeans(n_clusters=N_CLUSTERS, random_state=42)
    labels_vae = kmeans_vae.fit_predict(X_vae)
    
    # 5. Clustering (PCA Baseline)
    print("Running Baseline (PCA) comparison...")
    pca = PCA(n_components=LATENT_DIM)
    X_pca = pca.fit_transform(X_raw)
    kmeans_pca = KMeans(n_clusters=N_CLUSTERS, random_state=42)
    labels_pca = kmeans_pca.fit_predict(X_pca)
    
    # 6. Evaluation Metrics (UPDATED)
    print("Calculating Metrics (Silhouette & Calinski-Harabasz)...")
    
    # Silhouette (Range: -1 to 1, Higher is better)
    sil_vae = silhouette_score(X_vae, labels_vae)
    sil_pca = silhouette_score(X_pca, labels_pca)
    
    # Calinski-Harabasz (Range: 0 to Infinity, Higher is better)
    ch_vae = calinski_harabasz_score(X_vae, labels_vae)
    ch_pca = calinski_harabasz_score(X_pca, labels_pca)
    
    print("\n" + "="*60)
    print("RESULTS TABLE (FINAL AUDIT)")
    print("="*60)
    print(f"{'Metric':<25} | {'VAE (Yours)':<15} | {'PCA (Base)':<15}")
    print("-" * 60)
    print(f"{'Silhouette Score':<25} | {sil_vae:.4f}          | {sil_pca:.4f}")
    print(f"{'Calinski-Harabasz':<25} | {ch_vae:.4f}          | {ch_pca:.4f}")
    print("="*60)
    
    # 7. Visualization (UPDATED to t-SNE)
    print("\nGenerating t-SNE plots (This is slower than PCA, please wait)...")
    
    # We use t-SNE to project BOTH VAE and Raw data to 2D for a fair visual comparison
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
    
    # Plot 1: VAE Latent Space
    X_vae_2d = tsne.fit_transform(X_vae)
    
    # Plot 2: Raw Data (Baseline) - using t-SNE on raw data usually shows the mess better than PCA
    X_raw_2d = tsne.fit_transform(X_raw)
    
    plt.figure(figsize=(14, 6))
    
    # Subplot 1: VAE
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=X_vae_2d[:,0], y=X_vae_2d[:,1], hue=labels_vae, palette="tab10", s=10, alpha=0.6)
    plt.title(f"VAE Latent Space (t-SNE)\nSil: {sil_vae:.3f} | CH: {ch_vae:.1f}")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    
    # Subplot 2: Baseline
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=X_raw_2d[:,0], y=X_raw_2d[:,1], hue=labels_pca, palette="tab10", s=10, alpha=0.6)
    plt.title(f"Baseline Space (t-SNE on Raw Data)\nSil: {sil_pca:.3f} | CH: {ch_pca:.1f}")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "clustering_comparison_final.png"))
    print(f"Plot saved to {RESULTS_DIR}/clustering_comparison_final.png")

if __name__ == "__main__":
    perform_clustering_analysis()