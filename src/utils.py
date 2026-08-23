import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import contingency_matrix

def purity_score(y_true, y_pred):
    """Calculates cluster purity."""
    cm = contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(cm, axis=0)) / np.sum(cm)

def plot_reconstruction(original, reconstructed, save_path):
    """Saves a comparison plot of original vs reconstructed spectrogram."""
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Spectrogram")
    plt.imshow(original.cpu())
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Beta-VAE Reconstruction")
    plt.imshow(reconstructed.detach().cpu())
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f" Saved reconstruction proof to {save_path}")