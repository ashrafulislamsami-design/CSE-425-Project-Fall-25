# Unsupervised Music Clustering 

**Course:** CSE 425: Neural Networks
**Project:** Unsupervised Learning Pipeline for Music Clustering using Beta-CVAE

##  Project Overview
This project implements an unsupervised learning pipeline to cluster music tracks from a hybrid dataset (English and Bangla). We address the challenge of high-dimensional, multi-modal data (Audio + Lyrics + Genre) by progressing through three architectural stages:
1.  **Easy Task:** Baseline PCA + K-Means.
2.  **Medium Task:** Convolutional VAE (CVAE) with hybrid features.
3.  **Hard Task:** Conditional Beta-VAE ($\beta$-CVAE) for disentangled representation learning.

##  Dataset Access
The metadata and cleaned lyric features are included in this repository (`data/processed/`).
**The raw audio files are hosted externally due to size limits.**

* **Download Link:** [https://drive.google.com/file/d/1UHPHUGJMzt6169MK6TUahAaX8CRPfhu3/view?usp=sharing]
* **Instructions:**
    1.  Download `processed_audio.zip` from the link above.
    2.  Extract the `.wav` files into the `data/raw/` folder of this repository.

##  Results
Our experiments show a clear progression in model performance. The Hard Task (Beta-CVAE) achieved the highest interpretability (Cluster Purity) by leveraging genre conditioning.

| Task Level | Model Architecture | Best Metric | Score | Baseline / Comparison | Key Finding |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Easy** | Basic VAE | Silhouette | **0.1145** | PCA Baseline: 0.1033 | VAE slightly outperformed linear PCA in geometric clustering. |
| **Medium** | ConvVAE | ARI | **0.0471** | Silhouette: 0.0121 | Convolutional features struggled with high-dimensional sparsity (Low Silhouette). |
| **Hard** | **Beta-CVAE ($\beta=4$)** | **Purity** | **0.2731** | NMI: 0.0794 | **Best Result.** Conditioning on genre successfully disentangled the latent space. |

##  Installation & Usage
1.  **Clone the repository**
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the analysis:**
    Open `notebooks/main_analysis.ipynb` to view the training pipeline and results.

##  Repository Structure
* `data/`: Processed metadata CSV.
* `notebooks/`: Jupyter Notebook with full training loops and verification.
* `results/`: t-SNE plots and Spectrogram reconstructions.
* `src/`: Modular Python scripts for model architectures and preprocessing.
