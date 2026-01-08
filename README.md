# Multi-modal Unsupervised Music Clustering 

**Course:** CSE 425: Neural Networks  
**Project:** Unsupervised Learning Pipeline for Music Clustering via Disentangled Variational Autoencoders

## Project Overview
This project implements an end-to-end unsupervised learning pipeline to cluster music tracks using multi-modal data. By integrating raw audio Mel-spectrograms and semantic lyrical features, we evaluate the effectiveness of Variational Autoencoders (VAEs) in learning meaningful latent representations of music genres.

The project addresses the challenge of high-dimensional data representation by progressing through three architectural stages:
1. **Easy Task:** Baseline Linear `BasicVAE` for feature compression and geometric clustering.
2. **Medium Task:** Convolutional VAE (ConvVAE) with hybrid multi-modal fusion (Audio + TF-IDF Lyrics).
3. **Hard Task:** Genre-conditioned $\beta$-CVAE ($\beta=4.0$) designed for disentangled representation learning.

## Consolidated File Structure Note
Contrary to the initial specification of three separate notebooks, **all experimental tasks, training loops, and metric verifications have been consolidated into a single, comprehensive notebook** to ensure pipeline integrity and reproducibility:

* **Primary Notebook:** `notebooks/main_analysis.ipynb`

This consolidated approach allows for a unified preprocessing pipeline and direct comparison of metrics across all three task levels in one environment.

## Dataset Access
The metadata and cleaned lyric features are included in the repository. **The raw audio files are hosted externally due to size limits.**

* **Download Link:** [Download Audio Dataset](https://drive.google.com/file/d/1UHPHUGJMzt6169MK6TUahAaX8CRPfhu3/view?usp=sharing)
* **Instructions:**
    1. Download `processed_audio.zip` from the link above.
    2. Extract the `.wav` files into the `data/raw/` folder of this repository.

## Results Summary
Our experiments demonstrate that while global geometric metrics (Silhouette) are limited by the inherent spectral overlap of music, the $\beta$-CVAE successfully captures semantic structures.

| Task Level | Model Architecture | Best Metric | Score | Key Finding |
| :--- | :--- | :--- | :--- | :--- |
| **Easy** | Basic VAE (Linear) | Silhouette | **0.1145** | Slightly outperformed linear PCA baseline (0.1033). |
| **Medium** | ConvVAE + Lyrics | ARI | **0.0471** | High-dimensional sparsity in hybrid features led to cluster overlap. |
| **Hard** | **Beta-CVAE ($\beta=4$)** | **Purity** | **0.2731** | **Best Result.** Genre-conditioning successfully disentangled the latent space. |



## Technical Validation
Model validity is confirmed through:
* **Cluster Purity:** The Hard Task achieved **0.2731**, significantly outperforming random baseline assignments.
* **Reconstruction Proof:** The `Hard_Task_Proof.png` visualization demonstrates high-fidelity recovery of original Mel-spectrograms, proving the encoder successfully compressed essential data.

## Installation & Usage
1. **Clone the repository.**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
