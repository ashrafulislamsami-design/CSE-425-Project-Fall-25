# Unsupervised Clustering of Musical Audio using Linear VAE

**Author:** Md. Ashraful Islam Sami  
**Course:** CSE 425: Neural Networks  
**Project Type:** Unsupervised Learning (Easy Task)

##  Project Overview
This project implements a **Linear Variational Autoencoder (VAE)** to extract latent features from raw audio data (GTZAN dataset). It clusters these features using **K-Means** and demonstrates that VAE-learned features significantly outperform standard PCA dimensionality reduction.

## Key Results

| Metric | VAE (Our Method) | PCA (Baseline) | Improvement |
| :--- | :--- | :--- | :--- |
| **Silhouette Score** | **0.4811** | 0.3015 | +59% |
| **Calinski-Harabasz** | **8788.7** | 2200.9 | **+299%** |

##  Repository Structure
```text
project/
├── data/               # Raw audio (Not included in repo due to size)
├── src/
│   ├── dataset.py      # MFCC Feature Extraction & Normalization
│   ├── vae.py          # PyTorch Linear VAE Model
│   ├── train.py        # Training Loop
│   ├── clustering.py   # t-SNE Visualization & Metrics
│   └── preprocess.py   # Data preprocessing script
├── results/            # Generated Plots & Saved Models
└── requirements.txt    # Python dependencies

How to Run
1. Install Dependencies
Bash

pip install -r requirements.txt
2. Prepare Data
Place your .wav files in data/processed_audio/ and run the pre-processor (this calculates MFCCs and saves them as tensors):

Bash

python src/preprocess.py
3. Train the VAE
Trains the model for 20 epochs and saves vae_model.pth in the results folder.

Bash

python src/train.py
4. Analyze & Visualize
Runs K-Means clustering, calculates Silhouette/CH scores, and generates the t-SNE comparison plot.

Bash

python src/clustering.py