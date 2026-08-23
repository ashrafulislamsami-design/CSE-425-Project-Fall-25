# Unsupervised Music Clustering with Variational Autoencoders

Solo project for CSE 425 (Neural Networks), BRAC University, Fall 2025.

## Question

Can a variational autoencoder learn a latent representation of music audio in which genre structure emerges on its own, without genre labels being used to form the clusters?

On this data, at this scale, with these architectures: no. The reason it did not work is the useful part of this repository, so the results below are reported as measured rather than as hoped.

## Data

- **3,050 tracks across 8 genres**, from a Free Music Archive subset. Classes are unbalanced: International 610, Experimental 487, Pop 458, Folk 398, Rock 343, Hip-Hop 338, Electronic 327, Instrumental 89.
- **Audio features:** log-Mel spectrograms via `torchaudio` at 22.05 kHz mono, 128 mel bands, `n_fft` 2048, `hop_length` 512, padded or truncated to 640 frames.
- **Lyric features:** the lyrics are ASR transcripts and are noisy in places. Rows shorter than 10 characters or containing `#ERROR` are replaced with `[Instrumental]` before vectorising.
- **Raw audio (3.5 GB) is hosted externally** because it exceeds GitHub's size limit: [download link](https://drive.google.com/file/d/1UHPHUGJMzt6169MK6TUahAaX8CRPfhu3/view?usp=sharing). Extract the `.wav` files into `data/raw/`. Metadata and cleaned lyrics are committed in `data/processed/`.

## Experiments

Three models of increasing capacity, each measured against a classical baseline of PCA to 128 dimensions followed by K-means with k = 8.

| Stage | Model | Configuration |
| :-- | :-- | :-- |
| Easy | `BasicVAE` | Fully connected, 128x640 to 512 to 256 to latent 128. 10 epochs, Adam lr 1e-4. |
| Medium | `ConvVAE` | Three stride-2 convolutional layers to latent 128, concatenated with 128 TF-IDF lyric features. 20 epochs, Adam lr 1e-4. Clustering compared across K-means, agglomerative and DBSCAN. |
| Hard | `BetaCVAE` | Same convolutional trunk, genre one-hot (dim 8) concatenated at both encoder and decoder, beta = 4.0. 50 epochs, Adam lr 5e-5, gradient-norm clipping at 1.0. |

Metrics: silhouette, Calinski-Harabasz, Davies-Bouldin, adjusted Rand index, normalised mutual information, and cluster purity.

Everything runs from a single notebook, `notebooks/main_analysis.ipynb`, so that all three stages share one preprocessing pipeline and the metrics are directly comparable.

## Results

| Stage | Model | Silhouette | ARI | NMI | Purity |
| :-- | :-- | --: | --: | --: | --: |
| Baseline | PCA + K-means | 0.1033 | | | |
| Easy | BasicVAE | **0.1145** | | | |
| Medium | ConvVAE + TF-IDF | | **0.0471** | | |
| Hard | BetaCVAE (beta = 4.0) | 0.0221 | 0.0419 | 0.0794 | **0.2731** |

How to read these:

- **No latent space recovered genre structure.** ARI peaks at 0.047, against 0.0 for chance agreement.
- **The classical baseline held up.** PCA scored silhouette 0.1033 against the best VAE's 0.1145, and PCA actually beat `BasicVAE` on Calinski-Harabasz (629.0 against 415.3). The neural models did not earn their extra complexity here.
- **Purity of 0.2731 needs its baseline.** Assigning every track to the majority genre scores 0.200 (610 of 3,050), so this is a modest gain rather than a strong one.
- **`results/Hard_Task_Latent_Space.png` shows the genres intermixed.** It is not a separated latent space and should not be read as one.
- **`results/Hard_Task_Proof.png` shows the beta trade-off.** Reconstruction preserves the coarse spectral envelope while blurring harmonic detail, which is what beta = 4.0 should be expected to cost.

## What I would change

- **Concatenating sparse TF-IDF features onto dense spectrogram latents hurt rather than helped.** 128 sparse lexical dimensions sitting beside a dense audio embedding let lexical noise drive the distances. A separate encoder per modality with a learned fusion step, instead of raw concatenation, is the first thing I would rebuild.
- **beta = 4.0 was too aggressive.** It cost reconstruction detail without buying usable disentanglement. Annealing beta up from 0 during training would be the obvious next attempt.
- **The epoch budgets were unequal (10 / 20 / 50)**, so the three stages are not a clean comparison. Matching budgets and reporting variance across random seeds would make the ranking trustworthy.
- **Genre may be the wrong target.** Genre is a social category more than an acoustic one, so a low ARI against genre labels may mean the metric is unfair to the representation rather than that the representation is poor. Evaluating against acoustic attributes such as tempo, key or instrumentation would be a better test of what the latent space actually captured.

## Repository layout

```
src/preprocessing.py    metadata cleaning and Mel-spectrogram extraction
src/models.py           BasicVAE, ConvVAE, BetaCVAE
src/utils.py            cluster purity and reconstruction plotting
notebooks/              main_analysis.ipynb, all three stages end to end
data/processed/         cleaned metadata and lyrics (committed)
data/raw/               raw .wav files (not committed, see download link)
results/                t-SNE plots and the reconstruction figure
```

## Reproducing

```bash
pip install -r requirements.txt
# download and extract the raw audio into data/raw/ first
jupyter notebook notebooks/main_analysis.ipynb
```

Spectrogram extraction over the full set takes roughly 15 minutes on a GPU and is cached to `.pt` tensors, so later runs skip it.
