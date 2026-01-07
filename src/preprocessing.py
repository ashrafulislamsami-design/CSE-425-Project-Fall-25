import os
import torch
import torchaudio
import pandas as pd
from tqdm import tqdm

def clean_metadata(input_csv, output_csv):
    """Cleans the raw CSV and saves the processed version."""
    print(" CLEANING METADATA...")
    df = pd.read_csv(input_csv)
    df['lyrics'] = df['lyrics'].fillna("[Instrumental]").astype(str)
    
    # Surgical cleaning
    mask_garbage = (df['lyrics'].str.len() < 10) | (df['lyrics'].str.contains("#ERROR"))
    df.loc[mask_garbage, 'lyrics'] = "[Instrumental]"
    
    df.to_csv(output_csv, index=False)
    print(f" Cleaned CSV Saved to {output_csv}.")

def process_audio(audio_files, save_dir, device='cpu'):
    """Converts raw .wav files into Mel-Spectrogram tensors."""
    print("\n REBUILDING TENSORS...")
    os.makedirs(save_dir, exist_ok=True)
    
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=22050, n_mels=128, n_fft=2048, hop_length=512
    ).to(device)

    for f in tqdm(audio_files, desc="Processing Audio"):
        try:
            save_path = os.path.join(save_dir, os.path.basename(f).replace('.wav', '.pt'))
            if os.path.exists(save_path): continue 
            
            # Load & Resample
            waveform, sr = torchaudio.load(f)
            if sr != 22050: 
                waveform = torchaudio.transforms.Resample(sr, 22050)(waveform)
            if waveform.shape[0] > 1: 
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Convert to Mel Spectrogram
            mel = mel_transform(waveform.to(device))
            log_mel = torch.log(mel + 1e-9).cpu()
            
            # Standardize Width (640)
            if log_mel.shape[2] > 640: 
                log_mel = log_mel[:, :, :640]
            else: 
                log_mel = torch.nn.functional.pad(log_mel, (0, 640 - log_mel.shape[2]))
            
            torch.save(log_mel, save_path)
        except Exception as e:
            continue
    print(" AUDIO PROCESSING COMPLETE.")