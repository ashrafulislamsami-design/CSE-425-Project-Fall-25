import os

# --- FIX PERMISSION ERROR ---
# Force Numba to use a local folder for caching, not C:\Python313
os.environ["NUMBA_CACHE_DIR"] = os.path.join(os.getcwd(), "numba_cache")

import torch
import librosa
import soundfile as sf
import numpy as np
import time

# --- CONFIGURATION ---
AUDIO_DIR = os.path.join("..", "data", "processed_audio")
OUT_DIR = os.path.join("..", "data", "processed_features")
SAMPLE_RATE = 22050
N_MFCC = 13

def main():
    print("🚀 STARTING: Feature Pre-calculation...")
    
    # Ensure directories exist
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs("numba_cache", exist_ok=True) # Create local cache folder
    
    # Get all wav files
    files = [f for f in os.listdir(AUDIO_DIR) if f.endswith('.wav')]
    total = len(files)
    
    if total == 0:
        print("❌ ERROR: No .wav files found! Check data/processed_audio.")
        return

    print(f"✅ Found {total} audio clips. Converting to math tensors...")

    count = 0
    start_time = time.time()
    
    for filename in files:
        audio_path = os.path.join(AUDIO_DIR, filename)
        out_name = filename.replace(".wav", ".pt")
        out_path = os.path.join(OUT_DIR, out_name)
        
        # Skip if already exists (resume capability)
        if os.path.exists(out_path):
            count += 1
            continue
            
        try:
            # 1. Load Audio (Fast)
            y, sr = sf.read(audio_path)
            if y.ndim > 1: y = y.mean(axis=1) # Mono
                
            # 2. Extract MFCC
            mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
            
            # 3. Stats
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_var = np.var(mfcc, axis=1)
            feature_vector = np.concatenate((mfcc_mean, mfcc_var))
            
            # 4. Save
            tensor = torch.tensor(feature_vector, dtype=torch.float32)
            torch.save(tensor, out_path)
            
            count += 1
            if count % 100 == 0:
                print(f"   [{count}/{total}] Processed... ({(count/total)*100:.1f}%)")
                
        except Exception as e:
            print(f"❌ Error on {filename}: {e}")

    print(f"\n🎉 DONE! Processed {count} files in {time.time()-start_time:.1f} seconds.")

if __name__ == "__main__":
    main()