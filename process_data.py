import os
import glob
import librosa
import soundfile as sf
import numpy as np

# --- CONFIGURATION ---
SOURCE_FOLDER = "data/audio"        # Where your big English songs are
OUTPUT_AUDIO  = "data/processed_audio" # Where small clips will go
OUTPUT_LYRICS = "data/processed_lyrics" # Where text files will go

# Slice settings (Faculty said 3000-4000 size is good)
CLIP_DURATION = 3.0 # Seconds per slice
SR = 22050          # Standard Sample Rate

def process_dataset():
    print("🚀 STARTING: Slice & Dice Operation...")
    
    # 1. Create Output Folders
    os.makedirs(OUTPUT_AUDIO, exist_ok=True)
    os.makedirs(OUTPUT_LYRICS, exist_ok=True)
    
    # 2. Find English Files
    files = glob.glob(os.path.join(SOURCE_FOLDER, "eng_*"))
    print(f"found {len(files)} source files. Processing...")

    total_clips = 0

    for file_path in files:
        filename = os.path.basename(file_path)
        base_name, _ = os.path.splitext(filename)
        
        try:
            # Load Audio
            audio, _ = librosa.load(file_path, sr=SR)
            
            # Calculate number of slices
            samples_per_clip = int(CLIP_DURATION * SR)
            total_samples = len(audio)
            
            # Slice it up!
            for i in range(0, total_samples - samples_per_clip, samples_per_clip):
                # Extract chunk
                chunk = audio[i : i + samples_per_clip]
                
                # Naming: eng_pop.0001_slice0.wav
                new_name = f"{base_name}_slice{i//samples_per_clip}"
                
                # A. Save Audio Clip
                out_audio_path = os.path.join(OUTPUT_AUDIO, f"{new_name}.wav")
                sf.write(out_audio_path, chunk, SR)
                
                # B. Save Dummy Lyric (Requirement: Text Input)
                # Since we don't have real lyrics, we put the genre/filename as text
                out_lyric_path = os.path.join(OUTPUT_LYRICS, f"{new_name}.txt")
                with open(out_lyric_path, "w") as f:
                    f.write(f"This is a song clip from {base_name}. It contains English lyrics.")
                
                total_clips += 1
                
                if total_clips % 100 == 0:
                    print(f"   -> Created {total_clips} clips so far...")

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

    print(f"\n🎉 DONE! Total Dataset Size: {total_clips} clips.")
    print(f"📂 Audio saved to: {OUTPUT_AUDIO}")
    print(f"📂 Lyrics saved to: {OUTPUT_LYRICS}")

if __name__ == "__main__":
    process_dataset()