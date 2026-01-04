import os
import glob
import soundfile as sf
import math

# --- CONFIGURATION ---
PROJECT_ROOT = "."
RAW_AUDIO_DIR = os.path.join(PROJECT_ROOT, "data", "audio")
PROCESSED_AUDIO = os.path.join(PROJECT_ROOT, "data", "processed_audio")
PROCESSED_LYRICS = os.path.join(PROJECT_ROOT, "data", "processed_lyrics")

# Target: 3 second clips
CLIP_DURATION = 3.0 

def main():
    print("🚀 STARTING: Fast Slice Operation...")
    
    # Create Output Folders
    os.makedirs(PROCESSED_AUDIO, exist_ok=True)
    os.makedirs(PROCESSED_LYRICS, exist_ok=True)
    
    # Find all WAV files
    files = glob.glob(os.path.join(RAW_AUDIO_DIR, "*.wav"))
    
    if not files:
        print("❌ ERROR: No .wav files found in data/audio!")
        print("   Did you run the download/clone steps?")
        return

    print(f"✅ Found {len(files)} files. Slicing the first 400...")
    
    files_to_process = files[:400] # Limit to 400 songs (results in ~4000 clips)
    total_clips = 0

    for idx, file_path in enumerate(files_to_process):
        filename = os.path.basename(file_path)
        base_name = os.path.splitext(filename)[0]
        
        try:
            # FAST LOAD: Use soundfile directly (no resampling)
            audio, sr = sf.read(file_path)
            
            # Calculate samples per clip
            samples_per_clip = int(CLIP_DURATION * sr)
            total_samples = len(audio)
            
            # If song is shorter than 3s, skip
            if total_samples < samples_per_clip:
                continue

            # Slice Loop
            num_slices = math.floor(total_samples / samples_per_clip)
            
            # Print progress only every 10 files to keep terminal clean
            if idx % 10 == 0:
                print(f"[{idx}/{len(files_to_process)}] Slicing {filename} into {num_slices} clips...")

            for i in range(num_slices):
                start = i * samples_per_clip
                end = start + samples_per_clip
                chunk = audio[start:end]
                
                # Naming: eng_blues.00000_slice0.wav
                slice_name = f"{base_name}_slice{i}"
                
                # Save Audio
                out_audio_path = os.path.join(PROCESSED_AUDIO, f"{slice_name}.wav")
                sf.write(out_audio_path, chunk, sr)
                
                # Save Dummy Lyric
                out_lyric_path = os.path.join(PROCESSED_LYRICS, f"{slice_name}.txt")
                with open(out_lyric_path, "w") as f:
                    f.write(f"Lyrics for {base_name}")
                
                total_clips += 1

        except Exception as e:
            print(f"❌ Error on {filename}: {e}")

    print(f"\n🎉 DONE! Total Dataset Size: {total_clips} clips.")
    print(f"📂 Check folder: {PROCESSED_AUDIO}")

if __name__ == "__main__":
    main()