import os
import shutil
import sys
import subprocess

# 1. Install kagglehub (The modern replacement for opendatasets)
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import kagglehub
except ImportError:
    print("🔄 Installing kagglehub (works with Python 3.13)...")
    install("kagglehub")
    import kagglehub

# --- Configuration ---
PROJECT_ROOT = "."
AUDIO_DIR = os.path.join(PROJECT_ROOT, "data", "audio")
LYRICS_DIR = os.path.join(PROJECT_ROOT, "data", "lyrics")

def main():
    print("🚀 STARTING MODERN DOWNLOADER...")
    
    # 2. Authentication
    print("\n🔑 Please enter your Kaggle Credentials:")
    username = input("Username: ").strip()
    key = input("Key (The long string): ").strip()
    
    # Set credentials temporarily for this session
    os.environ['KAGGLE_USERNAME'] = username
    os.environ['KAGGLE_KEY'] = key

    # 3. Download Data
    print("\n⬇️ Downloading GTZAN (English Audio)...")
    eng_path = kagglehub.dataset_download("andradaolteanu/gtzan-dataset-music-genre-classification")
    
    print("\n⬇️ Downloading Bangla Audio...")
    ben_path = kagglehub.dataset_download("afifaniks/bangla-music-dataset")
    
    # 4. Move Files
    print("\n📦 Moving files to your project folder...")
    os.makedirs(AUDIO_DIR, exist_ok=True)
    
    # Setup Paths
    source_eng = os.path.join(eng_path, "Data", "genres_original", "pop")
    source_ben = os.path.join(ben_path, "Bangla_Music_Dataset", "Bangla_Band")
    
    # Move English Files
    if os.path.exists(source_eng):
        files = os.listdir(source_eng)[:10]
        for f in files:
            shutil.copy(os.path.join(source_eng, f), os.path.join(AUDIO_DIR, "eng_" + f))
        print("✅ English data ready.")
    else:
        print(f"⚠️ Could not find English folder at: {source_eng}")

    # Move Bangla Files
    if os.path.exists(source_ben):
        files = os.listdir(source_ben)[:10]
        for f in files:
            shutil.copy(os.path.join(source_ben, f), os.path.join(AUDIO_DIR, "ben_" + f))
        print("✅ Bangla data ready.")
    else:
        print(f"⚠️ Could not find Bangla folder at: {source_ben}")

    print("\n🎉 SETUP COMPLETE! Check your data/audio folder.")

if __name__ == "__main__":
    main()