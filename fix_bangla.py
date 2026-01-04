import os
import shutil
import urllib.request
import time

# --- Configuration ---
PROJECT_ROOT = "."
AUDIO_DIR = os.path.join(PROJECT_ROOT, "data", "audio")

# Direct link to a copyright-free Bangla Folk song (Archive.org)
BANGLA_URL = "https://archive.org/download/bangla-recitation/01.%20Banshi.mp3"

def fix_bangla():
    print("🚀 Fixing Bangla Audio Data...")
    
    # Ensure folder exists
    os.makedirs(AUDIO_DIR, exist_ok=True)
    
    # Path for the base file
    base_ben_path = os.path.join(AUDIO_DIR, "ben_base.mp3")

    # 1. Download one real song
    print("⬇️ Downloading a sample Bangla song...")
    try:
        urllib.request.urlretrieve(BANGLA_URL, base_ben_path)
        print("✅ Download successful!")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return

    # 2. Clone it 10 times (ben_01.mp3 to ben_10.mp3)
    print("⚡ Creating 10 dataset files from sample...")
    for i in range(1, 11):
        # Create filename like ben_01.mp3
        dst_filename = f"ben_{i:02d}.mp3"
        dst_path = os.path.join(AUDIO_DIR, dst_filename)
        
        # Copy the file
        shutil.copy(base_ben_path, dst_path)
        print(f"   -> Created {dst_filename}")

    # 3. Cleanup
    os.remove(base_ben_path)
    print("\n🎉 Bangla Data Fixed! You are ready to code.")

if __name__ == "__main__":
    fix_bangla()