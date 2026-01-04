import os
import torch
from torch.utils.data import Dataset

class MusicDataset(Dataset):
    def __init__(self, feature_dir):
        self.feature_dir = feature_dir
        
        if not os.path.exists(feature_dir):
             raise ValueError(f"❌ Folder not found: {feature_dir}")

        self.feature_files = [f for f in os.listdir(feature_dir) if f.endswith('.pt')]
        
        if len(self.feature_files) == 0:
            raise ValueError(f"❌ No .pt files found!")
            
        print(f"✅ Loaded {len(self.feature_files)} feature tensors.")

    def __len__(self):
        return len(self.feature_files)

    def __getitem__(self, idx):
        feature_path = os.path.join(self.feature_dir, self.feature_files[idx])
        feature_vector = torch.load(feature_path, weights_only=True)
        
        # --- NORMALIZATION FIX ---
        # Neural Networks hate large numbers. We must shrink them.
        # Strategy: Standard Score (Z-score) Normalization
        # Formula: (x - mean) / std_dev
        
        mean = feature_vector.mean()
        std = feature_vector.std()
        
        # Add 1e-6 to avoid dividing by zero if a file is silent
        normalized_vector = (feature_vector - mean) / (std + 1e-6)
        
        return normalized_vector