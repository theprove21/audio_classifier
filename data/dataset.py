import torch
from torch.utils.data import Dataset
import torchaudio
import os
import numpy as np
import pandas as pd
import sys
import time

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config


class UrbanSoundDataset(Dataset):
    def __init__(self, fold, transform=None):
        """
        Args:
            fold (list or int): Which fold(s) to use (1-10 for UrbanSound8K)
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.transform = transform
        
        # Load metadata
        self.metadata = pd.read_csv(Config.METADATA_PATH)
        
        # Convert single fold to list if necessary
        if isinstance(fold, int):
            fold = [fold]
        
        # Filter for the specified folds
        self.metadata_current_fold = self.metadata[self.metadata['fold'].isin(fold)].reset_index(drop=True)
        
        # Get all audio files for the specified folds
        self.data = []
        for _, row in self.metadata_current_fold.iterrows():
            self.data.append({
                'path': os.path.join(Config.DATA_DIR, f'fold{row["fold"]}', row['slice_file_name']),
                'label': Config.CLASS_MAPPING[row['class']]
            })

    def __len__(self):
        # return len(self.metadata)
        return len(self.data)
    
    # def __len_current_fold__(self):
    #     return len(self.data)
    
    def __getitem__(self, idx):
        # Add this print for the first item only
        if idx == 0 and not hasattr(self, '_printed_device'):
            print(f"\nDataset loading info:")
            waveform, sample_rate = torchaudio.load(self.data[idx]['path'])
            print(f"Initial waveform device: {waveform.device}")
            self._printed_device = True
        
        start_time = time.time()

        audio_path = self.data[idx]['path']
        label = self.data[idx]['label']
        
        # Time audio loading
        load_start = time.time()
        waveform, sample_rate = torchaudio.load(audio_path)
        load_time = time.time() - load_start
        
        # Convert to mono if stereo
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Time GPU transfer
        gpu_start = time.time()
        waveform = waveform.to(Config.DEVICE, non_blocking=True)
        gpu_time = time.time() - gpu_start
        
        if sample_rate != Config.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(
                sample_rate, Config.SAMPLE_RATE
            ).to(Config.DEVICE)
            waveform = resampler(waveform)
        
        # Time preprocessing
        prep_start = time.time()
        if self.transform:
            waveform = self.transform(waveform)
        prep_time = time.time() - prep_start
        
        # Create label tensor
        label = torch.tensor(label, device=Config.DEVICE)
        
        # Print times for first few items
        if idx < 3:
            print(f"\nTiming for item {idx}:")
            print(f"Audio load time: {load_time:.3f}s")
            print(f"GPU transfer time: {gpu_time:.3f}s")
            print(f"Preprocessing time: {prep_time:.3f}s")
            print(f"Total time: {time.time() - start_time:.3f}s")
        
        return waveform, label

if __name__ == "__main__":
    dataset = UrbanSoundDataset(fold=5)
    print("Length of dataset:", len(dataset))
    print("Length of current fold:", dataset.__len_current_fold__())

    print("First sample:", dataset[0])
    print("first sample label:", dataset[0][1])
    print("Sample shape:", dataset[0][0].shape)

