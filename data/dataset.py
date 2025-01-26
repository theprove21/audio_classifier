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
        import time
        start_time = time.time()
        
        # Time file loading
        load_start = time.time()
        audio_path = self.data[idx]['path']
        waveform, sample_rate = torchaudio.load(audio_path)
        load_time = time.time() - load_start
        
        # Time mono conversion
        mono_start = time.time()
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        mono_time = time.time() - mono_start
        
        # Time resampling
        resample_start = time.time()
        if sample_rate != Config.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(
                sample_rate, Config.SAMPLE_RATE
            ).to(Config.DEVICE)
            waveform = resampler(waveform)
        resample_time = time.time() - resample_start
        
        # Time GPU transfer
        gpu_start = time.time()
        waveform = waveform.to(Config.DEVICE, non_blocking=True)
        gpu_time = time.time() - gpu_start
        
        # Time preprocessing
        prep_start = time.time()
        if self.transform:
            waveform = self.transform(waveform)
        prep_time = time.time() - prep_start
        
        # Time label creation
        label_start = time.time()
        label = torch.tensor(self.data[idx]['label'], device=Config.DEVICE)
        label_time = time.time() - label_start
        
        total_time = time.time() - start_time
        
        # Print detailed timing for first few items
        if not hasattr(self, '_printed_times'):
            self._printed_times = 0
        
        if self._printed_times < 3:
            print(f"\nDetailed timing for item {idx}:")
            print(f"File loading: {load_time:.3f}s")
            print(f"Mono conversion: {mono_time:.3f}s")
            print(f"Resampling: {resample_time:.3f}s")
            print(f"GPU transfer: {gpu_time:.3f}s")
            print(f"Preprocessing: {prep_time:.3f}s")
            print(f"Label creation: {label_time:.3f}s")
            print(f"Total time: {total_time:.3f}s")
            self._printed_times += 1
        
        return waveform, label

if __name__ == "__main__":
    dataset = UrbanSoundDataset(fold=5)
    print("Length of dataset:", len(dataset))
    print("Length of current fold:", dataset.__len_current_fold__())

    print("First sample:", dataset[0])
    print("first sample label:", dataset[0][1])
    print("Sample shape:", dataset[0][0].shape)

