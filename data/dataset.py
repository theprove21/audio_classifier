import torch
from torch.utils.data import Dataset
import torchaudio
import os
import numpy as np
import pandas as pd
import sys

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
        audio_path = self.data[idx]['path']
        label = self.data[idx]['label']
        
        # Load audio file
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Resample if necessary
        if sample_rate != Config.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sample_rate, Config.SAMPLE_RATE)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Pad or trim to fixed duration
        target_length = int(Config.DURATION * Config.SAMPLE_RATE)
        current_length = waveform.shape[1]
        
        if current_length > target_length:
            waveform = waveform[:, :target_length]
        elif current_length < target_length:
            padding = target_length - current_length
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        # Apply transformations if any
        if self.transform:
            waveform = self.transform(waveform)
            
        return waveform, torch.tensor(label) 
if __name__ == "__main__":
    dataset = UrbanSoundDataset(fold=5)
    print("Length of dataset:", len(dataset))
    print("Length of current fold:", dataset.__len_current_fold__())

    print("First sample:", dataset[0])
    print("first sample label:", dataset[0][1])
    print("Sample shape:", dataset[0][0].shape)

