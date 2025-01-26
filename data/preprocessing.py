import torchaudio
import torch
import numpy as np
from config import Config

class AudioPreprocessor:
    def __init__(self, sample_rate=22050, n_mels=128, duration=4):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.duration = duration
        
        # Initialize mel spectrogram transform and move it to the correct device
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=n_mels
        ).to(Config.DEVICE)
        
        # Initialize amplitude to DB transform and move it to GPU
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB().to(Config.DEVICE)
        
        # Print device information for transforms
        print(f"\nPreprocessing transforms devices:")
        print(f"Mel spectrogram parameters device: {next(self.mel_spectrogram.parameters()).device if list(self.mel_spectrogram.parameters()) else 'No parameters'}")
        print(f"Amplitude to DB parameters device: {next(self.amplitude_to_db.parameters()).device if list(self.amplitude_to_db.parameters()) else 'No parameters'}")
    
    def preprocess(self, waveform):
        """
        Convert waveform to mel spectrogram and apply necessary preprocessing
        """
        # Add this print for the first call only
        if not hasattr(self, '_printed_device'):
            print(f"Waveform device before preprocessing: {waveform.device}")
            self._printed_device = True
        
        # Ensure waveform is on the correct device
        waveform = waveform.to(Config.DEVICE)
        
        # Ensure consistent length
        target_length = int(self.sample_rate * self.duration)
        if waveform.size(1) > target_length:
            waveform = waveform[:, :target_length]
        else:
            waveform = torch.nn.functional.pad(
                waveform, (0, target_length - waveform.size(1))
            )
            
        # Convert to mel spectrogram
        mel_spec = self.mel_spectrogram(waveform)
        
        # Convert to dB scale
        mel_spec_db = self.amplitude_to_db(mel_spec)
        
        # Normalize
        mel_spec_db = (mel_spec_db + 80) / 80  # Normalize approximately to [0,1]
        
        return mel_spec_db 