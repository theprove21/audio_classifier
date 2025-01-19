import torch
import torchaudio
import random
import numpy as np
import os
import platform

def set_seed(seed):
    """Set seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def save_checkpoint(model, optimizer, epoch, path):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

def load_checkpoint(model, optimizer, path):
    """Load model checkpoint"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch 

def check_environment():
    """Print information about the execution environment"""
    import torch
    import torchaudio
    import platform
    
    # Determine the best available device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tensor = torch.randn(3, 3)

    print("\n=== Environment Information ===")
    print(f"Python version: {platform.python_version()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Torchaudio version: {torchaudio.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Current CUDA device: {tensor.device}")
        print(f"Device name: {torch.cuda.get_device_name()}")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Device properties: {torch.cuda.get_device_properties(0)}")
    
    print(f"Using device: {device}")
    print("===============================\n")
    
   # return device  # Return the device so it can be used elsewhere in the code 