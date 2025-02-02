import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import numpy as np
import platform
import sys
import os
import gc
from torch.cuda.amp import autocast, GradScaler
import torch.multiprocessing as mp
import time

# Direct imports since files are in the same directory structure
from config import Config
from data.dataset import UrbanSoundDataset
from data.preprocessing import AudioPreprocessor
from models.cnn_model import AudioCNN
from utils.utils import check_environment

def train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, epoch, fold):
    """Train for one epoch and evaluate"""
    model.train()
    train_loss = 0
    
    # Time the entire epoch
    epoch_start = time.time()
    batch_times = []
    data_wait_times = []
    
    # Enable CUDA profiling
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f'Fold {fold}, Epoch {epoch}')):
            batch_start = time.time()
            data_wait_time = batch_start - (batch_times[-1] if batch_times else epoch_start)
            data_wait_times.append(data_wait_time)
            
            data = data.contiguous()
            target = target.contiguous()
            
            if batch_idx == 0:
                print(f"\nFirst batch info:")
                print(f"Input data device: {data.device}")
                print(f"Target device: {target.device}")
                print(f"Model device: {next(model.parameters()).device}")
                print(f"Input data shape: {data.shape}")
                print(f"CUDA memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
                print(f"CUDA memory cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")
                print(f"CUDA max memory allocated: {torch.cuda.max_memory_allocated()/1e9:.2f} GB\n")
            
            # Time data loading
            torch.cuda.synchronize()
            data_time = time.time() - batch_start
            
            optimizer.zero_grad(set_to_none=True)
            
            # Time forward pass
            forward_start = time.time()
            with torch.amp.autocast('cuda'):
                output = model(data)
                loss = criterion(output, target)
            torch.cuda.synchronize()
            forward_time = time.time() - forward_start
            
            # Time backward pass
            backward_start = time.time()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            backward_time = time.time() - backward_start
            
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            
            if batch_idx == 0:
                print(f"\nDetailed GPU timing:")
                print(f"Data wait time: {data_wait_time:.2f}s")
                print(f"Forward time: {forward_time:.2f}s")
                print(f"Backward time: {backward_time:.2f}s")
                print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB")
                print(f"GPU memory reserved: {torch.cuda.memory_reserved()/1e9:.2f}GB")
                print(f"Max memory allocated: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
                
                # Print model parameter gradients stats
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        print(f"{name} grad stats: mean={param.grad.mean():.3e}, std={param.grad.std():.3e}")
            
            prof.step()
            
            if batch_idx == 2:  # Print profile after a few iterations
                print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
                break
            
            train_loss += loss.item()
    
    # Validation phase
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(Config.DEVICE), target.to(Config.DEVICE)
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    # Calculate metrics
    train_loss = train_loss / len(train_loader)
    val_loss = val_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    return train_loss, val_loss, accuracy

def cleanup():
    """Clean up CUDA memory"""
    gc.collect()
    torch.cuda.empty_cache()

def train_model():
    try:
        # Set multiprocessing start method to 'spawn'
        if mp.get_start_method(allow_none=True) != 'spawn':
            mp.set_start_method('spawn', force=True)
        
        # Check environment first
        check_environment()
        
        # Set CUDA optimizations from config
        torch.backends.cudnn.benchmark = Config.CUDNN_BENCHMARK
        torch.backends.cuda.matmul.allow_tf32 = Config.ALLOW_TF32
        torch.backends.cudnn.allow_tf32 = Config.ALLOW_TF32
        
        # Create directories if they don't exist
        os.makedirs('models/saved', exist_ok=True)
        
        # Initialize model and move to GPU
        model = AudioCNN(Config.NUM_CLASSES).to(Config.DEVICE)
        print(f"\nModel device: {next(model.parameters()).device}")
        
        criterion = torch.nn.CrossEntropyLoss().to(Config.DEVICE)
        # For criterion, check its parameters instead
        print(f"Criterion parameters device: {next(criterion.parameters()).device if list(criterion.parameters()) else 'No parameters'}")
        
        optimizer = Adam(model.parameters(), lr=Config.LEARNING_RATE)
        
        # Initialize preprocessing
        preprocessor = AudioPreprocessor(
            sample_rate=Config.SAMPLE_RATE,
            n_mels=Config.N_MELS,
            duration=Config.DURATION
        )
        
        # Store results for each fold
        fold_accuracies = []
        
        # Increase batch size for better GPU utilization
        # BATCH_SIZE = 128  # or 256 depending on memory
        
        # Create CPU generator for DataLoader
        g = torch.Generator()
        g.manual_seed(Config.RANDOM_SEED)
        
        # Perform 10-fold cross validation
        for fold in range(1, 11):
            print(f'\n=== Training on Fold {fold} ===')
            
            train_folds = [f for f in range(1, 11) if f != fold]
            
            train_dataset = UrbanSoundDataset(fold=train_folds, transform=preprocessor.preprocess)
            test_dataset = UrbanSoundDataset(fold=[fold], transform=preprocessor.preprocess)
            
            print("Pre-processing dataset...")
            for idx in tqdm(range(len(train_dataset)), desc="Preprocessing"):
                _ = train_dataset[idx]
            print("Preprocessing complete!")
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=Config.PIN_MEMORY,
                persistent_workers=Config.PERSISTENT_WORKERS,
                prefetch_factor=Config.PREFETCH_FACTOR,
                drop_last=Config.DROP_LAST,
                generator=g,  # Use CPU generator
                multiprocessing_context='spawn'
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=Config.PIN_MEMORY,
                persistent_workers=Config.PERSISTENT_WORKERS,
                prefetch_factor=Config.PREFETCH_FACTOR,
                drop_last=Config.DROP_LAST,
                generator=g,  # Use CPU generator
                multiprocessing_context='spawn'
            )
            
            # Training loop for this fold
            best_accuracy = 0
            for epoch in range(Config.NUM_EPOCHS):
                train_loss, val_loss, accuracy = train_and_evaluate(
                    model, train_loader, test_loader, optimizer, criterion, epoch, fold
                )
                
                print(f'Fold: {fold}, Epoch: {epoch}')
                print(f'Training Loss: {train_loss:.4f}')
                print(f'Validation Loss: {val_loss:.4f}')
                print(f'Validation Accuracy: {accuracy:.2f}%')
                
                # Save best model for this fold
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    torch.save(model.state_dict(), f'models/saved/best_model_fold_{fold}.pt')
            
            # Store the best accuracy for this fold
            fold_accuracies.append(best_accuracy)
            
            print(f'\nBest accuracy for fold {fold}: {best_accuracy:.2f}%')
        
        # Calculate and print cross-validation results
        mean_accuracy = np.mean(fold_accuracies)
        std_accuracy = np.std(fold_accuracies)
        
        print('\n=== Cross-validation Results ===')
        print(f'Individual fold accuracies: {fold_accuracies}')
        print(f'Mean accuracy: {mean_accuracy:.2f}% ±{std_accuracy:.2f}%')
        
        # Save final results
        results = {
            'fold_accuracies': fold_accuracies,
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy
        }
        torch.save(results, 'models/saved/cross_validation_results.pt')
    finally:
        cleanup()

if __name__ == "__main__":
    train_model() 