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

# Direct imports since files are in the same directory structure
from config import Config
from data.dataset import UrbanSoundDataset
from data.preprocessing import AudioPreprocessor
from models.cnn_model import AudioCNN
from utils.utils import check_environment

def train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, epoch, fold):
    """Train for one epoch and evaluate"""
    # Training phase
    model.train()
    train_loss = 0
    
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f'Fold {fold}, Epoch {epoch}')):
        # Move data to GPU here
        data, target = data.to(Config.DEVICE), target.to(Config.DEVICE)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        
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
        # Check environment first
        check_environment()
        
        # Explicitly set the device
        torch.cuda.set_device(0)  # Use first GPU
        print(f"Using device: {torch.cuda.current_device()}")
        
        # Move model to GPU
        model = AudioCNN(Config.NUM_CLASSES).to(Config.DEVICE)
        criterion = torch.nn.CrossEntropyLoss().to(Config.DEVICE)
        optimizer = Adam(model.parameters(), lr=Config.LEARNING_RATE)
        
        # Create directories if they don't exist
        os.makedirs('models/saved', exist_ok=True)
        
        # Initialize preprocessing
        preprocessor = AudioPreprocessor(
            sample_rate=Config.SAMPLE_RATE,
            n_mels=Config.N_MELS,
            duration=Config.DURATION
        )
        
        # Store results for each fold
        fold_accuracies = []
        
        # Perform 10-fold cross validation
        for fold in range(1, 11):
            print(f'\n=== Training on Fold {fold} ===')
            
            train_folds = [f for f in range(1, 11) if f != fold]
            
            train_dataset = UrbanSoundDataset(fold=train_folds, transform=preprocessor.preprocess)
            test_dataset = UrbanSoundDataset(fold=[fold], transform=preprocessor.preprocess)
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=2,
                pin_memory=False,
                persistent_workers=True,
                multiprocessing_context='spawn'
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=2,
                pin_memory=False,
                persistent_workers=True,
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