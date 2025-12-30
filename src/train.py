"""
Training Script

Training loop for the multimodal property valuation model.
Designed to run on GPU (Google Colab recommended).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

from .dataset import PropertyDataset, create_dataloaders
from .models import FusionModel, TabularOnlyModel


class Trainer:
    """
    Trainer class for property valuation models.
    """
    
    def __init__(self, 
                 model,
                 train_loader,
                 val_loader=None,
                 device='cuda',
                 learning_rate=1e-4,
                 weight_decay=1e-5,
                 price_scaler=None):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            train_loader: Training DataLoader
            val_loader: Validation DataLoader (optional)
            device: Device to train on
            learning_rate: Initial learning rate
            weight_decay: L2 regularization
            price_scaler: Scaler to unscale predictions for metrics
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.price_scaler = price_scaler
        
        # Loss and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_rmse': [],
            'val_loss': [],
            'val_rmse': [],
            'val_r2': []
        }
        
        self.best_val_loss = float('inf')
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, (images, tabular, prices) in enumerate(pbar):
            # Move to device
            images = images.to(self.device)
            tabular = tabular.to(self.device)
            prices = prices.to(self.device).unsqueeze(1)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if hasattr(self.model, 'image_encoder'):
                # Fusion model
                outputs = self.model(images, tabular)
            else:
                # Tabular-only model
                outputs = self.model(tabular)
            
            # Compute loss
            loss = self.criterion(outputs, prices)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item() * len(prices)
            total_samples += len(prices)
            
            pbar.set_postfix({'loss': total_loss / total_samples})
        
        avg_loss = total_loss / total_samples
        rmse = np.sqrt(avg_loss)
        
        return avg_loss, rmse
    
    @torch.no_grad()
    def validate(self):
        """Validate the model."""
        if self.val_loader is None:
            return None, None, None
        
        self.model.eval()
        total_loss = 0
        total_samples = 0
        all_predictions = []
        all_targets = []
        
        for images, tabular, prices in self.val_loader:
            images = images.to(self.device)
            tabular = tabular.to(self.device)
            prices = prices.to(self.device).unsqueeze(1)
            
            if hasattr(self.model, 'image_encoder'):
                outputs = self.model(images, tabular)
            else:
                outputs = self.model(tabular)
            
            # Loss is calculated on SCALED values for stability
            loss = self.criterion(outputs, prices)
            
            total_loss += loss.item() * len(prices)
            total_samples += len(prices)
            
            # For metrics, we want UNSCALED values (actual dollars)
            preds = outputs.cpu().numpy().flatten()
            targets = prices.cpu().numpy().flatten()
            
            if self.price_scaler is not None:
                preds = self.price_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
                targets = self.price_scaler.inverse_transform(targets.reshape(-1, 1)).flatten()
            
            all_predictions.extend(preds)
            all_targets.extend(targets)
        
        avg_loss = total_loss / total_samples
        
        # Calculate RMSE on unscaled values
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        mse = np.mean((all_targets - all_predictions) ** 2)
        rmse = np.sqrt(mse)
        
        # Calculate R²
        ss_res = np.sum((all_targets - all_predictions) ** 2)
        ss_tot = np.sum((all_targets - all_targets.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        return avg_loss, rmse, r2
    
    def train(self, num_epochs, save_path=None, early_stopping_patience=10):
        """
        Full training loop.
        
        Args:
            num_epochs: Number of epochs to train
            save_path: Path to save best model
            early_stopping_patience: Epochs without improvement before stopping
        """
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss, train_rmse = self.train_epoch(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['train_rmse'].append(train_rmse)
            
            # Validate
            val_loss, val_rmse, val_r2 = self.validate()
            
            if val_loss is not None:
                self.history['val_loss'].append(val_loss)
                self.history['val_rmse'].append(val_rmse)
                self.history['val_r2'].append(val_r2)
                
                print(f'\nEpoch {epoch}:')
                print(f'  Train Loss: {train_loss:.4f}, Train RMSE: ${train_rmse:,.2f}')
                print(f'  Val Loss: {val_loss:.4f}, Val RMSE: ${val_rmse:,.2f}, Val R²: {val_r2:.4f}')
                
                # Learning rate scheduler
                self.scheduler.step(val_loss)
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    patience_counter = 0
                    
                    if save_path:
                        self.save_model(save_path)
                        print(f'  Saved best model with val_loss: {val_loss:.4f}')
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= early_stopping_patience:
                    print(f'\nEarly stopping after {epoch} epochs')
                    break
            else:
                print(f'\nEpoch {epoch}:')
                print(f'  Train Loss: {train_loss:.4f}, Train RMSE: ${train_rmse:,.2f}')
        
        return self.history
    
    def save_model(self, path):
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss
        }, path)
    
    def load_model(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        self.best_val_loss = checkpoint['best_val_loss']


def train_fusion_model(train_csv, train_image_dir, 
                       val_split=0.2,
                       num_epochs=50,
                       batch_size=32,
                       learning_rate=1e-4,
                       save_path='models/fusion_model.pth',
                       device='cuda'):
    """
    Train the fusion model.
    
    Args:
        train_csv: Path to training CSV
        train_image_dir: Directory with training images
        val_split: Fraction of data to use for validation
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        save_path: Path to save model
        device: Device to train on
    """
    # Check device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    print(f"Training on: {device}")
    
    # Create dataset
    full_dataset = PropertyDataset(
        csv_path=train_csv,
        image_dir=train_image_dir,
        is_train=True
    )
    
    # Split into train/val
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device == 'cuda' else False
    )
    
    # Create model
    num_features = full_dataset.get_feature_dim()
    print(f"Number of tabular features: {num_features}")
    
    # Save price scaler
    if full_dataset.price_scaler is not None:
        model_dir = Path(save_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        scaler_path = model_dir / 'price_scaler.joblib'
        import joblib
        joblib.dump(full_dataset.price_scaler, scaler_path)
        print(f"Saved price scaler to: {scaler_path}")
    
    model = FusionModel(
        tabular_input_dim=num_features,
        pretrained=True,
        freeze_backbone=False
    )
    
    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=learning_rate,
        price_scaler=full_dataset.price_scaler
    )
    
    history = trainer.train(
        num_epochs=num_epochs,
        save_path=save_path,
        early_stopping_patience=10
    )
    
    return model, history


def train_baseline_model(train_csv, 
                         val_split=0.2,
                         num_epochs=100,
                         batch_size=64,
                         learning_rate=1e-3,
                         save_path='models/baseline_model.pth',
                         device='cuda'):
    """
    Train the tabular-only baseline model.
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    # Check device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    
    print(f"Training baseline model on: {device}")
    
    # Load data
    df = pd.read_csv(train_csv)
    
    # Get features
    exclude = ['id', 'price', 'date', 'lat', 'long']
    feature_cols = [c for c in df.columns 
                    if c not in exclude 
                    and df[c].dtype in [np.int64, np.float64]]
    
    X = df[feature_cols].values.astype(np.float32)
    y = df['price'].values.astype(np.float32)
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, random_state=42
    )
    
    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    # Convert to tensors
    X_train = torch.from_numpy(X_train)
    X_val = torch.from_numpy(X_val)
    y_train = torch.from_numpy(y_train)
    y_val = torch.from_numpy(y_val)
    
    # Create datasets
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = TabularOnlyModel(input_dim=len(feature_cols)).to(device)
    
    # Training
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    best_val_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        # Train
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(y_batch)
        
        train_loss /= len(train_dataset)
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device).unsqueeze(1)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * len(y_batch)
        
        val_loss /= len(val_dataset)
        scheduler.step(val_loss)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Train RMSE: ${np.sqrt(train_loss):,.2f}, Val RMSE: ${np.sqrt(val_loss):,.2f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
    
    print(f'\nBest validation RMSE: ${np.sqrt(best_val_loss):,.2f}')
    
    return model


if __name__ == "__main__":
    from pathlib import Path
    
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
    DATA_IMAGES = PROJECT_ROOT / "data" / "images"
    MODELS_DIR = PROJECT_ROOT / "models"
    
    # Train baseline first
    print("="*60)
    print("Training Baseline (Tabular-Only) Model")
    print("="*60)
    
    train_baseline_model(
        train_csv=DATA_PROCESSED / "train.csv",
        save_path=MODELS_DIR / "baseline_model.pth"
    )
    
    print("\n" + "="*60)
    print("Training Fusion Model")
    print("="*60)
    
    train_fusion_model(
        train_csv=DATA_PROCESSED / "train.csv",
        train_image_dir=DATA_IMAGES / "train",
        save_path=MODELS_DIR / "fusion_model.pth"
    )
