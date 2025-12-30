"""
PyTorch Dataset Classes

Custom dataset classes for loading tabular + image data.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
import joblib
from torchvision import transforms


class PropertyDataset(Dataset):
    """
    PyTorch Dataset for property valuation.
    
    Returns (image_tensor, tabular_tensor, price) for each sample.
    """
    
    def __init__(self, 
                 csv_path,
                 image_dir,
                 feature_columns=None,
                 scaler=None,
                 transform=None,
                 is_train=True,
                 image_size=224,
                 price_scaler=None):
        """
        Initialize the dataset.
        
        Args:
            csv_path: Path to processed CSV file
            image_dir: Directory containing satellite images
            feature_columns: List of feature column names
            scaler: Fitted StandardScaler (or None to fit new one)
            transform: Image transforms (or None for default)
            is_train: Whether this is training data
            image_size: Size to resize images to
            price_scaler: Scaler for price target (optional)
        """
        self.csv_path = Path(csv_path)
        self.image_dir = Path(image_dir)
        self.is_train = is_train
        self.image_size = image_size
        self.price_scaler = price_scaler
        
        # Load data
        self.df = pd.read_csv(csv_path)
        
        # Set up image transforms
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform
        
        # Set up feature columns
        if feature_columns is None:
            self.feature_columns = self._get_default_features()
        else:
            self.feature_columns = feature_columns
        
        # Prepare tabular features
        self.tabular_features = self.df[self.feature_columns].values.astype(np.float32)
        
        # Set up scaler
        if scaler is None and is_train:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            self.tabular_features = self.scaler.fit_transform(self.tabular_features)
        elif scaler is not None:
            self.scaler = scaler
            self.tabular_features = self.scaler.transform(self.tabular_features)
        else:
            self.scaler = None
        
        # Get target if available
        if 'price' in self.df.columns:
            prices = self.df['price'].values.astype(np.float32).reshape(-1, 1)
            
            # Scale prices
            if self.price_scaler is None and is_train:
                from sklearn.preprocessing import StandardScaler
                self.price_scaler = StandardScaler()
                self.prices = self.price_scaler.fit_transform(prices).flatten()
            elif self.price_scaler is not None:
                self.prices = self.price_scaler.transform(prices).flatten()
            else:
                self.prices = prices.flatten()
        else:
            self.prices = None
        
        # Get IDs for image lookup
        self.ids = self.df['id'].values
        
        # Create placeholder image for missing images
        self.placeholder = self._create_placeholder()
    
    def _get_default_features(self):
        """Get default feature columns from the dataframe."""
        exclude = ['id', 'price', 'date', 'lat', 'long']
        exclude_patterns = ['unnamed', 'index']
        
        features = []
        for col in self.df.columns:
            if col.lower() not in [e.lower() for e in exclude]:
                if not any(p in col.lower() for p in exclude_patterns):
                    if self.df[col].dtype in [np.int64, np.float64, np.int32, np.float32]:
                        features.append(col)
        
        return features
    
    def _create_placeholder(self):
        """Create a placeholder tensor for missing images."""
        placeholder = torch.zeros(3, self.image_size, self.image_size)
        return placeholder
    
    def _load_image(self, idx):
        """Load and transform an image."""
        prop_id = self.ids[idx]
        image_path = self.image_dir / f"{prop_id}.png"
        
        if image_path.exists():
            try:
                img = Image.open(image_path).convert('RGB')
                img_tensor = self.transform(img)
                return img_tensor
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                return self.placeholder
        else:
            return self.placeholder
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Get a single sample.
        
        Returns:
            image_tensor: (3, H, W) tensor
            tabular_tensor: (num_features,) tensor
            price: scalar (or 0 if not available)
        """
        # Load image
        image_tensor = self._load_image(idx)
        
        # Get tabular features
        tabular_tensor = torch.from_numpy(self.tabular_features[idx])
        
        # Get price
        if self.prices is not None:
            price = torch.tensor(self.prices[idx])
        else:
            price = torch.tensor(0.0)
        
        return image_tensor, tabular_tensor, price
    
    def get_feature_dim(self):
        """Get the number of tabular features."""
        return len(self.feature_columns)


class PropertyDatasetImageOnly(Dataset):
    """
    Dataset that returns only images (for CNN pretraining or Grad-CAM).
    """
    
    def __init__(self, csv_path, image_dir, transform=None, image_size=224):
        self.csv_path = Path(csv_path)
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        
        self.df = pd.read_csv(csv_path)
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform
        
        self.ids = self.df['id'].values
        
        if 'price' in self.df.columns:
            self.prices = self.df['price'].values.astype(np.float32)
        else:
            self.prices = None
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        prop_id = self.ids[idx]
        image_path = self.image_dir / f"{prop_id}.png"
        
        if image_path.exists():
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img)
        else:
            img_tensor = torch.zeros(3, self.image_size, self.image_size)
        
        if self.prices is not None:
            price = torch.tensor(self.prices[idx])
        else:
            price = torch.tensor(0.0)
        
        return img_tensor, price


def create_dataloaders(train_csv, train_image_dir, 
                       test_csv=None, test_image_dir=None,
                       batch_size=32, num_workers=4,
                       feature_columns=None):
    """
    Create train and test DataLoaders.
    
    Args:
        train_csv: Path to training CSV
        train_image_dir: Directory with training images
        test_csv: Path to test CSV (optional)
        test_image_dir: Directory with test images (optional)
        batch_size: Batch size
        num_workers: Number of data loading workers
        feature_columns: List of feature columns to use
    
    Returns:
        train_loader, test_loader (or None), scaler, feature_columns
    """
    # Create training dataset
    train_dataset = PropertyDataset(
        csv_path=train_csv,
        image_dir=train_image_dir,
        feature_columns=feature_columns,
        is_train=True
    )
    
    # Get scaler and feature columns from training
    scaler = train_dataset.scaler
    feature_cols = train_dataset.feature_columns
    
    # Create training loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Create test dataset if provided
    if test_csv is not None and test_image_dir is not None:
        test_dataset = PropertyDataset(
            image_dir=test_image_dir,
            feature_columns=feature_cols,
            scaler=scaler,
            is_train=False,
            price_scaler=train_dataset.price_scaler
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        test_loader = None
    
    return train_loader, test_loader, scaler, feature_cols, train_dataset.price_scaler


if __name__ == "__main__":
    # Test the dataset
    from pathlib import Path
    
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
    DATA_IMAGES = PROJECT_ROOT / "data" / "images"
    
    print("Testing PropertyDataset...")
    
    dataset = PropertyDataset(
        csv_path=DATA_PROCESSED / "train.csv",
        image_dir=DATA_IMAGES / "train",
        is_train=True
    )
    
    print(f"Dataset length: {len(dataset)}")
    print(f"Feature columns: {dataset.feature_columns}")
    print(f"Feature dimension: {dataset.get_feature_dim()}")
    
    # Test loading a sample
    img, tab, price = dataset[0]
    print(f"\nSample shapes:")
    print(f"  Image: {img.shape}")
    print(f"  Tabular: {tab.shape}")
    print(f"  Price: {price}")
