"""
Inference Utilities

Functions for model inference and prediction.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from PIL import Image
from torchvision import transforms


def load_model(model_path, model_class, model_kwargs, device='cpu'):
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        model_class: Model class to instantiate
        model_kwargs: Keyword arguments for model initialization
        device: Device to load model on
    
    Returns:
        Loaded model in eval mode
    """
    model = model_class(**model_kwargs)
    
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model


def preprocess_image(image_path, image_size=224):
    """
    Preprocess a single image for inference.
    
    Args:
        image_path: Path to image file
        image_size: Target image size
    
    Returns:
        Preprocessed image tensor (1, 3, H, W)
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    
    return img_tensor


def preprocess_tabular(features_dict, scaler, feature_columns):
    """
    Preprocess tabular features for inference.
    
    Args:
        features_dict: Dictionary of feature name -> value
        scaler: Fitted StandardScaler
        feature_columns: List of feature column names (in order)
    
    Returns:
        Preprocessed features tensor (1, num_features)
    """
    # Extract features in correct order
    features = np.array([[features_dict.get(col, 0) for col in feature_columns]], 
                        dtype=np.float32)
    
    # Scale
    features_scaled = scaler.transform(features)
    
    # Convert to tensor
    features_tensor = torch.from_numpy(features_scaled)
    
    return features_tensor


def predict_single(model, image_tensor, tabular_tensor, device='cpu'):
    """
    Make a prediction for a single sample.
    
    Args:
        model: Trained model
        image_tensor: Preprocessed image tensor
        tabular_tensor: Preprocessed tabular tensor
        device: Device to use
    
    Returns:
        Predicted price
    """
    model.eval()
    
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        tabular_tensor = tabular_tensor.to(device)
        
        prediction = model(image_tensor, tabular_tensor)
    
    return prediction.item()


def predict_batch(model, dataloader, device='cpu'):
    """
    Make predictions for a batch of samples.
    
    Args:
        model: Trained model
        dataloader: DataLoader with samples
        device: Device to use
    
    Returns:
        numpy array of predictions
    """
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for images, tabular, _ in dataloader:
            images = images.to(device)
            tabular = tabular.to(device)
            
            preds = model(images, tabular)
            predictions.extend(preds.cpu().numpy().flatten())
    
    return np.array(predictions)


def generate_submission(model, test_loader, output_path, device='cpu'):
    """
    Generate submission CSV with predictions.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        output_path: Path to save submission CSV
        device: Device to use
    """
    predictions = predict_batch(model, test_loader, device)
    
    # Get IDs from dataset
    dataset = test_loader.dataset
    if hasattr(dataset, 'dataset'):
        # Handle Subset
        dataset = dataset.dataset
    
    ids = dataset.ids
    
    # Create submission DataFrame
    submission = pd.DataFrame({
        'id': ids[:len(predictions)],
        'predicted_price': predictions
    })
    
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to: {output_path}")
    
    return submission


class PropertyPredictor:
    """
    High-level predictor class for property valuation.
    """
    
    def __init__(self, 
                 model_path,
                 scaler_path,
                 feature_columns_path,
                 device='cpu'):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to trained model
            scaler_path: Path to fitted scaler
            feature_columns_path: Path to feature columns list
            device: Device to use
        """
        self.device = device
        
        # Load scaler and feature columns
        self.scaler = joblib.load(scaler_path)
        self.feature_columns = joblib.load(feature_columns_path)
        
        # Load model
        from .models import FusionModel
        
        self.model = FusionModel(
            tabular_input_dim=len(self.feature_columns),
            pretrained=False  # Don't need pretrained for inference
        )
        
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model = self.model.to(device)
        self.model.eval()
        
        # Image transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def predict(self, image_path, features_dict):
        """
        Predict property price.
        
        Args:
            image_path: Path to satellite image
            features_dict: Dictionary of property features
        
        Returns:
            Predicted price
        """
        # Preprocess image
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Preprocess tabular
        features = np.array([[features_dict.get(col, 0) for col in self.feature_columns]], 
                            dtype=np.float32)
        features_scaled = self.scaler.transform(features)
        tab_tensor = torch.from_numpy(features_scaled).to(self.device)
        
        # Predict
        with torch.no_grad():
            prediction = self.model(img_tensor, tab_tensor)
        
        return prediction.item()
    
    def predict_from_image_and_array(self, image_tensor, features_array):
        """
        Predict from preprocessed tensors.
        
        Args:
            image_tensor: Image tensor (1, 3, 224, 224)
            features_array: Feature array (1, num_features)
        
        Returns:
            Predicted price
        """
        # Scale features
        features_scaled = self.scaler.transform(features_array)
        tab_tensor = torch.from_numpy(features_scaled.astype(np.float32)).to(self.device)
        
        img_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            prediction = self.model(img_tensor, tab_tensor)
        
        return prediction.item()


if __name__ == "__main__":
    print("Inference module loaded successfully.")
