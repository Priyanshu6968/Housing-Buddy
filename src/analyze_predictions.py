
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from tqdm import tqdm
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.dataset import PropertyDataset, create_dataloaders
from src.models import FusionModel

def analyze_model():
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
    DATA_IMAGES = PROJECT_ROOT / "data" / "images"
    MODELS_DIR = PROJECT_ROOT / "models"
    MODEL_PATH = MODELS_DIR / "fusion_model.pth"
    SCALER_PATH = MODELS_DIR / "feature_scaler.joblib"
    FEATURE_COLS_PATH = MODELS_DIR / "feature_columns.joblib"
    PRICE_SCALER_PATH = MODELS_DIR / "price_scaler.joblib"
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load metadata
    print("Loading scaler and feature columns...")
    scaler = joblib.load(SCALER_PATH)
    feature_cols = joblib.load(FEATURE_COLS_PATH)
    
    # Load price scaler if exists
    price_scaler = None
    if PRICE_SCALER_PATH.exists():
        price_scaler = joblib.load(PRICE_SCALER_PATH)
        print("Loaded price scaler.")
    else:
        print("No price scaler found (dataset might fit its own or return raw).")
    
    # Create dataset
    print("Creating dataset...")
    # Using training data to check
    dataset = PropertyDataset(
        csv_path=DATA_PROCESSED / "train.csv",
        image_dir=DATA_IMAGES / "train",
        feature_columns=feature_cols,
        scaler=scaler,
        is_train=True,
        price_scaler=price_scaler
    )
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Load model
    print("Loading model...")
    num_features = len(feature_cols)
    model = FusionModel(
        tabular_input_dim=num_features,
        pretrained=True
    ).to(device)
    
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from {MODEL_PATH}")
    
    # Run predictions
    print("Running predictions...")
    all_preds = []
    all_actuals = []
    
    with torch.no_grad():
        for images, tabular, prices in tqdm(dataloader):
            images = images.to(device)
            tabular = tabular.to(device)
            
            outputs = model(images, tabular)
            
            # Unscale if needed
            preds = outputs.cpu().numpy().flatten()
            targets = prices.numpy().flatten()
            
            if price_scaler is not None:
                preds = price_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
                # Targets in dataset are already scaled if price_scaler was passed
                targets = price_scaler.inverse_transform(targets.reshape(-1, 1)).flatten()
            
            all_preds.extend(preds)
            all_actuals.extend(targets)
            
            if len(all_preds) > 1000: # Analyze first 1000 samples to be quick
                break
    
    all_preds = np.array(all_preds)
    all_actuals = np.array(all_actuals)
    
    # Analysis
    print("\n" + "="*50)
    print("PREDICTION ANALYSIS")
    print("="*50)
    
    print(f"\nStats (N={len(all_preds)}):")
    print(f"Actual Mean:    ${np.mean(all_actuals):,.2f}")
    print(f"Predicted Mean: ${np.mean(all_preds):,.2f}")
    print(f"Difference:     ${np.mean(all_preds) - np.mean(all_actuals):,.2f}")
    
    print(f"\nActual Median:    ${np.median(all_actuals):,.2f}")
    print(f"Predicted Median: ${np.median(all_preds):,.2f}")
    
    print(f"\nMin Actual: ${np.min(all_actuals):,.2f}")
    print(f"Max Actual: ${np.max(all_actuals):,.2f}")
    
    print(f"\nMin Predicted: ${np.min(all_preds):,.2f}")
    print(f"Max Predicted: ${np.max(all_preds):,.2f}")
    
    # Sample comparison
    print("\nSample Comparisons (First 10):")
    print(f"{'Actual':<15} | {'Predicted':<15} | {'Diff':<15} | {'Error %':<10}")
    print("-" * 65)
    
    for i in range(10):
        act = all_actuals[i]
        pred = all_preds[i]
        diff = pred - act
        err_pct = (diff / act) * 100 if act != 0 else 0
        
        print(f"${act:<14,.0f} | ${pred:<14,.0f} | ${diff:<14,.0f} | {err_pct:>8.2f}%")

if __name__ == "__main__":
    analyze_model()
