"""
Data Preprocessing Module

This module handles loading, cleaning, and preprocessing the tabular property data.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


def load_raw_data():
    """Load raw Excel files."""
    train_path = DATA_RAW / "train(1).xlsx"
    test_path = DATA_RAW / "test2.xlsx"
    
    print(f"Loading training data from: {train_path}")
    train_df = pd.read_excel(train_path)
    
    print(f"Loading test data from: {test_path}")
    test_df = pd.read_excel(test_path)
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    return train_df, test_df


def inspect_data(df, name="Dataset"):
    """Print basic information about a DataFrame."""
    print(f"\n{'='*50}")
    print(f"Inspecting: {name}")
    print(f"{'='*50}")
    
    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nBasic Statistics:\n{df.describe()}")
    
    return df.info()


def clean_data(df, is_train=True):
    """
    Clean and preprocess the data.
    
    Args:
        df: Input DataFrame
        is_train: Whether this is training data (has 'price' column)
    
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    
    # Create unique ID if not present
    if 'id' not in df.columns:
        df['id'] = range(len(df))
    
    # Ensure lat/long columns exist (required for satellite images)
    lat_cols = [c for c in df.columns if 'lat' in c.lower()]
    long_cols = [c for c in df.columns if 'long' in c.lower() or 'lng' in c.lower()]
    
    if lat_cols:
        df['lat'] = df[lat_cols[0]]
    if long_cols:
        df['long'] = df[long_cols[0]]
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            # Fill with median for numeric columns
            df[col] = df[col].fillna(df[col].median())
    
    # Handle categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna('Unknown')
    
    # Remove extreme outliers in price (for training data only)
    if is_train and 'price' in df.columns:
        q1 = df['price'].quantile(0.01)
        q99 = df['price'].quantile(0.99)
        original_len = len(df)
        df = df[(df['price'] >= q1) & (df['price'] <= q99)]
        print(f"Removed {original_len - len(df)} outliers from price column")
    
    return df


def get_feature_columns(df):
    """
    Get list of feature columns to use for modeling.
    
    Excludes id, date, and target columns.
    """
    exclude_cols = ['id', 'price', 'date', 'lat', 'long']
    exclude_patterns = ['unnamed', 'index']
    
    feature_cols = []
    for col in df.columns:
        col_lower = col.lower()
        if col not in exclude_cols and not any(p in col_lower for p in exclude_patterns):
            if df[col].dtype in [np.int64, np.float64, np.int32, np.float32]:
                feature_cols.append(col)
    
    return feature_cols


def prepare_features(train_df, test_df, feature_cols=None):
    """
    Prepare and scale features for modeling.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        feature_cols: List of feature columns to use
    
    Returns:
        Scaled train/test features and fitted scaler
    """
    if feature_cols is None:
        feature_cols = get_feature_columns(train_df)
    
    print(f"Using {len(feature_cols)} features: {feature_cols}")
    
    # Extract features
    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    scaler_path = MODELS_DIR / "feature_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler to: {scaler_path}")
    
    # Save feature columns list
    feature_cols_path = MODELS_DIR / "feature_columns.joblib"
    joblib.dump(feature_cols, feature_cols_path)
    print(f"Saved feature columns to: {feature_cols_path}")
    
    return X_train_scaled, X_test_scaled, scaler, feature_cols


def load_and_process():
    """
    Main function to load, clean, and process all data.
    
    Returns:
        Tuple of (train_df, test_df)
    """
    # Ensure output directory exists
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    
    # Load raw data
    train_df, test_df = load_raw_data()
    
    # Inspect data
    inspect_data(train_df, "Training Data")
    inspect_data(test_df, "Test Data")
    
    # Clean data
    print("\nCleaning training data...")
    train_clean = clean_data(train_df, is_train=True)
    
    print("\nCleaning test data...")
    test_clean = clean_data(test_df, is_train=False)
    
    # Save processed data
    train_path = DATA_PROCESSED / "train.csv"
    test_path = DATA_PROCESSED / "test.csv"
    
    train_clean.to_csv(train_path, index=False)
    test_clean.to_csv(test_path, index=False)
    
    print(f"\nSaved processed training data to: {train_path}")
    print(f"Saved processed test data to: {test_path}")
    
    return train_clean, test_clean


if __name__ == "__main__":
    train_df, test_df = load_and_process()
    
    # Prepare features
    print("\n" + "="*50)
    print("Preparing features for modeling...")
    print("="*50)
    
    X_train, X_test, scaler, feature_cols = prepare_features(train_df, test_df)
    
    print(f"\nTraining features shape: {X_train.shape}")
    print(f"Test features shape: {X_test.shape}")
    print("\nPreprocessing complete!")
