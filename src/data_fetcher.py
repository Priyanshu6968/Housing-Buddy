"""
Satellite Image Data Fetcher

This module handles downloading satellite imagery for property locations.
Supports Mapbox and Google Maps Static APIs.
"""

import os
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
from dotenv import load_dotenv
from PIL import Image
import io

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_IMAGES = PROJECT_ROOT / "data" / "images"


class SatelliteImageFetcher:
    """
    Fetches satellite imagery from Mapbox or Google Maps APIs.
    """
    
    def __init__(self, provider='mapbox', image_size=256, zoom=18):
        """
        Initialize the fetcher.
        
        Args:
            provider: 'mapbox' or 'google'
            image_size: Size of images to fetch (256 or 512)
            zoom: Zoom level (15-20, higher = more detail)
        """
        self.provider = provider
        self.image_size = image_size
        self.zoom = zoom
        
        # Get API keys
        self.mapbox_token = os.getenv('MAPBOX_ACCESS_TOKEN')
        self.google_key = os.getenv('GOOGLE_MAPS_API_KEY')
        
        # Validate
        if provider == 'mapbox' and not self.mapbox_token:
            print("WARNING: MAPBOX_ACCESS_TOKEN not found in environment")
        elif provider == 'google' and not self.google_key:
            print("WARNING: GOOGLE_MAPS_API_KEY not found in environment")
    
    def get_mapbox_url(self, lat, lon):
        """Generate Mapbox Static Images API URL."""
        return (
            f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/"
            f"{lon},{lat},{self.zoom},0/{self.image_size}x{self.image_size}"
            f"?access_token={self.mapbox_token}"
        )
    
    def get_google_url(self, lat, lon):
        """Generate Google Maps Static API URL."""
        return (
            f"https://maps.googleapis.com/maps/api/staticmap?"
            f"center={lat},{lon}&zoom={self.zoom}&size={self.image_size}x{self.image_size}"
            f"&maptype=satellite&key={self.google_key}"
        )
    
    def fetch_image(self, lat, lon, save_path, max_retries=3):
        """
        Fetch a single satellite image.
        
        Args:
            lat: Latitude
            lon: Longitude
            save_path: Path to save the image
            max_retries: Number of retry attempts
        
        Returns:
            True if successful, False otherwise
        """
        # Skip if already exists
        if save_path.exists():
            return True
        
        # Get URL based on provider
        if self.provider == 'mapbox':
            url = self.get_mapbox_url(lat, lon)
        else:
            url = self.get_google_url(lat, lon)
        
        # Fetch with retries
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=30)
                
                if response.status_code == 200:
                    # Verify it's a valid image
                    img = Image.open(io.BytesIO(response.content))
                    
                    # Check if image is mostly black/error
                    img_array = np.array(img)
                    if img_array.mean() < 5:  # Mostly black
                        print(f"Warning: Image appears to be blank for ({lat}, {lon})")
                        return False
                    
                    # Save image
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    img.save(save_path)
                    return True
                
                elif response.status_code == 429:  # Rate limited
                    wait_time = (attempt + 1) * 5
                    print(f"Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                
                else:
                    print(f"Error {response.status_code} for ({lat}, {lon})")
                    return False
                    
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for ({lat}, {lon}): {e}")
                time.sleep(2)
        
        return False
    
    def fetch_dataset(self, df, output_dir, id_col='id', lat_col='lat', lon_col='long'):
        """
        Fetch satellite images for an entire dataset.
        
        Args:
            df: DataFrame with lat/long coordinates
            output_dir: Directory to save images
            id_col: Column name for unique IDs
            lat_col: Column name for latitude
            lon_col: Column name for longitude
        
        Returns:
            DataFrame with image paths and fetch status
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        successful = 0
        failed = 0
        
        print(f"Fetching {len(df)} satellite images using {self.provider}...")
        print(f"Output directory: {output_dir}")
        print(f"Image size: {self.image_size}x{self.image_size}, Zoom: {self.zoom}")
        print()
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Fetching images"):
            prop_id = row[id_col]
            lat = row[lat_col]
            lon = row[lon_col]
            
            # Image filename
            image_name = f"{prop_id}.png"
            image_path = output_dir / image_name
            
            # Fetch image
            success = self.fetch_image(lat, lon, image_path)
            
            if success:
                successful += 1
            else:
                failed += 1
            
            results.append({
                'id': prop_id,
                'lat': lat,
                'long': lon,
                'image_path': str(image_path) if success else None,
                'fetch_success': success
            })
            
            # Rate limiting
            time.sleep(0.1)
        
        print(f"\nFetch complete: {successful} successful, {failed} failed")
        
        return pd.DataFrame(results)


def generate_placeholder_images(df, output_dir, id_col='id', lat_col='lat', lon_col='long', size=256):
    """
    Generate placeholder images for development without API keys.
    Creates colored squares based on lat/long for visual differentiation.
    
    Args:
        df: DataFrame with property data
        output_dir: Directory to save images
        id_col: Column for unique IDs
        lat_col: Column for latitude
        lon_col: Column for longitude
        size: Image size
    
    Returns:
        DataFrame with image paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    print(f"Generating {len(df)} placeholder images...")
    print(f"Output directory: {output_dir}")
    print()
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating placeholders"):
        prop_id = row[id_col]
        lat = row[lat_col]
        lon = row[lon_col]
        
        # Generate a unique color based on coordinates
        lat_norm = (lat - df[lat_col].min()) / (df[lat_col].max() - df[lat_col].min())
        lon_norm = (lon - df[lon_col].min()) / (df[lon_col].max() - df[lon_col].min())
        
        # Create gradient-based image
        img_array = np.zeros((size, size, 3), dtype=np.uint8)
        
        # Create a texture pattern
        for i in range(size):
            for j in range(size):
                r = int(50 + lat_norm * 100 + (i / size) * 50)
                g = int(80 + lon_norm * 80 + (j / size) * 40)
                b = int(30 + (i + j) / (2 * size) * 60)
                img_array[i, j] = [r, g, b]
        
        # Add some noise for texture
        noise = np.random.randint(-20, 20, (size, size, 3))
        img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Save image
        img = Image.fromarray(img_array)
        image_path = output_dir / f"{prop_id}.png"
        img.save(image_path)
        
        results.append({
            'id': prop_id,
            'lat': lat,
            'long': lon,
            'image_path': str(image_path),
            'fetch_success': True,
            'is_placeholder': True
        })
    
    print(f"\nGenerated {len(results)} placeholder images")
    
    return pd.DataFrame(results)


def fetch_all_images(use_placeholders=False):
    """
    Main function to fetch satellite images for train and test data.
    
    Args:
        use_placeholders: If True, generate placeholder images instead of fetching
    """
    # Load processed data
    train_df = pd.read_csv(DATA_PROCESSED / "train.csv")
    test_df = pd.read_csv(DATA_PROCESSED / "test.csv")
    
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Check for API keys
    mapbox_token = os.getenv('MAPBOX_ACCESS_TOKEN')
    google_key = os.getenv('GOOGLE_MAPS_API_KEY')
    
    if not use_placeholders and not mapbox_token and not google_key:
        print("\n" + "="*60)
        print("WARNING: No API keys found!")
        print("Set MAPBOX_ACCESS_TOKEN or GOOGLE_MAPS_API_KEY in .env file")
        print("Generating placeholder images instead...")
        print("="*60 + "\n")
        use_placeholders = True
    
    if use_placeholders:
        # Generate placeholders
        train_results = generate_placeholder_images(
            train_df, 
            DATA_IMAGES / "train",
            id_col='id'
        )
        test_results = generate_placeholder_images(
            test_df,
            DATA_IMAGES / "test", 
            id_col='id'
        )
    else:
        # Fetch real satellite images
        provider = 'mapbox' if mapbox_token else 'google'
        fetcher = SatelliteImageFetcher(provider=provider)
        
        print("\nFetching training images...")
        train_results = fetcher.fetch_dataset(
            train_df,
            DATA_IMAGES / "train",
            id_col='id'
        )
        
        print("\nFetching test images...")
        test_results = fetcher.fetch_dataset(
            test_df,
            DATA_IMAGES / "test",
            id_col='id'
        )
    
    # Save image mapping files
    train_results.to_csv(DATA_IMAGES / "train_image_mapping.csv", index=False)
    test_results.to_csv(DATA_IMAGES / "test_image_mapping.csv", index=False)
    
    print(f"\nImage mapping saved to:")
    print(f"  - {DATA_IMAGES / 'train_image_mapping.csv'}")
    print(f"  - {DATA_IMAGES / 'test_image_mapping.csv'}")
    
    return train_results, test_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch satellite images for properties')
    parser.add_argument('--placeholders', action='store_true', 
                        help='Generate placeholder images instead of fetching')
    args = parser.parse_args()
    
    fetch_all_images(use_placeholders=args.placeholders)
