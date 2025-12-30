"""
FastAPI Server for Property Valuation

REST API for predicting property prices using the trained multimodal model.
"""

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Union
import numpy as np
import torch
from pathlib import Path
import joblib
import io
from PIL import Image
from torchvision import transforms
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# Initialize FastAPI app
app = FastAPI(
    title="Property Valuation API",
    description="Predict property prices using satellite imagery and property features",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request, call_next):
    print(f"Request: {request.method} {request.url}")
    try:
        response = await call_next(request)
        print(f"Response: {response.status_code}")
        return response
    except Exception as e:
        print(f"Request FAILED: {str(e)}")
        raise e

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_IMAGES = PROJECT_ROOT / "data" / "images"
EXPLANATIONS_DIR = PROJECT_ROOT / "reports" / "api_explanations"

# Create explanations directory if it doesn't exist
EXPLANATIONS_DIR.mkdir(parents=True, exist_ok=True)

# Global model and scaler (loaded on startup)
model = None
scaler = None
feature_columns = None
device = "cpu"


# Request/Response models
class PropertyFeatures(BaseModel):
    """Property features for prediction."""
    bedrooms: Optional[float] = Field(None, description="Number of bedrooms")
    bathrooms: Optional[float] = Field(None, description="Number of bathrooms")
    sqft_living: Optional[float] = Field(None, description="Living area in sqft")
    sqft_lot: Optional[float] = Field(None, description="Lot size in sqft")
    floors: Optional[float] = Field(None, description="Number of floors")
    waterfront: Optional[float] = Field(0, description="Waterfront property (0/1)")
    view: Optional[float] = Field(0, description="View rating (0-4)")
    condition: Optional[float] = Field(3, description="Condition rating (1-5)")
    grade: Optional[float] = Field(7, description="Grade (1-13)")
    sqft_above: Optional[float] = Field(None, description="Above ground sqft")
    sqft_basement: Optional[float] = Field(0, description="Basement sqft")
    yr_built: Optional[float] = Field(None, description="Year built")
    yr_renovated: Optional[float] = Field(0, description="Year renovated")
    zipcode: Optional[float] = Field(None, description="Zipcode")
    lat: Optional[float] = Field(None, description="Latitude")
    long: Optional[float] = Field(None, description="Longitude")
    sqft_living15: Optional[float] = Field(None, description="Avg sqft of 15 nearest neighbors")
    sqft_lot15: Optional[float] = Field(None, description="Avg lot size of 15 nearest neighbors")
    
    class Config:
        json_schema_extra = {
            "example": {
                "bedrooms": 3,
                "bathrooms": 2,
                "sqft_living": 1800,
                "sqft_lot": 5000,
                "floors": 1,
                "waterfront": 0,
                "view": 0,
                "condition": 3,
                "grade": 7,
                "sqft_above": 1800,
                "sqft_basement": 0,
                "yr_built": 1990,
                "yr_renovated": 0,
                "zipcode": 98103,
                "lat": 47.5112,
                "long": -122.257,
                "sqft_living15": 1800,
                "sqft_lot15": 5000
            }
        }


class PredictionResponse(BaseModel):
    """Prediction response."""
    predicted_price: float = Field(..., description="Predicted property price")
    confidence: str = Field(..., description="Confidence level or message")
    explanation_url: Optional[str] = Field(None, description="URL to the Grad-CAM heatmap image")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    device: str


@app.on_event("startup")
async def load_model():
    """Load model and scaler on startup."""
    global model, scaler, feature_columns, device
    
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load scaler and feature columns
    scaler_path = MODELS_DIR / "feature_scaler.joblib"
    feature_cols_path = MODELS_DIR / "feature_columns.joblib"
    model_path = MODELS_DIR / "fusion_model.pth"
    
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        print(f"Loaded scaler from: {scaler_path}")
    else:
        print(f"WARNING: Scaler not found at {scaler_path}")
    
    if feature_cols_path.exists():
        feature_columns = joblib.load(feature_cols_path)
        print(f"Loaded feature columns: {feature_columns}")
    else:
        print(f"WARNING: Feature columns not found at {feature_cols_path}")
    
    # Load model
    if model_path.exists():
        try:
            from src.models import FusionModel
            
            model = FusionModel(
                tabular_input_dim=len(feature_columns) if feature_columns else 16,
                pretrained=False
            )
            
            checkpoint = torch.load(model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model = model.to(device)
            model.eval()
            
            print(f"Loaded model from: {model_path}")
        except Exception as e:
            print(f"ERROR loading model: {e}")
            model = None
    else:
        print(f"WARNING: Model not found at {model_path}")





@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        device=device
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_price(features: PropertyFeatures):
    """
    Predict property price from features.
    
    Note: This endpoint uses a placeholder image. For full prediction with
    satellite imagery, use /predict-with-image endpoint.
    """
    global model, scaler, feature_columns
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if scaler is None or feature_columns is None:
        raise HTTPException(status_code=503, detail="Scaler or feature columns not loaded")
    
    try:
        # Convert features to dict
        features_dict = features.model_dump()
        
        # Extract features in correct order
        feature_values = np.array([[features_dict.get(col, 0) for col in feature_columns]], 
                                   dtype=np.float32)
        
        # Handle any None values
        feature_values = np.nan_to_num(feature_values, nan=0.0)
        
        # Scale features
        feature_scaled = scaler.transform(feature_values)
        tab_tensor = torch.from_numpy(feature_scaled).to(device)
        
        # Create placeholder image (since we don't have actual satellite image)
        img_tensor = torch.zeros(1, 3, 224, 224).to(device)
        
        # Predict
        with torch.no_grad():
            prediction = model(img_tensor, tab_tensor)
        
        predicted_price = float(prediction.item())
        
        # Ensure non-negative prediction
        predicted_price = max(0, predicted_price)
        
        return PredictionResponse(
            predicted_price=predicted_price,
            confidence="low (no satellite image provided)"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-explain", response_model=PredictionResponse)
async def predict_with_explanation(
    features: Any = Form(...),
    image: UploadFile = File(...)
):
    """
    Predict property price and generate Grad-CAM explanation.
    Returns prediction and path to saved heatmap.
    """
    global model, scaler, feature_columns
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        import json
        if isinstance(features, str):
            features_dict = json.loads(features)
        else:
            features_dict = features
        features_obj = PropertyFeatures(**features_dict)
        
        # Process image
        image_data = await image.read()
        img = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        from src.explainability import visualize_gradcam
        import os
        
        # Save original image temporarily or use the buffer
        temp_img_path = Path("temp_predict_img.png")
        img.save(temp_img_path)
        
        # Process features
        features_dict = features_obj.model_dump()
        feature_values = np.array([[features_dict.get(col, 0) for col in feature_columns]], 
                                   dtype=np.float32)
        feature_values = np.nan_to_num(feature_values, nan=0.0)
        feature_scaled = scaler.transform(feature_values)
        tab_tensor = torch.from_numpy(feature_scaled).to(device)
        
        # Paths for output
        output_dir = PROJECT_ROOT / "reports" / "api_explanations"
        output_dir.mkdir(parents=True, exist_ok=True)
        viz_path = output_dir / f"explanation_{os.getpid()}.png"
        
        # Generate Grad-CAM
        cam, overlay = visualize_gradcam(
            model=model,
            image_path=str(temp_img_path),
            tabular_features=tab_tensor,
            save_path=str(viz_path),
            device=device
        )
        
        # Predict
        with torch.no_grad():
            img_tensor = torch.from_numpy(np.array(img.resize((224, 224)))).permute(2, 0, 1).float().unsqueeze(0).to(device)
            # Normalize img_tensor matches training
            img_tensor = img_tensor / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            img_tensor = (img_tensor - mean) / std
            
            prediction = model(img_tensor, tab_tensor)
        
        predicted_price = max(0, float(prediction.item()))
        
        # Clean up
        if temp_img_path.exists():
            os.remove(temp_img_path)
        
        return PredictionResponse(
            predicted_price=predicted_price,
            confidence="explained: satellite imagery integrated",
            explanation_url=f"/explanations/{viz_path.name}"
        )
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
async def model_info():
    """Get information about the loaded model."""
    if model is None:
        return {"error": "Model not loaded"}
    
    return {
        "model_type": "FusionModel",
        "num_features": len(feature_columns) if feature_columns else None,
        "feature_columns": feature_columns,
        "device": device,
        "parameters": sum(p.numel() for p in model.parameters()) if model else None
    }


# Mount static files (must be at the end to avoid catching API routes)
# Create explanations directory if it doesn't exist
EXPLANATIONS_DIR.mkdir(parents=True, exist_ok=True)

# Mount static files for explanations
app.mount("/explanations", StaticFiles(directory=str(EXPLANATIONS_DIR)), name="explanations")

# Serve frontend at root
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
