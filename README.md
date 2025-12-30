# Satellite Property Valuation

A multimodal deep learning system that combines tabular property data with satellite imagery to predict property prices.

## 🌟 Features

- **Multimodal Fusion**: Combines tabular features with CNN-extracted image features
- **Explainability**: Grad-CAM visualizations to understand model predictions
- **REST API**: FastAPI-based inference endpoint
- **Reproducible**: Complete training and evaluation pipeline

## 📁 Project Structure

```
satellite-property-valuation/
├── data/
│   ├── raw/                    # Original Excel files
│   ├── processed/              # Cleaned CSV files
│   └── images/                 # Satellite imagery
├── src/
│   ├── preprocessing.py        # Data cleaning & feature engineering
│   ├── data_fetcher.py         # Satellite image downloader
│   ├── dataset.py              # PyTorch Dataset class
│   ├── models.py               # Model architectures
│   ├── train.py                # Training script
│   └── explainability.py       # Grad-CAM implementation
├── api/
│   └── main.py                 # FastAPI server
├── notebooks/                  # Jupyter notebooks for EDA/training
├── models/                     # Saved model checkpoints
├── reports/                    # Visualizations and reports
└── submission/                 # Prediction outputs
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/satellite-property-valuation.git
cd satellite-property-valuation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. For GPU training (Google Colab):
```bash
pip install -r requirements-gpu.txt
```

### Configuration

Create a `.env` file in the project root:
```env
MAPBOX_ACCESS_TOKEN=your_mapbox_token_here
# Or for Google Maps:
# GOOGLE_MAPS_API_KEY=your_google_maps_key_here
```

## 📊 Data Pipeline

### 1. Preprocess Tabular Data
```bash
python -m src.preprocessing
```

### 2. Fetch Satellite Images
```bash
python -m src.data_fetcher
```

## 🧠 Model Training

Training requires GPU and is designed to run in Google Colab:

1. Upload the `notebooks/02_training.ipynb` to Google Colab
2. Enable GPU runtime
3. Follow the notebook instructions

## 🔮 Inference API

Start the FastAPI server:
```bash
uvicorn api.main:app --reload
```

### Endpoints

- `GET /health`: Checks server status and model loading.
- `POST /predict`: Predicts price using tabular features.
- `POST /predict-explain`: Predicts price and generates a Grad-CAM heatmap (requires image upload).

### Example Requests

#### Tabular Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
    "sqft_living15": 1800,
    "sqft_lot15": 5000
  }'
```

#### Multimodal Prediction with Explanation
```bash
curl -X POST "http://localhost:8000/predict-explain" \
  -F "image=@data/images/train/1000102.png" \
  -F 'features={"bedrooms": 3, "bathrooms": 2, "sqft_living": 1800, "sqft_lot": 5000, "floors": 1, "waterfront": 0, "view": 0, "condition": 3, "grade": 7, "sqft_above": 1800, "sqft_basement": 0, "yr_built": 1990, "yr_renovated": 0, "zipcode": 98103, "sqft_living15": 1800, "sqft_lot15": 5000}'
```

## 📈 Results

| Model | RMSE | R² |
|-------|------|-----|
| Tabular Only (Baseline) | TBD | TBD |
| Multimodal Fusion | TBD | TBD |

## 🔍 Explainability

Grad-CAM visualizations show which regions of satellite images influence predictions:

- Roads and infrastructure
- Green spaces
- Water proximity
- Building density

## 📝 License

MIT License

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.
