import pandas as pd
import numpy as np
import pickle
import os
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def list_models():
    """
    Liệt kê các models trong thư mục models/ (tên file *_rf_model.pkl).
    """
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    if not os.path.exists(models_dir):
        logger.warning(f"Models directory not found: {models_dir}")
        return []
    
    models = [
        f.replace('_rf_model.pkl', '') 
        for f in os.listdir(models_dir) 
        if f.endswith('_rf_model.pkl')
    ]
    logger.info(f"Found {len(models)} models: {models}")
    return models

def predict_kepler_model(model_name, df_processed):
    """
    Dự đoán sử dụng model RandomForest (nhận df đã processed).
    - Load model.
    - Reorder theo required_features (fill missing=0 nếu cần).
    - Predict.
    """
    # Đường dẫn
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    model_path = os.path.join(models_dir, f'{model_name}_rf_model.pkl')
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        raise FileNotFoundError(f"Model '{model_name}' not found.")
    
    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    logger.info(f"Loaded model: {model_path}")
    
    # Required features từ model
    required_features = list(model.feature_names_in_)
    logger.info(f"Required features: {len(required_features)}")
    
    # Fill missing cols bằng 0 và reorder
    for col in required_features:
        if col not in df_processed.columns:
            df_processed[col] = 0.0
            logger.warning(f"Filled missing feature '{col}' with 0")
    df_processed = df_processed[required_features]
    
    # Predict
    prediction = model.predict(df_processed)[0]
    proba = model.predict_proba(df_processed)[0]
    probabilities = dict(zip(model.classes_, proba))
    confidence = float(max(proba))
    
    logger.info(f"Prediction: {prediction} (confidence: {confidence:.2f})")
    
    return {
        'prediction': str(prediction),
        'probabilities': probabilities,
        'confidence': confidence
    }