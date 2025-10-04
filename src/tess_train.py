import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import json
import os
import logging
import time

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_tess_model(data_path, model_name, output_dir=None, param_grid=None):
    """
    Huấn luyện mô hình Random Forest trên dữ liệu TESS.
    - Input: 
        - data_path: Đường dẫn file CSV đã xử lý.
        - model_name: Tên mô hình để lưu (ví dụ: 'tess_model1').
        - output_dir: Thư mục lưu file .pkl (mặc định: models/).
        - param_grid: Tham số GridSearchCV (mặc định: cố định).
    - Output: (model_path, stats)
        - model_path: Đường dẫn file .pkl.
        - stats: dict chứa train/test accuracy, precision, recall, F1, feature importance.
    """
    start_time = time.time()
    logger.info("Starting TESS model training...")

    # Thiết lập đường dẫn
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    
    if not os.path.exists(data_path):
        logger.error(f"File not found: {data_path}")
        raise FileNotFoundError(f"Dataset file {data_path} does not exist.")

    # Load dữ liệu
    df = pd.read_csv(data_path)
    logger.info(f"Loaded data: {df.shape}")

    # Kiểm tra cột bắt buộc
    if 'tfopwg_disp' not in df.columns:
        logger.error("Target column 'tfopwg_disp' not found.")
        raise ValueError("Target column 'tfopwg_disp' not found in dataset.")

    # Tách features và target
    X = df.drop(['tfopwg_disp', 'toi'], axis=1, errors='ignore')
    y = df['tfopwg_disp']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Thiết lập param_grid tối ưu cho production
    if param_grid is None or os.getenv('ENV') == 'production':
        param_grid = {
            'n_estimators': [100],  # Giảm để tối ưu thời gian
            'max_depth': [10],
            'min_samples_split': [5],
            'max_features': ['sqrt']
        }
    else:
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [8, 10],
            'min_samples_split': [5, 10],
            'max_features': ['sqrt']
        }

    # Thiết lập n_jobs cho production
    n_jobs = 1 if os.getenv('ENV') == 'production' else -1

    # Huấn luyện mô hình với GridSearchCV
    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3 if os.getenv('ENV') == 'production' else 5,  # Giảm folds trong production
        n_jobs=n_jobs,
        scoring='accuracy'
    )
    grid_search.fit(X_train, y_train)
    logger.info(f"GridSearchCV completed in {time.time() - start_time:.2f} seconds")

    # Lấy mô hình tốt nhất
    best_model = grid_search.best_estimator_
    logger.info(f"Best parameters: {grid_search.best_params_}")

    # Đánh giá mô hình
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    
    stats = {
        'train_accuracy': float(accuracy_score(y_train, y_train_pred)),
        'test_accuracy': float(accuracy_score(y_test, y_test_pred)),
        'train_precision': float(precision_score(y_train, y_train_pred, average='weighted')),
        'test_precision': float(precision_score(y_test, y_test_pred, average='weighted')),
        'train_recall': float(recall_score(y_train, y_train_pred, average='weighted')),
        'test_recall': float(recall_score(y_test, y_test_pred, average='weighted')),
        'train_f1': float(f1_score(y_train, y_train_pred, average='weighted')),
        'test_f1': float(f1_score(y_test, y_test_pred, average='weighted')),
        'feature_importance': {
            col: float(imp) for col, imp in zip(X.columns, best_model.feature_importances_)
        },
        'training_time_seconds': float(time.time() - start_time)
    }
    logger.info(f"Training stats: {stats}")

    # Lưu mô hình
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f'{model_name}_rf_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    logger.info(f"Saved model: {model_path}")

    # Lưu features info vào JSON
    features_info = {
        'num_features': len(X.columns),
        'features': X.columns.tolist(),
        'importance': dict(stats['feature_importance'])
    }
    features_path = os.path.join(output_dir, f'{model_name}_features.json')
    with open(features_path, 'w') as f:
        json.dump(features_info, f, indent=2)
    logger.info(f"Saved features info: {features_path} (num: {features_info['num_features']})")

    logger.info(f"Total training time: {time.time() - start_time:.2f} seconds")
    return model_path, stats