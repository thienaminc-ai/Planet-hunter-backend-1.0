from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
import logging
import json
import time
from kepler_data_processing import preprocess_kepler_data, transform_single_row as transform_kepler
from tess_data_preprocessing import preprocess_tess_data, transform_single_row as transform_tess
from kepler_train import train_kepler_model
from tess_train import train_tess_model
from kepler_test import predict_kepler_model
from tess_test import predict_tess_model

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Ghi log ra console
        logging.FileHandler('app.log')  # Lưu log vào file để debug trên Render
    ]
)
logger = logging.getLogger(__name__)

# Load .env file
load_dotenv()

app = Flask(__name__)

# Lấy FRONTEND_URL từ env
frontend_url = os.getenv('FRONTEND_URL', 'http://localhost:3000')

# Cấu hình CORS
CORS(app, origins=[
    "https://planet-hunter-frontend-1-0.vercel.app",
    "http://localhost:3000",
    frontend_url  # Động để hỗ trợ các môi trường khác
], supports_credentials=True)

# Route health check cho Render.com
@app.route('/', methods=['GET'])
def health_check():
    logger.info("Health check accessed")
    return jsonify({
        'status': 'healthy',
        'message': 'Planet Hunter Backend API is running',
        'version': '1.0',
        'supported_datasets': ['kepler', 'tess'],
        'environment': os.getenv('ENV', 'development')
    }), 200

# Hàm chuyển đổi NumPy types thành Python types
def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

# Route /list_datasets: Liệt kê các file CSV biến thể (support both Kepler & TESS)
@app.route('/list_datasets', methods=['GET'])
def list_datasets():
    try:
        dataset = request.args.get('dataset', 'kepler')  # Default Kepler, or 'tess'
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', dataset)
        logger.info(f"Listing datasets in: {data_dir} for {dataset}")
        
        if not os.path.exists(data_dir):
            logger.warning(f"Directory not found: {data_dir}. Returning empty dataset list.")
            return jsonify({
                'status': 'success',
                'datasets': [],
                'message': 'No processed datasets found.',
                'dataset': dataset
            })
        
        datasets = [
            f.replace('_processed.csv', '') 
            for f in os.listdir(data_dir) 
            if f.endswith('_processed.csv') and os.path.isfile(os.path.join(data_dir, f))
        ]
        
        logger.info(f"Found {len(datasets)} datasets: {datasets}")
        return jsonify({
            'status': 'success',
            'datasets': datasets,
            'message': f'Found {len(datasets)} processed datasets.',
            'dataset': dataset
        })
    
    except Exception as e:
        logger.error(f"List datasets error: {str(e)}")
        return jsonify({'status': 'error', 'message': f'Failed to list datasets: {str(e)}'}), 500

# Route /analyze: Phân tích shape và columns từ CSV (support Kepler & TESS)
@app.route('/analyze', methods=['POST'])
def analyze_columns():
    start_time = time.time()
    try:
        data = request.get_json(force=True)
        dataset = data.get('dataset', 'kepler')  # 'kepler' or 'tess'
        
        if dataset == 'kepler':
            file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'kepler', 'kepler data.csv')
        elif dataset == 'tess':
            file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'tess', 'tess data.csv')
        else:
            return jsonify({'status': 'error', 'message': 'Invalid dataset: kepler or tess'}), 400
            
        logger.info(f"Loading file: {file_path} for {dataset}")
        df = pd.read_csv(file_path, comment='#')
        
        shape = df.shape
        columns = df.columns.tolist()
        
        result = {
            'status': 'success',
            'shape': {'rows': int(shape[0]), 'cols': int(shape[1])},
            'columns': columns,
            'message': f'Analysis successful: {shape[0]} rows, {shape[1]} columns.',
            'frontend_redirect': f'{frontend_url}/preprocess?dataset={dataset}',
            'dataset': dataset,
            'execution_time_seconds': float(time.time() - start_time)
        }
        
        logger.info(f"Analyze completed in {time.time() - start_time:.2f} seconds: shape {shape}, {len(columns)} columns for {dataset}")
        return jsonify(result)
    
    except FileNotFoundError as fe:
        logger.error(f"FileNotFound: {fe}")
        return jsonify({'status': 'error', 'message': f'File "{dataset} data.csv" not found at {file_path}.'}), 404
    except Exception as e:
        logger.error(f"Analyze error: {str(e)}")
        return jsonify({'status': 'error', 'message': f'Analysis failed: {str(e)}'}), 500

# Route /create_variant: Tạo variant dữ liệu từ columns và name (support Kepler & TESS)
@app.route('/create_variant', methods=['POST'])
def create_variant():
    start_time = time.time()
    try:
        data = request.get_json(force=True)
        logger.info(f"Received create_variant payload: {data}")
        
        columns = data.get('columns', [])
        name = data.get('name', 'default_variant').strip()
        remove_outliers = data.get('remove_outliers', False)
        dataset = data.get('dataset', 'kepler')  # 'kepler' or 'tess'
        
        logger.info(f"Calling preprocess_{dataset}_data with: name={name}, columns={columns}, remove_outliers={remove_outliers}")
        
        if not name:
            logger.error("Dataset name is required.")
            return jsonify({'status': 'error', 'message': 'Dataset name is required.'}), 400
        
        if dataset == 'kepler':
            csv_path, imputer_path, scaler_path, stats = preprocess_kepler_data(columns, name, remove_outliers)
        elif dataset == 'tess':
            csv_path, imputer_path, scaler_path, stats = preprocess_tess_data(columns, name, remove_outliers)
        else:
            return jsonify({'status': 'error', 'message': 'Invalid dataset: kepler or tess'}), 400
            
        stats = convert_numpy_types(stats)
        logger.info(f"Preprocessing completed in {time.time() - start_time:.2f} seconds: stats={stats}")
        
        message = f'Created {dataset.upper()} variant successfully!'
        if stats['flag_noise_dropped'] > 0:
            message += f' (Dropped {stats["flag_noise_dropped"]} noisy flag rows)'
        if stats['outliers_dropped'] > 0:
            message += f' (Dropped {stats["outliers_dropped"]} outlier rows)'
        if stats['total_noise_removed_pct'] > 0:
            message += f' (Total noise removed: {stats["total_noise_removed_pct"]}%)'
        
        files = {'csv': csv_path, 'scaler': scaler_path}
        if imputer_path:
            files['imputer'] = imputer_path
        
        return jsonify({
            'status': 'success',
            'message': message,
            'name': name,
            'files': files,
            'stats': stats,
            'dataset': dataset,
            'execution_time_seconds': float(time.time() - start_time)
        })
    
    except ValueError as ve:
        logger.error(f"ValueError in create_variant: {ve}")
        return jsonify({'status': 'error', 'message': str(ve)}), 400
    except Exception as e:
        logger.error(f"Create_variant error: {str(e)}")
        return jsonify({'status': 'error', 'message': f'Processing failed: {str(e)}'}), 500

# Route /train: Huấn luyện mô hình trên file CSV đã xử lý (support Kepler & TESS)
@app.route('/train', methods=['POST'])
def train():
    start_time = time.time()
    try:
        data = request.get_json(force=True)
        logger.info(f"Received train payload: {data}")
        
        dataset_name = data.get('dataset_name', '').strip()
        model_name = data.get('model_name', f'{data.get("dataset", "kepler")}_{dataset_name}_model').strip()
        param_grid = data.get('param_grid')  # Optional param_grid
        dataset = data.get('dataset', 'kepler')  # 'kepler' or 'tess'
        
        if not dataset_name:
            logger.error("Dataset name is required.")
            return jsonify({'status': 'error', 'message': 'Dataset name is required.'}), 400
        if not model_name:
            logger.error("Model name is required.")
            return jsonify({'status': 'error', 'message': 'Model name is required.'}), 400
        
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', dataset, f'{dataset_name}_processed.csv')
        logger.info(f"Training {dataset} model with data: {data_path}, model_name: {model_name}, param_grid: {param_grid}")
        
        # Trong production, khuyến khích dùng pre-trained models
        if os.getenv('ENV') == 'production':
            model_path = os.path.join(os.path.dirname(__file__), '..', 'models', f'{model_name}_rf_model.pkl')
            if os.path.exists(model_path):
                logger.info(f"Pre-trained model found: {model_path}. Skipping training.")
                with open(model_path, 'rb') as f:
                    stats = {
                        'train_accuracy': 0.0,
                        'test_accuracy': 0.0,
                        'train_precision': 0.0,
                        'test_precision': 0.0,
                        'train_recall': 0.0,
                        'test_recall': 0.0,
                        'train_f1': 0.0,
                        'test_f1': 0.0,
                        'feature_importance': {},
                        'training_time_seconds': 0.0
                    }
                    features_path = os.path.join(os.path.dirname(__file__), '..', 'models', f'{model_name}_features.json')
                    if os.path.exists(features_path):
                        with open(features_path, 'r') as f:
                            features_info = json.load(f)
                            stats['feature_importance'] = features_info.get('importance', {})
                    return jsonify({
                        'status': 'success',
                        'message': f'Using pre-trained {dataset.upper()} model: {model_name}',
                        'model_name': model_name,
                        'model_path': model_path,
                        'stats': stats,
                        'best_params': 'Pre-trained',
                        'dataset': dataset,
                        'execution_time_seconds': float(time.time() - start_time)
                    })
        
        if dataset == 'kepler':
            model_path, stats = train_kepler_model(data_path, model_name, param_grid=param_grid)
        elif dataset == 'tess':
            model_path, stats = train_tess_model(data_path, model_name, param_grid=param_grid)
        else:
            return jsonify({'status': 'error', 'message': 'Invalid dataset: kepler or tess'}), 400
            
        stats = convert_numpy_types(stats)
        
        logger.info(f"Training completed in {time.time() - start_time:.2f} seconds: stats={stats}")
        
        display_model_name = os.path.basename(model_path).replace('.pkl', '')
        
        message = f'Trained {dataset.upper()} model successfully! Test accuracy: {stats["test_accuracy"]:.2f}'
        
        return jsonify({
            'status': 'success',
            'message': message,
            'model_name': display_model_name,
            'model_path': model_path,
            'stats': stats,
            'best_params': stats.get('best_params', 'Default'),
            'dataset': dataset,
            'execution_time_seconds': float(time.time() - start_time)
        })
    
    except FileNotFoundError as fe:
        logger.error(f"FileNotFound: {fe}")
        return jsonify({'status': 'error', 'message': str(fe)}), 404
    except ValueError as ve:
        logger.error(f"ValueError in train: {ve}")
        return jsonify({'status': 'error', 'message': str(ve)}), 400
    except Exception as e:
        logger.error(f"Train error: {str(e)}")
        return jsonify({'status': 'error', 'message': f'Training failed: {str(e)}'}), 500

# Route /list_models: Liệt kê các models trong models/ (support both)
@app.route('/list_models', methods=['GET'])
def list_models_route():
    start_time = time.time()
    try:
        dataset = request.args.get('dataset', 'kepler')
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        models = [f for f in os.listdir(models_dir) if f.endswith('.pkl') and dataset in f]
        models = [m.replace('.pkl', '') for m in models]
        logger.info(f"Listed models for {dataset}: {models}")
        return jsonify({
            'status': 'success',
            'models': models,
            'message': f'Found {len(models)} trained models for {dataset}.',
            'dataset': dataset,
            'execution_time_seconds': float(time.time() - start_time)
        })
    except Exception as e:
        logger.error(f"List models error: {str(e)}")
        return jsonify({'status': 'error', 'message': f'Failed to list models: {str(e)}'}), 500

# Route /predict: Predict trên model với input data (support Kepler & TESS)
@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    try:
        data = request.get_json(force=True)
        logger.info(f"Received predict payload: {data}")
        
        model_name = data.get('model_name', '').strip()
        input_data = data.get('input_data', {})
        dataset = data.get('dataset', 'kepler')
        
        if not model_name:
            logger.error("Model name is required.")
            return jsonify({'status': 'error', 'message': 'Model name is required.'}), 400
        
        if not input_data:
            logger.error("Input data is required.")
            return jsonify({'status': 'error', 'message': 'Input data is required.'}), 400
        
        base_model_name = model_name.replace('_rf_model', '')
        dataset_name = base_model_name.replace(f'{dataset}_', '').replace('_model', '')
        
        if dataset == 'kepler':
            df_processed = transform_kepler(dataset_name, input_data)
            result = predict_kepler_model(model_name, df_processed)
        elif dataset == 'tess':
            df_processed = transform_tess(dataset_name, input_data)
            result = predict_tess_model(model_name, df_processed)
        else:
            return jsonify({'status': 'error', 'message': 'Invalid dataset: kepler or tess'}), 400
            
        result = convert_numpy_types(result)
        
        logger.info(f"Prediction completed in {time.time() - start_time:.2f} seconds: {result}")
        
        message = f'Prediction: {result["prediction"]} (confidence: {result["confidence"]:.2f})'
        
        return jsonify({
            'status': 'success',
            'message': message,
            'model_name': model_name,
            'result': result,
            'dataset': dataset,
            'execution_time_seconds': float(time.time() - start_time)
        })
    
    except FileNotFoundError as fe:
        logger.error(f"Model not found: {fe}")
        return jsonify({'status': 'error', 'message': str(fe)}), 404
    except ValueError as ve:
        logger.error(f"ValueError in predict: {ve}")
        return jsonify({'status': 'error', 'message': str(ve)}), 400
    except Exception as e:
        logger.error(f"Predict error: {str(e)}")
        return jsonify({'status': 'error', 'message': f'Prediction failed: {str(e)}'}), 500

# Route /model_features: Lấy info features của model (support both)
@app.route('/model_features', methods=['GET'])
def model_features():
    start_time = time.time()
    try:
        model_name = request.args.get('model_name', '').strip()
        dataset = request.args.get('dataset', 'kepler')
        if not model_name:
            return jsonify({'status': 'error', 'message': 'Model name is required.'}), 400
        
        base_model_name = model_name.replace('_rf_model', '')
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        features_path = os.path.join(models_dir, f'{base_model_name}_features.json')
        
        if not os.path.exists(features_path):
            logger.error(f"Features file not found: {features_path}")
            return jsonify({'status': 'error', 'message': f'Features info for "{model_name}" not found. Retrain model?'}), 404
        
        with open(features_path, 'r') as f:
            features_info = json.load(f)
        
        logger.info(f"Returned features for {model_name}: {features_info['num_features']} fields in {time.time() - start_time:.2f} seconds")
        return jsonify({
            'status': 'success',
            'model_name': model_name,
            'num_features': features_info['num_features'],
            'features': features_info['features'],
            'importance': features_info['importance'],
            'dataset': dataset,
            'execution_time_seconds': float(time.time() - start_time)
        })
    
    except Exception as e:
        logger.error(f"Model features error: {str(e)}")
        return jsonify({'status': 'error', 'message': f'Failed to get features: {str(e)}'}), 500

# Route /status: Kiểm tra trạng thái backend
@app.route('/status', methods=['GET'])
def status():
    start_time = time.time()
    try:
        logger.info("Status check")
        return jsonify({
            'status': 'Backend running',
            'version': '1.0',
            'frontend_url': frontend_url,
            'supported_datasets': ['kepler', 'tess'],
            'environment': os.getenv('ENV', 'development'),
            'execution_time_seconds': float(time.time() - start_time)
        })
    except Exception as e:
        logger.error(f"Status error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Global error handler
@app.errorhandler(Exception)
def handle_global_error(error):
    logger.error(f"Global error: {str(error)}, path: {request.path}")
    return jsonify({
        'status': 'error',
        'message': 'Internal server error. Please try again.',
        'path': request.path
    }), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5001))
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)