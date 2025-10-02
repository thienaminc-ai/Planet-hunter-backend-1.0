from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
import logging
import json  # Added missing import
from kepler_data_processing import preprocess_kepler_data, transform_single_row
from kepler_train import train_kepler_model
from kepler_test import predict_kepler_model, list_models
# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load .env file
load_dotenv()

app = Flask(__name__)

# Lấy FRONTEND_URL từ env
frontend_url = os.getenv('FRONTEND_URL', 'http://localhost:3000')

# Cấu hình CORS
CORS(app, origins=[frontend_url], supports_credentials=True)

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

# Route /list_datasets: Liệt kê các file CSV biến thể trong data/kepler
@app.route('/list_datasets', methods=['GET'])
def list_datasets():
    try:
        # Sửa đường dẫn: từ src/data/kepler thành data/kepler
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'kepler')
        logger.info(f"Listing datasets in: {data_dir}")
        
        # Kiểm tra thư mục tồn tại
        if not os.path.exists(data_dir):
            logger.warning(f"Directory not found: {data_dir}. Returning empty dataset list.")
            return jsonify({
                'status': 'success',
                'datasets': [],
                'message': 'No processed datasets found.'
            })
        
        # Chỉ lấy file CSV có hậu tố _processed.csv
        datasets = [
            f.replace('_processed.csv', '') 
            for f in os.listdir(data_dir) 
            if f.endswith('_processed.csv') and os.path.isfile(os.path.join(data_dir, f))
        ]
        
        logger.info(f"Found {len(datasets)} datasets: {datasets}")
        return jsonify({
            'status': 'success',
            'datasets': datasets,
            'message': f'Found {len(datasets)} processed datasets.'
        })
    
    except Exception as e:
        logger.error(f"List datasets error: {str(e)}")
        return jsonify({'status': 'error', 'message': f'Failed to list datasets: {str(e)}'}), 500

# Route /analyze: Phân tích shape và columns từ CSV
@app.route('/analyze', methods=['POST'])
def analyze_columns():
    try:
        # Sửa đường dẫn
        file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'kepler', 'kepler data.csv')
        logger.info(f"Đang load file: {file_path}")
        df = pd.read_csv(file_path, comment='#')
        
        shape = df.shape
        columns = df.columns.tolist()
        
        result = {
            'status': 'success',
            'shape': {'rows': int(shape[0]), 'cols': int(shape[1])},
            'columns': columns,
            'message': f'Phân tích thành công: {shape[0]} hàng, {shape[1]} cột.',
            'frontend_redirect': f'{frontend_url}/preprocess'
        }
        
        logger.info(f"Analyze OK: shape {shape}, {len(columns)} columns")
        return jsonify(result)
    
    except FileNotFoundError as fe:
        logger.error(f"FileNotFound: {fe}")
        return jsonify({'status': 'error', 'message': f'File "kepler data.csv" không tồn tại ở {file_path}.'}), 404
    except Exception as e:
        logger.error(f"Analyze error: {str(e)}")
        return jsonify({'status': 'error', 'message': f'Lỗi phân tích: {str(e)}'}), 500

# Route /create_variant: Tạo variant dữ liệu từ columns và name
@app.route('/create_variant', methods=['POST'])
def create_variant():
    try:
        data = request.get_json(force=True)
        logger.info(f"Received payload: {data}")
        
        columns = data.get('columns', [])
        name = data.get('name', 'default_variant').strip()
        remove_outliers = data.get('remove_outliers', False)
        
        logger.info(f"Calling preprocess_kepler_data with: name={name}, columns={columns}, remove_outliers={remove_outliers}")
        
        if not name:
            logger.error("Dataset name is required.")
            return jsonify({'status': 'error', 'message': 'Dataset name is required.'}), 400
        
        # Unpack 4 returns (an toàn: imputer_path có thể None)
        csv_path, imputer_path, scaler_path, stats = preprocess_kepler_data(columns, name, remove_outliers)
        stats = convert_numpy_types(stats)
        
        logger.info(f"Preprocessing completed: stats={stats}")
        
        message = 'Tạo biến thể thành công!'
        if stats['flag_noise_dropped'] > 0:
            message += f' (Loại bỏ {stats["flag_noise_dropped"]} rows flag nhiễu)'
        if stats['outliers_dropped'] > 0:
            message += f' (Loại bỏ {stats["outliers_dropped"]} rows outliers)'
        if stats['total_noise_removed_pct'] > 0:
            message += f' (Tổng nhiễu loại bỏ: {stats["total_noise_removed_pct"]}%)'
        
        # Files dict: Backward-compatible, thêm imputer optional
        files = {'csv': csv_path, 'scaler': scaler_path}
        if imputer_path:
            files['imputer'] = imputer_path
        
        return jsonify({
            'status': 'success',
            'message': message,
            'name': name,
            'files': files,
            'stats': stats
        })
    
    except ValueError as ve:
        logger.error(f"ValueError in create_variant: {ve}")
        return jsonify({'status': 'error', 'message': str(ve)}), 400
    except Exception as e:
        logger.error(f"Create_variant error: {str(e)}")
        return jsonify({'status': 'error', 'message': f'Processing failed: {str(e)}'}), 500

# Route /train: Huấn luyện mô hình trên file CSV đã xử lý
@app.route('/train', methods=['POST'])
def train():
    try:
        data = request.get_json(force=True)
        logger.info(f"Received payload: {data}")
        
        dataset_name = data.get('dataset_name', '').strip()
        model_name = data.get('model_name', f'{dataset_name}_model').strip()
        
        if not dataset_name:
            logger.error("Dataset name is required.")
            return jsonify({'status': 'error', 'message': 'Dataset name is required.'}), 400
        if not model_name:
            logger.error("Model name is required.")
            return jsonify({'status': 'error', 'message': 'Model name is required.'}), 400
        
        # Sửa đường dẫn
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'kepler', f'{dataset_name}_processed.csv')
        logger.info(f"Training model with data: {data_path}, model_name: {model_name}")
        
        model_path, stats = train_kepler_model(data_path, model_name)
        stats = convert_numpy_types(stats)
        
        logger.info(f"Training completed: stats={stats}")
        
        message = f'Huấn luyện thành công! Test accuracy: {stats["test_accuracy"]:.2f}'
        
        return jsonify({
            'status': 'success',
            'message': message,
            'model_name': model_name,
            'model_path': model_path,
            'stats': stats
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

# Route /list_models: Liệt kê các models trong models/
@app.route('/list_models', methods=['GET'])
def list_models_route():
    try:
        models = list_models()
        return jsonify({
            'status': 'success',
            'models': models,
            'message': f'Found {len(models)} trained models.'
        })
    except Exception as e:
        logger.error(f"List models error: {str(e)}")
        return jsonify({'status': 'error', 'message': f'Failed to list models: {str(e)}'}), 500

# Route /predict: Predict trên model với input data
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        logger.info(f"Received predict payload: {data}")
        
        model_name = data.get('model_name', '').strip()
        input_data = data.get('input_data', {})  # Dict of features (raw)
        
        if not model_name:
            logger.error("Model name is required.")
            return jsonify({'status': 'error', 'message': 'Model name is required.'}), 400
        
        if not input_data:
            logger.error("Input data is required.")
            return jsonify({'status': 'error', 'message': 'Input data is required.'}), 400
        
        # Extract dataset_name
        dataset_name = model_name.replace('_model', '')
        
        # Preprocess input raw → df_processed (sử dụng hàm từ module)
        df_processed = transform_single_row(dataset_name, input_data)
        
        # Gọi predict với df_processed
        result = predict_kepler_model(model_name, df_processed)
        result = convert_numpy_types(result)
        
        logger.info(f"Prediction completed: {result}")
        
        message = f'Prediction: {result["prediction"]} (confidence: {result["confidence"]:.2f})'
        
        return jsonify({
            'status': 'success',
            'message': message,
            'model_name': model_name,
            'result': result
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

# Route /model_features: Lấy info features của model (số lượng, tên, importance)
@app.route('/model_features', methods=['GET'])
def model_features():
    try:
        model_name = request.args.get('model_name', '').strip()
        if not model_name:
            return jsonify({'status': 'error', 'message': 'Model name is required.'}), 400
        
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        features_path = os.path.join(models_dir, f'{model_name}_features.json')
        
        if not os.path.exists(features_path):
            logger.error(f"Features file not found: {features_path}")
            return jsonify({'status': 'error', 'message': f'Features info for "{model_name}" not found. Retrain model?'}), 404
        
        with open(features_path, 'r') as f:
            features_info = json.load(f)
        
        logger.info(f"Returned features for {model_name}: {features_info['num_features']} fields")
        return jsonify({
            'status': 'success',
            'model_name': model_name,
            'num_features': features_info['num_features'],
            'features': features_info['features'],
            'importance': features_info['importance']  # Có thể sort top 10 ở frontend
        })
    
    except Exception as e:
        logger.error(f"Model features error: {str(e)}")
        return jsonify({'status': 'error', 'message': f'Failed to get features: {str(e)}'}), 500
        
@app.route('/status', methods=['GET'])
def status():
    try:
        logger.info("Status check")
        return jsonify({
            'status': 'Backend running',
            'version': '1.0',
            'frontend_url': frontend_url
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
        'message': 'Lỗi server nội bộ. Vui lòng thử lại.',
        'path': request.path
    }), 500

if __name__ == '__main__':
    port = int(os.getenv('API_PORT', 5001))
    app.run(debug=True, host='0.0.0.0', port=port, use_reloader=False)