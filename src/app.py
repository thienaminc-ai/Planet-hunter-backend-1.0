# src/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv  # Import để load .env
import os
import pandas as pd  # Giữ nguyên nếu có

# Load .env file
load_dotenv()

app = Flask(__name__)

# Lấy FRONTEND_URL từ env (fallback dev nếu thiếu)
frontend_url = os.getenv('FRONTEND_URL', 'http://localhost:3000')

# Cấu hình CORS: Chỉ cho phép từ frontend URL (an toàn)
CORS(app, origins=[frontend_url])

# Route /analyze (ví dụ, giữ nguyên code Pandas từ trước)
@app.route('/analyze', methods=['POST'])
def analyze_columns():
    try:
        file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'kepler_data.csv')
        df = pd.read_csv(file_path, comment='#')
        
        shape = df.shape
        columns = df.columns.tolist()
        
        result = {
            'status': 'success',
            'shape': {
                'rows': shape[0],
                'columns': shape[1]
            },
            'columns': columns,  # Chỉ danh sách tên cột
            'message': f'Phân tích thành công: {shape[0]} hàng, {shape[1]} cột.',
            'frontend_redirect': f'{frontend_url}/preprocess'
        }
        
        return jsonify(result)
    
    except FileNotFoundError:
        return jsonify({'status': 'error', 'message': 'File kepler_data.csv không tồn tại.'}), 404
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Lỗi phân tích: {str(e)}'}), 500

# Các route khác (thêm frontend_redirect tương tự nếu cần)
@app.route('/preprocess', methods=['POST'])
def preprocess():
    data = request.json
    return jsonify({
        'status': 'success',
        'processed_data': f'Cleaned {data.get("action", "data")}',
        'frontend_redirect': f'{frontend_url}/preprocess'
    })

@app.route('/train', methods=['POST'])
def train():
    return jsonify({
        'status': 'success',
        'accuracy': 0.95,
        'frontend_redirect': f'{frontend_url}/train'
    })

@app.route('/test', methods=['POST'])
def test():
    return jsonify({
        'status': 'success',
        'predictions': ['Exoplanet detected!'],
        'frontend_redirect': f'{frontend_url}/test'
    })

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        'status': 'Backend running',
        'version': '1.0',
        'frontend_url': frontend_url  # Test: Trả env var để check
    })

if __name__ == '__main__':
    port = int(os.getenv('API_PORT', 5001))
    app.run(debug=True, host='0.0.0.0', port=port)