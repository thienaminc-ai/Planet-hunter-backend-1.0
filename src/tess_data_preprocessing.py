import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pickle
import os
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FLAG_COLS_PATTERN = 'lim'  # TESS-specific: cols ending with 'lim' are 0/1 limit flags

def preprocess_tess_data(columns_to_keep, dataset_name, remove_outliers=False, raw_path=None, output_dir=None):
    """
    Quy trình tiền xử lý dữ liệu TESS-specific:
    - Bước 1: Tải và lọc cột (skip # comments)
    - Bước 2: Drop cột toàn NaN
    - Bước 3: Xử lý NaNs: impute median cho non-flags; drop rows nếu target NaN
    - Bước 3.5: Default drop rows flag noise (non-binary) cho cột có 'lim'
    - Bước 3.6: Nếu remove_outliers=True, drop rows outliers (IQR) cho non-flags
    - Bước 4: Giữ nguyên 6 nhãn của tfopwg_disp (không gộp thành nhị phân)
    - Bước 5: Scale non-flags
    - Bước 6: Lưu CSV/imputer/scaler + tính stats (rows, cols, noise removed %, label dist)
    Trả về: (csv_path, imputer_path, scaler_path, stats)
    """
    # Bước 1: Thiết lập đường dẫn
    if raw_path is None:
        raw_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'tess', 'tess data.csv')
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'tess')

    if not os.path.exists(raw_path):
        logger.error(f"File not found: {raw_path}")
        raise FileNotFoundError(f"Dataset file {raw_path} does not exist.")

    # Tải dữ liệu thô
    df = pd.read_csv(raw_path, comment='#')
    original_rows = df.shape[0]
    logger.info(f"Loaded data: {df.shape}")

    # Validate columns_to_keep
    if not columns_to_keep:
        logger.error("No columns specified for preprocessing.")
        raise ValueError("No columns specified.")
    
    invalid_columns = [col for col in columns_to_keep if col not in df.columns]
    if invalid_columns:
        logger.error(f"Invalid columns specified: {invalid_columns}")
        raise ValueError(f"Columns {invalid_columns} not found in dataset.")
    
    df = df[columns_to_keep]
    logger.info(f"Filtered to {len(columns_to_keep)} columns: {df.shape}")

    # Bước 2: Drop cột toàn NaN
    df = df.dropna(axis=1, how='all')
    logger.info(f"After dropping all-NaN columns: {df.shape}")

    # Xác định loại cột
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    flag_cols = [col for col in df.columns if col.endswith(FLAG_COLS_PATTERN) and col in numeric_cols]
    impute_cols = [col for col in numeric_cols if col not in flag_cols]
    scale_cols = [col for col in impute_cols if col != 'tfopwg_disp']
    logger.info(f"Flag columns: {flag_cols}")
    logger.info(f"Scale columns: {scale_cols}")

    target_col = 'tfopwg_disp' if 'tfopwg_disp' in df.columns else None
    if not target_col:
        logger.warning("Target column 'tfopwg_disp' not present – skipping target NaN drop.")

    # Bước 3: Xử lý NaNs cho target
    if target_col:
        initial_rows = df.shape[0]
        df = df.dropna(subset=[target_col])
        logger.info(f"Dropped {initial_rows - df.shape[0]} rows with NaN in target: {df.shape}")

    # Bước 3.5: Default drop rows flag noise (non-binary) cho cột có 'lim'
    dropped_flag_noise = 0
    if flag_cols:
        for col in flag_cols:
            # Kiểm tra giá trị không phải 0/1, kể cả NaN
            noise_mask = (df[col] < 0) | (df[col] > 1) | df[col].isna()
            col_dropped = noise_mask.sum()
            if col_dropped > 0:
                logger.info(f"Found {col_dropped} non-binary or NaN values in {col}: {df[col][noise_mask].unique()}")
                df = df[~noise_mask]
                logger.info(f"Dropped {col_dropped} rows with non-binary/NaN values in {col}: {df.shape}")
            dropped_flag_noise += col_dropped
        logger.info(f"Total dropped flag noise rows: {dropped_flag_noise}")
    else:
        logger.info("No flag columns (ending with 'lim') found.")

    # Bước 3.6: Drop outliers nếu remove_outliers=True
    dropped_outliers = 0
    if remove_outliers and scale_cols:
        outlier_mask = pd.Series(False, index=df.index)
        for col in scale_cols:
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            bounds = (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
            col_outliers = (df[col] < bounds[0]) | (df[col] > bounds[1])
            if col_outliers.sum() > 0:
                logger.info(f"Found {col_outliers.sum()} outliers in {col} (bounds: {bounds})")
            outlier_mask |= col_outliers
        dropped_outliers = outlier_mask.sum()
        if dropped_outliers > 0:
            df = df[~outlier_mask]
            logger.info(f"Dropped {dropped_outliers} rows with outliers (IQR method): {df.shape}")
        else:
            logger.info("No outliers detected.")
    elif remove_outliers and not scale_cols:
        logger.info("No numeric non-flag columns available for outlier removal.")

    # Bước 4: Giữ nguyên 6 nhãn của tfopwg_disp (không gộp thành nhị phân)
    if target_col:
        # Kiểm tra và thay thế giá trị không hợp lệ bằng "UNKNOWN" nếu cần
        valid_labels = ['FP', 'PC', 'KP', 'APC', 'FA', 'CP']
        df[target_col] = df[target_col].apply(lambda x: x if pd.notna(x) and x in valid_labels else 'UNKNOWN')
        logger.info(f"Kept 6 original labels for 'tfopwg_disp': {valid_labels}")

    # Bước 5: Impute non-flag numeric columns (sau khi xóa outliers để tránh ảnh hưởng)
    imputer = None
    imputer_path = None
    if impute_cols:
        imputer = SimpleImputer(strategy='median')
        df[impute_cols] = imputer.fit_transform(df[impute_cols])
        logger.info(f"Imputed median for {len(impute_cols)} non-flag numeric columns")

        # Save imputer
        os.makedirs(output_dir, exist_ok=True)
        imputer_path = os.path.join(output_dir, f'{dataset_name}_imputer.pkl')
        with open(imputer_path, 'wb') as f:
            pickle.dump(imputer, f)
        logger.info(f"Saved imputer: {imputer_path}")

    # Bước 6: Scale non-flag numeric columns
    scaler = None
    scaler_path = None
    if scale_cols:
        scaler = StandardScaler()
        df[scale_cols] = scaler.fit_transform(df[scale_cols])
        logger.info(f"Scaled {len(scale_cols)} non-flag numeric columns")

        # Save scaler
        os.makedirs(output_dir, exist_ok=True)
        scaler_path = os.path.join(output_dir, f'{dataset_name}_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        logger.info(f"Saved scaler: {scaler_path}")

    # Bước 7: Lưu file CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f'{dataset_name}_processed.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved processed CSV: {csv_path}")

    # Tính stats
    final_rows = df.shape[0]
    final_cols = df.shape[1]
    noise_removed_pct = ((original_rows - final_rows) / original_rows * 100) if original_rows > 0 else 0
    label_dist = {}
    if target_col:
        label_dist = (df[target_col].value_counts(normalize=True) * 100).round(2).to_dict()
        logger.info(f"Label distribution: {label_dist}")

    stats = {
        'original_rows': original_rows,
        'final_rows': final_rows,
        'final_cols': final_cols,
        'flag_noise_dropped': dropped_flag_noise,
        'outliers_dropped': dropped_outliers,
        'total_noise_removed_pct': round(noise_removed_pct, 2),
        'label_dist': label_dist
    }
    logger.info(f"Preprocessing stats: {stats}")

    return csv_path, imputer_path, scaler_path, stats

def transform_single_row(dataset_name, input_data, data_dir=None):
    """
    Transform single row input (raw) để đồng bộ với training (chỉ transform, không fit/drop).
    - Identify flag_cols, impute_cols, scale_cols (same logic as preprocess).
    - Load & apply imputer.transform nếu có file.
    - Load & apply scaler.transform nếu có file.
    - Giữ nguyên nhãn tfopwg_disp (không mã hóa).
    - Trả về df processed (single row).
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'tess')
    
    # Load imputer nếu có
    imputer_path = os.path.join(data_dir, f'{dataset_name}_imputer.pkl')
    imputer = None
    if os.path.exists(imputer_path):
        with open(imputer_path, 'rb') as f:
            imputer = pickle.load(f)
        logger.info(f"Loaded imputer: {imputer_path}")
    else:
        logger.warning(f"Imputer not found: {imputer_path} - Will fill NaN with 0")
    
    # Load scaler nếu có
    scaler_path = os.path.join(data_dir, f'{dataset_name}_scaler.pkl')
    scaler = None
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        logger.info(f"Loaded scaler: {scaler_path}")
    else:
        logger.warning(f"Scaler not found: {scaler_path} - No scaling applied")
    
    # Không sử dụng label encoder để giữ nguyên nhãn
    # Xóa phần load và transform label encoder
    
    # Tạo df input từ dict
    df_input = pd.DataFrame([input_data])
    
    # Identify cols (same as preprocess_tess_data)
    numeric_cols = df_input.select_dtypes(include=[np.number]).columns.tolist()
    flag_cols = [col for col in df_input.columns if col.endswith(FLAG_COLS_PATTERN) and col in numeric_cols]
    impute_cols = [col for col in numeric_cols if col not in flag_cols]
    scale_cols = [col for col in impute_cols if col != 'tfopwg_disp']  # Target không scale
    
    logger.info(f"Single row cols: Flag={flag_cols}, Impute={impute_cols}, Scale={scale_cols}")
    
    # Impute (transform nếu có, else fill 0)
    if impute_cols:
        if imputer:
            df_input[impute_cols] = imputer.transform(df_input[impute_cols])
        else:
            df_input[impute_cols] = df_input[impute_cols].fillna(0)
        logger.info(f"Transformed (imputed) {len(impute_cols)} columns")
    
    # Scale (transform nếu có)
    if scale_cols and scaler:
        df_input[scale_cols] = scaler.transform(df_input[scale_cols])
        logger.info(f"Transformed (scaled) {len(scale_cols)} columns")
    elif scale_cols:
        logger.warning("Scaler not found - Skipping scaling")
    
    # Giữ nguyên nhãn tfopwg_disp (không mã hóa)
    if 'tfopwg_disp' in df_input.columns:
        valid_labels = ['FP', 'PC', 'KP', 'APC', 'FA', 'CP']
        df_input['tfopwg_disp'] = df_input['tfopwg_disp'].apply(lambda x: x if pd.notna(x) and x in valid_labels else 'UNKNOWN')
        logger.info(f"Kept original 'tfopwg_disp' values with validation: {valid_labels}")
    
    return df_input
