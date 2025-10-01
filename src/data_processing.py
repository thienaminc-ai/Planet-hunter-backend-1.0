import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Tạo thư mục uncut nếu chưa có
os.makedirs('data/uncut', exist_ok=True)

# Load dữ liệu raw
df = pd.read_csv('data/kepler_data.csv', comment='#')

# Giữ chỉ cột số và koi_disposition (target), loại bỏ koi_pdisposition, flag, cột err, và cột name
columns_to_drop = ['koi_pdisposition', 'flag'] + [col for col in df.columns if 'err' in col.lower() or 'name' in col.lower()]
columns_to_keep = [col for col in df.columns if col not in columns_to_drop]
df = df[columns_to_keep]

# Cập nhật numeric_columns sau khi loại bỏ
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

# Xử lý NaN bằng mean
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# Chuẩn hóa
scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Lưu scaler và file xử lý
with open('data/uncut/scaler_uncut.pkl', 'wb') as f:
    pickle.dump(scaler, f)
df.to_csv('data/uncut/kepler_processed_uncut.csv', index=False)

# Xác nhận kết quả
print("\nSố hàng, số cột sau khi tiền xử lý:", df.shape)
print("Danh sách cột sau khi tiền xử lý:\n", df.columns.tolist())
print("Thống kê sau chuẩn hóa:\n", df[numeric_columns].describe())
print("\nFile đã được lưu tại: data/uncut/kepler_processed_uncut.csv")
print("Scaler đã được lưu tại: data/uncut/scaler_uncut.pkl")