import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
from sklearn.preprocessing import StandardScaler  # Thêm để load scaler nếu cần

# Load dữ liệu đã xử lý từ uncut
data_path = 'data/uncut/kepler_processed_uncut.csv'
df = pd.read_csv(data_path)

# Tách features (X) và target (y)
X = df.drop('koi_disposition', axis=1)
y = df['koi_disposition']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Xây dựng mô hình Random Forest với GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],  # Mở rộng để thử nghiệm
    'max_depth': [8, 10, 12],
    'min_samples_split': [5, 10, 15],
    'max_features': ['sqrt', 'log2']
}
model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Lấy mô hình tốt nhất
best_model = grid_search.best_estimator_
print(f"\nTham số tốt nhất: {grid_search.best_params_}")

# Dự đoán và đánh giá trên cả train và test
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"\nĐộ chính xác trên tập train: {train_accuracy:.2f}")
print(f"Độ chính xác trên tập test: {test_accuracy:.2f}")

# Kiểm tra feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("\nFeature Importance (từ cao đến thấp):")
print(feature_importance.to_string(index=False))

# Lưu mô hình với tên phù hợp cho uncut variant
with open('models/random_forest_kp_model_uncut.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("Mô hình đã được lưu tại: models/random_forest_kp_model_uncut.pkl")