# train_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os

# 1. Load and prepare data
df = pd.read_csv(r"C:\Users\User\VEHICLE_ANALYSIS_APP\cleaned_vehicles.csv")

# Create features
df['age'] = 2023 - df['year']
features = ['year', 'odometer', 'condition', 'manufacturer', 'fuel', 'transmission', 'age']
X = df[features].copy()
y = df['price']

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

# 3. Define preprocessing
numeric_features = ['year', 'odometer', 'age']
categorical_features = ['condition', 'manufacturer', 'fuel', 'transmission']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 4. Create and train pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

model.fit(X_train, y_train)

# 5. Calculate metrics
def calculate_metrics(model, X, y_true):
    y_pred = model.predict(X)
    return {
        'r2': r2_score(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),  # Manual RMSE calculation
        'samples': len(X)
    }

metrics = {
    'train': calculate_metrics(model, X_train, y_train),
    'test': calculate_metrics(model, X_test, y_test),
    'overall': calculate_metrics(model, X, y)
}

# 6. Save everything
os.makedirs('models', exist_ok=True)

save_data = {
    'model': model,
    'metrics': metrics,
    'feature_names': numeric_features + categorical_features,
    'timestamp': pd.Timestamp.now()
}

joblib.dump(save_data, 'models/vehicle_price_predictor.pkl')

# 7. Print results
print("Model training complete!")
print(f"Saved to: models/vehicle_price_predictor.pkl")
print("\nModel Performance:")
print(f"Train R²: {metrics['train']['r2']:.3f} (on {metrics['train']['samples']:,} samples)")
print(f"Test R²: {metrics['test']['r2']:.3f} (on {metrics['test']['samples']:,} samples)")
print(f"Test MAE: ${metrics['test']['mae']:,.2f}")
print(f"Test RMSE: ${metrics['test']['rmse']:,.2f}")