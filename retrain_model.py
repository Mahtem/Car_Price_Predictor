# car_price_predictor.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import shap

# ------------------------
# Load Data
# ------------------------
data_path = r"D:\ML-Projects\Car_Price Prediction\car_prediction_data.csv"
df = pd.read_csv(data_path)

# ------------------------
# Feature Engineering
# ------------------------
# Compute Car_Age
current_year = 2026
df['Car_Age'] = current_year - df['Year']

# Encode categorical variables
df = pd.get_dummies(df, columns=['Fuel_Type', 'Seller_Type', 'Transmission'], drop_first=True)

# Features & target
features = [
    'Kms_Driven', 'Owner', 'Car_Age',
    'Fuel_Type_Diesel', 'Fuel_Type_Petrol',
    'Seller_Type_Individual', 'Transmission_Manual'
]
X = df[features]
y = df['Selling_Price']

# ------------------------
# Train/Test Split
# ------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------
# Train Random Forest
# ------------------------
rf_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)
rf_model.fit(X_train, y_train)

# ------------------------
# Model Evaluation
# ------------------------
y_pred = rf_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='r2')

print("📊 Model Performance:")
print(f"R2 Score: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print("\n📊 Cross Validation R2 Scores:", cv_scores)
print("Mean CV R2:", cv_scores.mean())

# Feature Importance
importance = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)
print("\n🔥 Feature Importance:\n", importance)

# ------------------------
# Save Model
# ------------------------
model_path = r"D:\ML-Projects\Car_Price Prediction\car_price_model.pkl"
joblib.dump(rf_model, model_path)
print(f"\n✅ Model saved as {model_path}")

# ------------------------
# SHAP Explainability
# ------------------------
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X)

# Plot summary
shap.summary_plot(shap_values, X, feature_names=features)