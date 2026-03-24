import pandas as pd
import numpy as np
import pickle


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# ------------------------
# Load Data
# ------------------------
data = pd.read_csv("car_prediction_data.csv")

print(data.head())

# ------------------------
# Feature Engineering
# ------------------------
# Convert Year → Age
data['Car_Age'] = 2025 - data['Year']
data.drop(['Year', 'Car_Name'], axis=1, inplace=True)

# ------------------------
# One-Hot Encoding
# ------------------------
data = pd.get_dummies(data, drop_first=True)

print(data.columns)

# ------------------------
# Split Features & Target
# ------------------------
X = data.drop("Selling_Price", axis=1)
y = data["Selling_Price"]

# ------------------------
# Train/Test Split
# ------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------
# Models
# ------------------------
lr = LinearRegression()
dt = DecisionTreeRegressor(random_state=42)
rf = RandomForestRegressor(n_estimators=100, random_state=42)

models = {
    "Linear Regression": lr,
    "Decision Tree": dt,
    "Random Forest": rf
}

# ------------------------
# Train & Evaluate
# ------------------------
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n{name}")
    print("R2 Score:", r2_score(y_test, y_pred))
    print("MAE:", mean_absolute_error(y_test, y_pred))

best_model = rf  # usually RF wins

with open("car_price_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("✅ Best model saved!")

feature_importance = pd.Series(
    rf.feature_importances_, index=X.columns
).sort_values(ascending=False)

print("\nFeature Importance:\n", feature_importance)