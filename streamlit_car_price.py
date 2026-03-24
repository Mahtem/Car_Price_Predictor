# streamlit_car_price_production.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

# -----------------------------
# Load the trained model
# -----------------------------
with open("car_price_model.pkl", "rb") as f:
    model = pickle.load(f)

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="🚗 Car Price Predictor",
    layout="wide",
    page_icon="🚗"
)

# Title and description
st.title("🚗 Car Price Prediction with Explainable AI")
st.markdown(
    """
    Predict the selling price of a used car in India and see which features influenced the prediction.  
    **Features considered:** Kms Driven, Owner, Car Age, Fuel Type, Seller Type, Transmission.
    """
)

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Enter Car Details")

col1, col2 = st.sidebar.columns(2)

with col1:
    year = st.number_input("Year of Manufacture", min_value=1990, max_value=2026, value=2015)
    kms_driven = st.number_input("Kms Driven", min_value=0, max_value=500000, value=50000)
    owner = st.selectbox("No. of Previous Owners", [0, 1, 2, 3])

with col2:
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
    seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

# -----------------------------
# Preprocess Inputs
# -----------------------------
current_year = 2026
car_age = current_year - year

# One-hot encoding
fuel_type_diesel = 1 if fuel_type == "Diesel" else 0
fuel_type_petrol = 1 if fuel_type == "Petrol" else 0
seller_type_individual = 1 if seller_type == "Individual" else 0
transmission_manual = 1 if transmission == "Manual" else 0

input_features = np.array([[kms_driven, owner, car_age, fuel_type_diesel,
                            fuel_type_petrol, seller_type_individual, transmission_manual]])

feature_names = ['Kms_Driven', 'Owner', 'Car_Age', 'Fuel_Type_Diesel',
                 'Fuel_Type_Petrol', 'Seller_Type_Individual', 'Transmission_Manual']

# -----------------------------
# Prediction Section
# -----------------------------
st.subheader("🔹 Prediction")
if st.button("Predict Selling Price"):
    prediction = model.predict(input_features)[0]
    st.success(f"💰 Predicted Selling Price: ₹ {prediction:.2f} Lakhs")

    # -----------------------------
    # SHAP Explainability
    # -----------------------------
    st.subheader("🔹 Feature Contribution (SHAP)")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_features)

    # Force Plot
    st.write("Feature impact on prediction (positive/negative influence):")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    shap.initjs()
    shap.force_plot(explainer.expected_value, shap_values, input_features, feature_names=feature_names, matplotlib=True)
    st.pyplot(bbox_inches='tight')

    # Bar chart for global feature importance
    st.write("Feature importance (global view from Random Forest):")
    importances = model.feature_importances_
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(feature_names, importances, color="skyblue")
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest Feature Importance")
    ax.invert_yaxis()  # Highest importance at top
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f"{width:.2f}", va='center')
    st.pyplot(fig)

# -----------------------------
# Footer / Info
# -----------------------------
st.markdown("---")
st.markdown(
    """
    🔹 **Note:** This prediction is based on a machine learning model trained on Kaggle dataset.  
    🔹 **Model:** Random Forest Regressor
    🔹 **SHAP** shows which features increased or decreased the predicted price.
    """
)