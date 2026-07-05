import streamlit as st
import numpy as np
import pickle

# Load Model and Scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Calories Burnt Prediction", page_icon="🔥")

st.title("🔥 Calories Burnt Prediction")

st.write("Enter your details below:")

# Inputs
gender = st.selectbox("Gender", ["Male", "Female"])

age = st.number_input("Age", min_value=1, max_value=100, value=25)

height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0)

weight = st.number_input("Weight (kg)", min_value=20.0, max_value=200.0, value=70.0)

duration = st.number_input("Exercise Duration (minutes)", min_value=1.0, max_value=300.0, value=30.0)

heart_rate = st.number_input("Heart Rate", min_value=40, max_value=220, value=90)

body_temp = st.number_input("Body Temperature (°C)", min_value=35.0, max_value=45.0, value=38.5)

# Prediction
if st.button("Predict Calories Burnt"):

    # Manual One-Hot Encoding
    gender_male = 1 if gender == "Male" else 0

    features = np.array([[age,
                          height,
                          weight,
                          duration,
                          heart_rate,
                          body_temp,
                          gender_male]])

    # Scale Input
    features = scaler.transform(features)

    # Predict
    prediction = model.predict(features)

    st.success(f"🔥 Estimated Calories Burnt: {prediction[0]:.2f} kcal")