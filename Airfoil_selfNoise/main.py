import streamlit as st
import joblib
import numpy as np
import os

# Get the folder where main.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model and scaler
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))


def predict_sspl(f, alpha, c, U_infinity, delta):
    input_data = np.array([[f, alpha, c, U_infinity, delta]])

    scaled = scaler.transform(input_data)

    prediction = model.predict(scaled)

    return round(float(prediction[0]), 2)


st.set_page_config(
    page_title="Airfoil Self-Noise Predictor",
    page_icon="✈️"
)

st.title("✈️ Airfoil Self-Noise Predictor")

st.write(
    "Enter the airfoil parameters to predict "
    "Sound Pressure Level (SSPL)."
)

f = st.number_input("Frequency (f)", value=1000.0)
alpha = st.number_input("Angle of Attack (alpha)", value=0.0)
c = st.number_input("Chord Length (c)", value=0.1)
U_infinity = st.number_input(
    "Free-stream Velocity (U_infinity)",
    value=50.0
)
delta = st.number_input(
    "Displacement Thickness (delta)",
    value=0.001
)

if st.button("Predict SSPL"):

    result = predict_sspl(
        f,
        alpha,
        c,
        U_infinity,
        delta
    )

    st.success(f"Predicted SSPL: {result} dB")
