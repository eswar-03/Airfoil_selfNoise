import streamlit as st
import joblib
import numpy as np

# Load trained model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Prediction function
def predict_sspl(f, alpha, c, U_infinity, delta):
    input_data = np.array([[f, alpha, c, U_infinity, delta]])

    # Scale input
    scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(scaled)

    return round(float(prediction[0]), 2)


# Streamlit page configuration
st.set_page_config(
    page_title="Airfoil Self-Noise Predictor",
    page_icon="✈️",
    layout="centered"
)

# Title
st.title("✈️ Airfoil Self-Noise Predictor")

st.write(
    "Enter the airfoil parameters below to predict "
    "the Sound Pressure Level (SSPL)."
)

# Input fields
f = st.number_input(
    "Frequency (f)",
    value=1000.0
)

alpha = st.number_input(
    "Angle of Attack (alpha)",
    value=0.0
)

c = st.number_input(
    "Chord Length (c)",
    value=0.1
)

U_infinity = st.number_input(
    "Free-stream Velocity (U_infinity)",
    value=50.0
)

delta = st.number_input(
    "Displacement Thickness (delta)",
    value=0.001
)

# Prediction button
if st.button("Predict SSPL"):

    prediction = predict_sspl(
        f,
        alpha,
        c,
        U_infinity,
        delta
    )

    st.success(f"Predicted SSPL: {prediction} dB")
