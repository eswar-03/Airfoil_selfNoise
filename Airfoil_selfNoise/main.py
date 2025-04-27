import gradio as gr
import joblib
import numpy as np

# Load trained model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Define prediction function
def predict_sspl(f, alpha, c, U_infinity, delta):
    input_data = np.array([[f, alpha, c, U_infinity, delta]])
    scaled = scaler.transform(input_data)
    prediction = model.predict(scaled)
    return round(prediction[0], 2)

# Create Gradio interface
interface = gr.Interface(
    fn=predict_sspl,
    inputs=[
        gr.Number(label="Frequency (f)"),
        gr.Number(label="Angle of Attack (alpha)"),
        gr.Number(label="Chord Length (c)"),
        gr.Number(label="Free-stream Velocity (U_infinity)"),
        gr.Number(label="Displacement Thickness (delta)"),
    ],
    outputs=gr.Number(label="Predicted SSPL"),
    title="Airfoil Self-Noise Predictor",
    description="Enter input features to predict Sound Pressure Level (SSPL)."
)

interface.launch()