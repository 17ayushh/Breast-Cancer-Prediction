import streamlit as st
import numpy as np
import pickle

# Load the trained model
model = pickle.load(open('breast_cancer_model.pkl', 'rb'))

# Feature names (30)
feature_names = [
    "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
    "mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error", "smoothness error",
    "compactness error", "concavity error", "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area", "worst smoothness",
    "worst compactness", "worst concavity", "worst concave points", "worst symmetry", "worst fractal dimension"
]

st.title(" Breast Cancer Prediction App")
st.subheader("Enter the tumor measurements below")

# Create input fields for all features
inputs = []
for name in feature_names:
    val = st.text_input(f"{name}", key=name)
    inputs.append(val)

# Prediction button
if st.button("Predict"):
    try:
        # Convert inputs to float
        float_inputs = np.array([float(i) for i in inputs]).reshape(1, -1)
        prediction = model.predict(float_inputs)[0]

        if prediction == 1:
            st.error("⚠️ The tumor is likely **Malignant**")
        else:
            st.success("✅ The tumor is likely **Benign**")
    except ValueError:
        st.warning("⚠️ Please enter valid numerical values for all fields.")