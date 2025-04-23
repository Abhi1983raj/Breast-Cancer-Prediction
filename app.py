# app.py

import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# ✅ Set page config as the first Streamlit command
st.set_page_config(page_title="Breast Cancer Predictor", layout="centered")

# Load model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('breast_cancer_model_finetuned.h5')

model = load_model()
class_names = ['Benign', 'Malignant']

# Streamlit UI
st.title("🧬 Breast Cancer Prediction")
st.write("Upload a histopathological image of a breast tissue sample to predict whether it's **Benign** or **Malignant**.")

# File uploader
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    # Predict
    prediction = model.predict(img_array)[0]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Display results
    st.markdown(f"### 🧠 Prediction: **{predicted_class}**")
    st.markdown(f"**Confidence:** `{confidence:.2f}%`")

    # Optional: show raw probabilities as bar chart
    prob_dict = {class_names[i]: float(prediction[i]) for i in range(len(class_names))}
    st.subheader("Prediction Probabilities")
    st.bar_chart(prob_dict)
