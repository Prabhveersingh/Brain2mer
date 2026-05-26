import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

IMAGE_SIZE = 64
MODEL_PATH = "model.h5"

@st.cache_resource
def load_cnn_model():
    return load_model(MODEL_PATH)

model = load_cnn_model()

st.title("🧠 Brain Tumor Detection from MRI")
st.markdown("Upload a brain MRI image (JPG/PNG) to detect if it shows signs of a tumor.")

uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = np.argmax(model.predict(img_array))
    result = "🧠 Tumor Detected" if pred == 1 else "✅ No Tumor Detected"
    st.subheader("🔍 Prediction Result:")
    st.success(result)