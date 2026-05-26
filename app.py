import streamlit as st
import numpy as np
import time
from tensorflow.keras.models import load_model
from PIL import Image

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="wide")

# ---------- CSS ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

/* Remove Streamlit default header and padding */
header {
    visibility: hidden;
    height: 0;
}

.block-container {
    padding-top: 0rem !important;
    padding-bottom: 1rem !important;
}

body {
    font-family: 'Poppins', sans-serif;
}

/* Animated Gradient Background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    background-size: 400% 400%;
    animation: gradientShift 10s ease infinite;
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Glassmorphic Card */
.glass-card {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(12px);
    border-radius: 32px;
    padding: 1.5rem;
    border: 1px solid rgba(255, 255, 255, 0.2);
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    margin-top: 0;
    margin-bottom: 1rem;
}

/* Neon Glow Text */
.neon-text {
    font-size: 2.5rem;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(135deg, #aaffff, #ff66ff);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    text-shadow: 0 0 10px rgba(170, 255, 255, 0.5);
    animation: pulse 2s infinite;
    margin-top: 0;
}

@keyframes pulse {
    0% { opacity: 0.8; text-shadow: 0 0 5px cyan; }
    100% { opacity: 1; text-shadow: 0 0 20px magenta; }
}

/* Upload Button Styling */
div[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.05);
    border: 2px dashed cyan;
    border-radius: 24px;
    padding: 20px;
    transition: all 0.3s;
}

div[data-testid="stFileUploader"]:hover {
    border-color: magenta;
    background: rgba(255,255,255,0.1);
    box-shadow: 0 0 20px cyan;
}

/* Result Card */
.result-card {
    background: linear-gradient(135deg, rgba(0,255,255,0.1), rgba(255,0,255,0.1));
    border-left: 6px solid cyan;
    border-radius: 24px;
    padding: 1rem 1.5rem;
    margin-top: 1.5rem;
    animation: fadeInUp 0.6s ease-out;
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Scanning Animation Overlay */
.scan-overlay {
    position: relative;
    display: inline-block;
    border-radius: 20px;
    overflow: hidden;
}

.scan-line {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 4px;
    background: linear-gradient(90deg, transparent, cyan, magenta, transparent);
    animation: scanMove 1.5s cubic-bezier(0.4, 0.0, 0.2, 1) infinite;
    box-shadow: 0 0 12px cyan;
}

@keyframes scanMove {
    0% { top: 0; }
    100% { top: 100%; }
}
</style>
""", unsafe_allow_html=True)

# ---------- LOAD MODEL ----------
IMAGE_SIZE = 64
MODEL_PATH = "model.h5"

@st.cache_resource
def load_cnn_model():
    return load_model(MODEL_PATH)

model = load_cnn_model()

# ---------- TITLE ----------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<p class="neon-text">🧠 Brain Tumor Detection</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------- UPLOAD ----------
uploaded_file = st.file_uploader("📤 Drop your MRI scan here", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="scan-overlay">', unsafe_allow_html=True)
        st.image(image, caption="🧬 Uploaded MRI", use_container_width=True)
        st.markdown('<div class="scan-line"></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Scanning progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for percent in range(0, 101, 10):
        time.sleep(0.15)
        progress_bar.progress(percent)
        if percent < 30:
            status_text.markdown('<p style="color:cyan;">🔬 Initializing neural scan...</p>', unsafe_allow_html=True)
        elif percent < 70:
            status_text.markdown('<p style="color:magenta;">⚡ Analyzing MRI signals...</p>', unsafe_allow_html=True)
        else:
            status_text.markdown('<p style="color:lime;">🧠 Cross-referencing tumor patterns...</p>', unsafe_allow_html=True)
    
    # Prediction
    img_resized = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction_probs = model.predict(img_array)[0]
    pred_class = np.argmax(prediction_probs)
    confidence = float(np.max(prediction_probs)) * 100
    
    time.sleep(0.3)
    progress_bar.progress(100)
    status_text.markdown('<p style="color:lime;">✅ Scan complete!</p>', unsafe_allow_html=True)
    time.sleep(0.2)
    progress_bar.empty()
    status_text.empty()
    
    # Result (no advice, no download button)
    if pred_class == 1:
        result_title = "⚠️ Tumor Detected"
        result_icon = "🧠⚠️"
        color = "#ff4d4d"
    else:
        result_title = "✅ No Tumor Detected"
        result_icon = "🧠✅"
        color = "#4caf50"
    
    st.markdown(f"""
    <div class="result-card" style="border-left-color: {color};">
        <h2 style="margin:0; color:{color};">{result_icon} {result_title}</h2>
        <hr style="background:{color}; height:2px; border:none;">
        <p style="font-size:1.3rem; font-weight:bold;">Confidence Score: <span style="color:cyan;">{confidence:.2f}%</span></p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center; padding:2rem; background:rgba(255,255,255,0.03); border-radius:32px;">
        <p style="color:#aaa;">🌟 Upload an MRI image to begin scanning</p>
        <small style="color:#555;">Supports JPG, PNG</small>
    </div>
    """, unsafe_allow_html=True)