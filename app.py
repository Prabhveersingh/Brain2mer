import streamlit as st
import numpy as np
import time
from tensorflow.keras.models import load_model
from PIL import Image

st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="wide")

st.markdown("""
<style>
header { visibility: hidden; height: 0; }
.block-container { padding-top: 0rem !important; }
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    animation: gradientShift 10s ease infinite;
}
@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    100% { background-position: 100% 50%; }
}
.glass-card {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    border-radius: 32px;
    padding: 1.5rem;
    border: 1px solid rgba(255,255,255,0.2);
}
.neon-text {
    font-size: 2.5rem;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(135deg, #aaffff, #ff66ff);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}
div[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.05);
    border: 2px dashed cyan;
    border-radius: 24px;
    padding: 20px;
}
.result-card {
    background: linear-gradient(135deg, rgba(0,255,255,0.1), rgba(255,0,255,0.1));
    border-radius: 24px;
    padding: 1rem 1.5rem;
    margin-top: 1.5rem;
    animation: fadeInUp 0.6s ease-out;
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(30px); }
    to { opacity: 1; transform: translateY(0); }
}
.scan-overlay { position: relative; display: inline-block; border-radius: 20px; overflow: hidden; }
.scan-line {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 4px;
    background: linear-gradient(90deg, transparent, cyan, magenta, transparent);
    animation: scanMove 1.5s infinite;
}
@keyframes scanMove {
    0% { top: 0; }
    100% { top: 100%; }
}
</style>
""", unsafe_allow_html=True)

IMAGE_SIZE = 64
MODEL_PATH = "model.h5"

@st.cache_resource
def load_cnn_model():
    return load_model(MODEL_PATH)

model = load_cnn_model()

st.markdown('<div class="glass-card"><p class="neon-text">🧠 Brain Tumor Detection</p></div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Drop your MRI scan here", type=["jpg", "jpeg", "png"])

if uploaded_file and model:
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="scan-overlay">', unsafe_allow_html=True)
        st.image(image, caption="🧬 Uploaded MRI", use_container_width=True)
        st.markdown('<div class="scan-line"></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for percent in range(0, 101, 10):
        time.sleep(0.1)
        progress_bar.progress(percent)
        if percent < 30:
            status_text.markdown('<p style="color:cyan;">🔬 Initializing...</p>', unsafe_allow_html=True)
        elif percent < 70:
            status_text.markdown('<p style="color:magenta;">⚡ Analyzing...</p>', unsafe_allow_html=True)
        else:
            status_text.markdown('<p style="color:lime;">🧠 Processing...</p>', unsafe_allow_html=True)
    
    img_resized = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction_probs = model.predict(img_array, verbose=0)[0]
    tumor_prob = prediction_probs[1] if len(prediction_probs) > 1 else prediction_probs[0]
    
    time.sleep(0.2)
    progress_bar.progress(100)
    status_text.markdown('<p style="color:lime;">✅ Complete!</p>', unsafe_allow_html=True)
    time.sleep(0.2)
    progress_bar.empty()
    status_text.empty()
    
    # ---------- THRESHOLD: 55% ----------
    # Agar 55% se zyada tumor probability → Tumor, warna No Tumor
    if tumor_prob >= 0.55:
        result_title = "⚠️ Tumor Detected"
        result_icon = "🧠⚠️"
        color = "#ff4d4d"
        confidence = tumor_prob * 100
    else:
        result_title = "✅ No Tumor Detected"
        result_icon = "🧠✅"
        color = "#4caf50"
        confidence = (1 - tumor_prob) * 100
    
    st.markdown(f"""
    <div class="result-card" style="border-left: 6px solid {color};">
        <h2 style="margin:0; color:{color};">{result_icon} {result_title}</h2>
        <hr style="background:{color}; height:2px; border:none;">
        <p style="font-size:1.3rem; font-weight:bold;">Confidence: <span style="color:cyan;">{confidence:.1f}%</span></p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center; padding:2rem; background:rgba(255,255,255,0.03); border-radius:32px;">
        <p style="color:#aaa;">🌟 Upload MRI image to begin scanning</p>
    </div>
    """, unsafe_allow_html=True)