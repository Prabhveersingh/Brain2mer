import streamlit as st
import numpy as np
import time
import tensorflow as tf
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
    border-left: 6px solid cyan;
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
    try:
        model = load_model(MODEL_PATH)
        # Model summary dikhao
        return model
    except Exception as e:
        st.error(f"Model load nahi ho raha: {e}")
        return None

model = load_cnn_model()

# Model info sidebar mein
if model:
    with st.sidebar:
        st.markdown("### 🧠 Model Status")
        st.success("✅ Model loaded successfully")
        
        # Check model input shape
        input_shape = model.input_shape
        st.write(f"Input shape: `{input_shape}`")
        
        # Expected preprocessing
        st.write(f"Expected image size: `{input_shape[1]} x {input_shape[2]}`")
        
        # Test with dummy image
        dummy = np.random.rand(1, IMAGE_SIZE, IMAGE_SIZE, 3)
        dummy_pred = model.predict(dummy, verbose=0)
        st.write(f"Test prediction shape: `{dummy_pred.shape}`")
        st.write(f"Classes: No Tumor (index 0) | Tumor (index 1)")

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
    
    # IMPORTANT: Same preprocessing jo pehle thi
    img_resized = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Prediction
    prediction_probs = model.predict(img_array, verbose=0)[0]
    pred_class = int(np.argmax(prediction_probs))
    confidence = float(np.max(prediction_probs)) * 100
    
    # Sidebar mein raw values dikhao
    with st.sidebar:
        st.markdown("### 📊 Raw Prediction")
        st.write(f"No Tumor: `{prediction_probs[0]:.6f}`")
        st.write(f"Tumor: `{prediction_probs[1]:.6f}`")
        st.write(f"Argmax: `{pred_class}`")
        st.write(f"Confidence: `{confidence:.2f}%`")
        
        if prediction_probs[0] > prediction_probs[1]:
            st.success("Model decision: NO TUMOR")
        else:
            st.error("Model decision: TUMOR")
    
    time.sleep(0.2)
    progress_bar.progress(100)
    status_text.markdown('<p style="color:lime;">✅ Complete!</p>', unsafe_allow_html=True)
    time.sleep(0.2)
    progress_bar.empty()
    status_text.empty()
    
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
        <p style="font-size:1.3rem; font-weight:bold;">Confidence: <span style="color:cyan;">{confidence:.2f}%</span></p>
    </div>
    """, unsafe_allow_html=True)

elif not model:
    st.error("❌ Model file 'model.h5' nahi mil raha. Check karo file exist karti hai ya nahi.")

else:
    st.markdown("""
    <div style="text-align:center; padding:2rem; background:rgba(255,255,255,0.03); border-radius:32px;">
        <p style="color:#aaa;">🌟 Upload MRI image</p>
    </div>
    """, unsafe_allow_html=True)