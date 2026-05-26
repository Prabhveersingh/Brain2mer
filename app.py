import streamlit as st
import numpy as np
import time
import os
from tensorflow.keras.models import load_model
from PIL import Image
from datetime import datetime

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
    font-size: 2rem;
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
    padding: 1rem;
    margin-top: 1rem;
    animation: fadeInUp 0.6s ease-out;
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(30px); }
    to { opacity: 1; transform: translateY(0); }
}
.scan-line {
    height: 4px;
    background: linear-gradient(90deg, transparent, cyan, magenta, transparent);
    animation: scanMove 1.5s infinite;
}
@keyframes scanMove {
    0% { width: 0%; }
    100% { width: 100%; }
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

# Multi-file upload
uploaded_files = st.file_uploader(
    "📤 Select multiple MRI images (100-150 at once)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files and model:
    st.markdown(f"### 📊 Scanning {len(uploaded_files)} images...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # List to store wrong detections
    wrong_detections = []
    correct_no_tumor = []
    correct_tumor = []
    
    for idx, uploaded_file in enumerate(uploaded_files):
        # Update progress
        percent = int((idx + 1) / len(uploaded_files) * 100)
        progress_bar.progress(percent)
        status_text.markdown(f'<p style="color:cyan;">🔬 Scanning image {idx + 1} of {len(uploaded_files)}...</p>', unsafe_allow_html=True)
        
        # Process image
        image = Image.open(uploaded_file).convert('RGB')
        img_resized = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        prediction_probs = model.predict(img_array, verbose=0)[0]
        tumor_prob = prediction_probs[1] if len(prediction_probs) > 1 else prediction_probs[0]
        pred_class = np.argmax(prediction_probs)
        
        # File name without extension
        file_name = uploaded_file.name
        
        # Logica: Agar "no" word hai file name mein aur model Tumor bata raha hai
        # Ya agar "tumor" word hai aur model No Tumor bata raha hai
        is_no_image = "no" in file_name.lower() or "normal" in file_name.lower()
        is_tumor_image = "tumor" in file_name.lower() or "cancer" in file_name.lower()
        
        if is_no_image and pred_class == 1:
            # No image ko Tumor dikhaya - WRONG
            wrong_detections.append(f"{file_name} | Tumor Probability: {tumor_prob*100:.1f}%")
            st.markdown(f'<p style="color:red;">❌ WRONG: {file_name} → Tumor (should be No Tumor)</p>', unsafe_allow_html=True)
        elif is_tumor_image and pred_class == 0:
            # Tumor image ko No Tumor dikhaya - WRONG
            wrong_detections.append(f"{file_name} | No Tumor Probability: {(1-tumor_prob)*100:.1f}%")
            st.markdown(f'<p style="color:red;">❌ WRONG: {file_name} → No Tumor (should be Tumor)</p>', unsafe_allow_html=True)
        elif pred_class == 1:
            correct_tumor.append(file_name)
            st.markdown(f'<p style="color:orange;">⚠️ {file_name} → Tumor (Correct)</p>', unsafe_allow_html=True)
        else:
            correct_no_tumor.append(file_name)
            st.markdown(f'<p style="color:green;">✅ {file_name} → No Tumor (Correct)</p>', unsafe_allow_html=True)
        
        # Small delay for animation
        time.sleep(0.05)
    
    progress_bar.progress(100)
    status_text.markdown('<p style="color:lime;">✅ Scan complete!</p>', unsafe_allow_html=True)
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()
    
    # SUMMARY
    st.markdown("---")
    st.markdown("## 📊 Summary Report")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Images", len(uploaded_files))
    with col2:
        st.metric("Wrong Detections", len(wrong_detections), delta="⚠️ Need Deletion")
    with col3:
        st.metric("Correct Detections", len(correct_no_tumor) + len(correct_tumor))
    
    # Show wrong detections list
    if wrong_detections:
        st.markdown("### 🗑️ Files to Delete (Wrongly Detected as Tumor)")
        
        wrong_text = "\n".join(wrong_detections)
        st.code(wrong_text, language="text")
        
        # Create TXT file for download
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_content = f"""
BRAIN TUMOR DETECTION REPORT
================================
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Total Images Scanned: {len(uploaded_files)}
Wrong Detections: {len(wrong_detections)}
Correct Detections: {len(correct_no_tumor) + len(correct_tumor)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🗑️ FILES TO DELETE (Wrongly Detected)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{chr(10).join(wrong_detections)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ CORRECTLY DETECTED - NO TUMOR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{chr(10).join(correct_no_tumor) if correct_no_tumor else "None"}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ CORRECTLY DETECTED - TUMOR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{chr(10).join(correct_tumor) if correct_tumor else "None"}
"""
        
        st.download_button(
            label="📥 Download Delete List (TXT)",
            data=report_content,
            file_name=f"delete_these_files_{timestamp}.txt",
            mime="text/plain"
        )
        
        st.info("💡 **How to delete:** Download the TXT file, open in Termux, and run:\n```bash\ncd /storage/emulated/0/Download/B2R-RAi-main/B2R-RAi-main/extracted/no\nrm file1.jpg file2.jpg\n```")
    else:
        st.success("🎉 All images detected correctly! No files to delete.")
    
    # Show sample of correct detections
    with st.expander("📋 View All Results"):
        st.write("**✅ No Tumor (Correct):**", correct_no_tumor[:20])
        st.write("**⚠️ Tumor (Correct):**", correct_tumor[:20])
        if wrong_detections:
            st.write("**❌ Wrong Detections (Delete these):**", wrong_detections)

else:
    st.markdown("""
    <div style="text-align:center; padding:2rem; background:rgba(255,255,255,0.03); border-radius:32px;">
        <p style="color:#aaa;">🌟 Select multiple MRI images (100-150 at once)</p>
        <small style="color:#555;">Press Ctrl/Cmd + Click to select multiple files</small>
    </div>
    """, unsafe_allow_html=True)