from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
model = tf.keras.models.load_model("model.h5")

def preprocess(img):
    img = img.resize((64, 64))
    img = np.array(img) / 255.0
    return img.reshape(1, 64, 64, 3)

@app.route('/')
def home():
    return '''
        <h2>🧠 Brain Tumor Detection (Flask)</h2>
        <form action="/predict" method="post" enctype="multipart/form-data">
            <input type="file" name="image">
            <input type="submit" value="Predict">
        </form>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No file uploaded."

    file = request.files['image']
    img = Image.open(file.stream).convert("RGB")
    processed = preprocess(img)
    pred = model.predict(processed)
    label = "Tumor" if np.argmax(pred) == 1 else "No Tumor"
    confidence = float(np.max(pred))

    return f"<h3>Prediction: {label} ({confidence*100:.2f}%)</h3>"

if __name__ == '__main__':
    app.run(debug=True)
