from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import rasterio
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
from PIL import Image

# --- Setup ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# --- Load model and normalization arrays ---
model = load_model("best_unet.keras", compile=False)
band_mins = np.load("band_mins.npy")
band_maxs = np.load("band_maxs.npy")

# --- Helper function ---
def preprocess_tif(tif_path):
    with rasterio.open(tif_path) as src:
        image = src.read().astype(np.float32)
        image = np.moveaxis(image, 0, -1)
        print("✅ Loaded TIF:", tif_path)
        print("Shape:", image.shape)
        print("Min/Max before norm:", image.min(), image.max())

    # Dynamic normalization per image
    image_min = image.min(axis=(0, 1), keepdims=True)
    image_max = image.max(axis=(0, 1), keepdims=True)
    image = (image - image_min) / (image_max - image_min + 1e-8)
    image = np.clip(image, 0, 1)
    print("After norm min/max:", image.min(), image.max())

    # RGB preview (auto-select)
    rgb_bands = [0, 1, 2] if image.shape[-1] == 3 else [3, 2, 1]
    rgb = (image[..., rgb_bands] * 255).astype(np.uint8)

    # Predict
    input_img = np.expand_dims(image, axis=0)
    print("Model input shape:", input_img.shape)
    prediction = model.predict(input_img)[0, ..., 0]
    print("Prediction stats:", prediction.min(), prediction.max(), prediction.mean())

    mask = (prediction > 0.5).astype(np.uint8) * 255
    return rgb, mask


# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    print(f"📁 File saved to {file_path}")

    try:
        rgb, mask = preprocess_tif(file_path)
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

    rgb_path = os.path.join(app.config['RESULT_FOLDER'], f"rgb_{filename}.png")
    mask_path = os.path.join(app.config['RESULT_FOLDER'], f"mask_{filename}.png")
    Image.fromarray(rgb).save(rgb_path)
    Image.fromarray(mask).save(mask_path)

    print("✅ Prediction complete")
    return jsonify({'rgb': rgb_path, 'mask': mask_path})
@app.route('/test')
def test():
    return "<h1>Flask test route works ✅</h1>"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True)

