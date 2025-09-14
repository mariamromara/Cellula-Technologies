import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --- Load model ---
MODEL_PATH = "best_teeth_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# --- Define classes ---
diseases = ["CaS", "CoS", "Gum", "MC", "OC", "OLP", "OT"]
img_size = (256, 256)  # must match training size

# --- Streamlit UI ---
st.title("🦷 Teeth Disease Classifier")
st.write("Upload an image of a tooth to classify its condition.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_array = np.array(image.resize(img_size, Image.Resampling.LANCZOS)) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

    # Prediction
    preds = model.predict(img_array)
    predicted_class = np.argmax(preds, axis=1)[0]
    confidence = np.max(preds)

    # Show results
    st.subheader(f"Prediction: **{diseases[predicted_class]}**")
    st.write(f"Confidence: {confidence:.2f}")

    # Show probabilities as bar chart
    st.bar_chart(dict(zip(diseases, preds[0])))
