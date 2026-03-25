import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import random

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('plant_disease_model.h5')
    with open('class_indices.json', 'r') as f:
        class_indices = json.load(f)
        labels = {v: k for k, v in class_indices.items()}
    return model, labels

model, labels = load_model()

# --- FAKE SENSORS ---
soil = random.randint(30, 70)
temp = round(random.uniform(20.0, 35.0), 1)

# --- UI ---
st.title("🌾 Smart Agri-Vision Crop Doctor")
st.write(f"**Live Sensors** -> Soil Moisture: {soil}% | Temperature: {temp}°C")

st.write("Upload a photo of a leaf, or use your camera to scan it!")

# Camera input OR file upload
uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])
camera_file = st.camera_input("Take a picture")

image_to_process = camera_file if camera_file else uploaded_file

if image_to_process is not None:
    image = Image.open(image_to_process)
    st.image(image, caption='Scanned Leaf', use_column_width=True)
    
    st.write("Scanning...")
    
    # Process image for AI
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    predictions = model.predict(img_array)
    max_score = np.max(predictions)
    index = np.argmax(predictions)
    
    display_name = labels[index].replace("_", " ")
    
    if "healthy" in display_name.lower():
        st.success(f"Detected: {display_name} ({int(max_score*100)}%)")
    else:
        st.error(f"Detected: {display_name} ({int(max_score*100)}%)")