import numpy as np
import tensorflow as tf
import json
from tensorflow.keras.preprocessing import image

print("Loading AI Brain...")
model = tf.keras.models.load_model('plant_disease_model.h5')

with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)
    labels = {v: k for k, v in class_indices.items()}

print("\nReady!")

img_path = input("Drag and drop a leaf image file into this terminal, then press Enter: ")


img_path = img_path.strip().strip("'").strip('"')

try:
    # 1. Load the image EXACTLY how the AI studied it
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # 2. Predict
    predictions = model.predict(img_array, verbose=0)
    max_score = np.max(predictions)
    disease_name = labels[np.argmax(predictions)]

    # 3. Print the result
    print("\n" + "="*40)
    print(f"DETECTED: {disease_name.replace('_', ' ')}")
    print(f"CONFIDENCE: {int(max_score * 100)}%")
    print("="*40 + "\n")

except Exception as e:
    print(f"\nError reading image: {e}")
    print("Make sure you dragged an actual image file (like .jpg or .png)")