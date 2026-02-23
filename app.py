import cv2
import numpy as np
import tensorflow as tf
import json
import time
import random

# --- LOAD THE AI MODEL ---
try:
    print("Loading Model...")
    model = tf.keras.models.load_model('plant_disease_model.h5')
    with open('class_indices.json', 'r') as f:
        class_indices = json.load(f)
        labels = {v: k for k, v in class_indices.items()}
    print("Model Loaded!")
except:
    print("Error: Model not found. Run train_model.py first.")
    exit()

# --- FAKE SENSORS (Soil & Weather/Temp) ---
def update_sensors():
    return random.randint(30, 70), round(random.uniform(20.0, 35.0), 1)

# --- ADVICE LOGIC ---
def get_advice(disease_name):
    name_lower = disease_name.lower()
    if "healthy" in name_lower:
        return "Plant is healthy. Moderate watering."
    elif "blight" in name_lower or "rot" in name_lower:
        return "Fungal issue: Apply fungicide, reduce humidity."
    elif "rust" in name_lower or "spot" in name_lower:
        return "Bacterial/Fungal: Remove affected leaves, use copper spray."
    elif "virus" in name_lower:
        return "Virus detected: Isolate plant immediately."
    else:
        return "Disease detected: Monitor closely and check soil."

# --- APP SETUP ---
cap = cv2.VideoCapture(0)
current_result = "Ready: Align leaf in the box and press 's'"
current_color = (0, 255, 255) # Yellow
current_advice = "> Waiting for scan..."
soil, temp = update_sensors()
last_sensor_update = time.time()

print("Camera open. Press 's' to scan, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    h, w, _ = frame.shape

    # Update sensors every 3 seconds
    if time.time() - last_sensor_update > 3:
        soil, temp = update_sensors()
        last_sensor_update = time.time()

    # --- DRAW UI ---
    # Black info box at bottom
    cv2.rectangle(frame, (0, int(h - 160)), (w, h), (0, 0, 0), -1)
    
    # --- DRAW SCANNER TARGET BOX ---
    box_size = 300
    x1 = int(w/2 - box_size/2)
    y1 = int((h-160)/2 - box_size/2)
    x2 = x1 + box_size
    y2 = y1 + box_size
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
    cv2.putText(frame, "PLACE LEAF HERE", (x1 + 60, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Detect Keyboard Presses
    key = cv2.waitKey(1) & 0xFF
    
    # IF USER PRESSES 's' -> RUN THE AI
    if key == ord('s'):
        current_result = "Scanning..."
        cv2.imshow('Agri-Project Scanner', frame)
        cv2.waitKey(10) # Force UI update
        
        # 1. Grab the image INSIDE the box
        roi = frame[y1:y2, x1:x2]
        
        # 2. Fix colors
        img_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        
        # 3. ANTI-SCREEN FILTER: Blurs out the phone pixels so the AI sees the leaf clearly
        img_filtered = cv2.GaussianBlur(img_rgb, (5, 5), 0)
        
        # 4. Resize and Predict
        img_resized = cv2.resize(img_filtered, (224, 224))
        img_array = np.expand_dims(img_resized, axis=0) / 255.0
        
        predictions = model.predict(img_array, verbose=0)
        max_score = np.max(predictions)
        index = np.argmax(predictions)
        
        display_name = labels[index].replace("_", " ")
        current_result = f"Detected: {display_name} ({int(max_score*100)}%)"
        current_advice = f"> {get_advice(display_name)}"
        
        if "healthy" in display_name.lower():
            current_color = (0, 255, 0)
        else:
            current_color = (0, 0, 255)

    elif key == ord('q'):
        break

    # --- DISPLAY TEXT ---
    text_y = int(h - 120)
    cv2.putText(frame, current_result, (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, current_color, 2)
    cv2.putText(frame, f"Soil: {soil}% | Temp: {temp}C", (20, text_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, current_advice, (20, text_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1)
    cv2.putText(frame, "INSTRUCTIONS: Fill the white box with the leaf and press 's'", (20, text_y + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    cv2.imshow('Agri-Project Scanner', frame)

cap.release()
cv2.destroyAllWindows()