import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import os

# --- NEW PATH SETTINGS (Matching your sidebar exactly) ---
TRAIN_DIR = 'dataset/plant village/train' 
VALID_DIR = 'dataset/plant village/valid'

if not os.path.exists(TRAIN_DIR):
    print(f"ERROR: Could not find folder at {TRAIN_DIR}")
    exit()

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

print("Found dataset! Starting preparation...")

# --- DATA SETUP (Updated for train/valid folders) ---
train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

validation_generator = valid_datagen.flow_from_directory(
    VALID_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# --- BUILD MODEL ---
print("Building the AI Brain...")
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False 

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- TRAIN ---
print("Training started... (This will take 10-15 mins)")
model.fit(train_generator, epochs=3, validation_data=validation_generator)

# --- SAVE ---
model.save('plant_disease_model.h5')
print("SUCCESS! Model saved as 'plant_disease_model.h5'")

import json
with open('class_indices.json', 'w') as f:
    json.dump(train_generator.class_indices, f)