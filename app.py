import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications import VGG16
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd # To make the output look nice

# --- 1. DEFINE YOUR MODEL'S "SKELETON" ---
# This function builds the *exact same* architecture as your winning
# 90.92% Kaggle model (Model 7: VGG16 + Flatten/Deeper).

def build_champion_model():
    # Define constants
    IMG_SIZE = (224, 224)
    num_classes = 4 # We know this from our data
    
    # Load the "Eyes" (VGG16)
    base_model_vgg16 = VGG16(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights='imagenet' # It will download this *once*
    )
    base_model_vgg16.trainable = False # Freeze the eyes
    
    # Build the "Head"
    model = Sequential([
        base_model_vgg16,               # The "Eyes"
        Flatten(),                      # The "Neck" that works
        Dense(256, activation='relu'),  # "Thinking" Layer 1
        Dropout(0.5),                   
        Dense(128, activation='relu'),  # "Thinking" Layer 2
        Dropout(0.3),                   
        Dense(num_classes, activation='softmax') # The "Decision"
    ])
    
    return model

# --- 2. LOAD YOUR CHAMPION "WEIGHTS" ---
print("Building model skeleton...")
model = build_champion_model()

print("Loading champion weights...")
# This is the new, smaller file you just created
weights_file = 'model_FINAL_BEST.weights.h5' 
model.load_weights(weights_file)
# --- END OF FIX ---

print("Model loaded successfully!")


# --- 3. DEFINE YOUR CLASS NAMES ---
CLASS_NAMES = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TUBERCULOSIS']
IMG_SIZE = (224, 224)

# --- 4. DEFINE THE "BRAIN" FUNCTION ---
def predict(input_image):
    
    # 1. Pre-process the image
    img = ImageOps.fit(input_image, IMG_SIZE, Image.Resampling.LANCZOS)
    img_array = np.array(img)
    
    if img_array.ndim == 2: 
        img_array = np.stack((img_array,)*3, axis=-1)
    
    img_array = img_array / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    
    # 2. Make Prediction
    predictions = model.predict(img_batch)[0]
    
    # 3. Format the Output
    confidences = {CLASS_NAMES[i]: float(predictions[i]) for i in range(len(CLASS_NAMES))}
    
    return confidences

# --- 5. BUILD THE WEB INTERFACE (GRADIO) ---
title = "Infectious Lung Disease AI Classifier"
description = (
    "A demo for my AI project. This model (VGG16 + Flatten/Deeper Head) "
    "was trained on a Kaggle dataset of Chest X-Rays to identify COVID-19, "
    "Normal, Pneumonia, or Tuberculosis."
)

gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a Chest X-Ray"), 
    outputs=gr.Label(num_top_classes=4, label="Diagnosis Results"),
    title=title,
    description=description,
    examples=[
        # "example_covid.jpeg",
        # "example_pneumonia.jpeg"
    ]
).launch()

