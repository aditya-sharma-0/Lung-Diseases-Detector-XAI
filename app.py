import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd # To make the output look nice

# --- 1. LOAD YOUR CHAMPION MODEL (THE FIX) ---
# We load the model *once* when the app starts.
# It's stored in a global variable.
print("Loading champion model... This will run only once.")
# This MUST be the name of your best .h5 file
# (This name comes from your final, champion model)
model = load_model('model_CHAMPION_BEST.h5') 
print("Model loaded successfully!")


# --- 2. DEFINE YOUR CLASS NAMES ---
# IMPORTANT: This order MUST match your Kaggle notebook's
# 'train_generator.class_indices'
CLASS_NAMES = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TUBERCULOSIS']
IMG_SIZE = (224, 224)

# --- 3. DEFINE THE "BRAIN" FUNCTION ---
# This is the function that runs when you click "Analyze"
# It takes the uploaded image and returns the results.
def predict(input_image):
    
    # 1. Pre-process the image
    # We use ImageOps.fit to resize and crop to 224x224
    img = ImageOps.fit(input_image, IMG_SIZE, Image.Resampling.LANCZOS)
    img_array = np.array(img)
    
    # Ensure image is 3-channel (RGB)
    if img_array.ndim == 2: # Handle grayscale images
        img_array = np.stack((img_array,)*3, axis=-1)
    
    # Normalize and create a batch
    img_array = img_array / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    
    # 2. Make Prediction
    predictions = model.predict(img_batch)[0] # Get the [0.1, 0.05, 0.8, 0.05] array
    
    # 3. Format the Output
    # This turns the array into a dictionary:
    # {'COVID19': 0.1, 'NORMAL': 0.05, ...}
    confidences = {CLASS_NAMES[i]: float(predictions[i]) for i in range(len(CLASS_NAMES))}
    
    return confidences

# --- 4. BUILD THE WEB INTERFACE (GRADIO) ---
# This is where Gradio builds your UI for you.
title = "Infectious Lung Disease AI Classifier"
description = (
    "A demo for my AI project. This model (VGG16 + Flatten/Dense + EarlyStopping) "
    "was trained on a Kaggle dataset of Chest X-Rays to identify COVID-19, "
    "Normal, Pneumonia, or Tuberculosis."
)

gr.Interface(
    fn=predict,  # This is the "brain" function to call
    
    inputs=gr.Image(type="pil", label="Upload a Chest X-Ray"), # This creates the file uploader
    
    outputs=gr.Label(num_top_classes=4, label="Diagnosis Results"), # This creates the beautiful results box
    
    title=title,
    description=description,
    examples=[
        # You can add example image paths here if you add them to your repo
        # "example_covid.jpeg",
        # "example_pneumonia.jpeg"
    ]
).launch()

