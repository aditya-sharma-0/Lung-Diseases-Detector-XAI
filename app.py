import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd


print("Loading model... This will run only once.")


model = load_model('model_FINAL_BEST.h5') 

print("Model loaded successfully!")



CLASS_NAMES = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TUBERCULOSIS']
IMG_SIZE = (224, 224)


def predict(input_image):
    
 
    img = ImageOps.fit(input_image, IMG_SIZE, Image.Resampling.LANCZOS)
    img_array = np.array(img)
    

    if img_array.ndim == 2: 
        img_array = np.stack((img_array,)*3, axis=-1)
    

    img_array = img_array / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    

    predictions = model.predict(img_batch)[0] 
    

    confidences = {CLASS_NAMES[i]: float(predictions[i]) for i in range(len(CLASS_NAMES))}
    
    return confidences


title = "Infectious Lung Disease AI Classifier"
description = (
    "A demo for my AI project. This model (VGG16 + Flatten/Dense + EarlyStopping) "
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

    ]
).launch()

