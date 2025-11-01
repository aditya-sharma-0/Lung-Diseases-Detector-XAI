import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K
from PIL import Image, ImageOps
import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries
import warnings

# --- 0. Setup & Constants ---
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# Define your class names in the *exact* order your model was trained on
# You can check this by running `print(test_generator.class_indices)` in Kaggle
CLASS_NAMES = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TUBERCULOSIS']
IMG_SIZE = (224, 224)
num_classes = 4

# --- 1. DEFINE YOUR MODEL'S "SKELETON" ---
# This function builds the *exact same* architecture as your winning
# "Model 7" (VGG16 + Flatten/Deeper).
def build_champion_model():
    print("Building model skeleton...")
    
    base_model_vgg16 = VGG16(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights='imagenet' # Gradio will download this once
    )
    base_model_vgg16.trainable = False
    
    model = Sequential([
        base_model_vgg16,               # The "Eyes"
        Flatten(),                      # The "Neck" that works (25,088 features)
        Dense(256, activation='relu'),  # "Thinking" Layer 1
        Dropout(0.5),                   
        Dense(128, activation='relu'),  # "Thinking" Layer 2
        Dropout(0.3),                   
        Dense(num_classes, activation='softmax') # The "Decision"
    ], name="Lung_Disease_Classifier")
    
    print("Model skeleton built.")
    return model

# --- 2. LOAD YOUR CHAMPION "WEIGHTS" ---
print("Building model and loading champion weights...")
model = build_champion_model()
weights_file = 'model_FINAL_BEST_weights.h5' 
model.load_weights(weights_file)
print("Model weights loaded successfully!")

# --- 3. DEFINE THE AI "BRAIN" FUNCTIONS ---

# Helper function to preprocess the user's image
def preprocess_image(pil_image):
    img = ImageOps.fit(pil_image, IMG_SIZE, Image.Resampling.LANCZOS)
    
    # Ensure image is 3-channel (RGB)
    if img.mode == 'L' or img.mode == 'RGBA':
        img = img.convert('RGB')
        
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch, img_array # Return both batch and simple array

# Helper function for LIME to understand our model
def model_predict_for_lime(images_for_lime):
    # LIME gives us a batch of (num_samples, 224, 224, 3)
    # Our model is already trained on 0-1 range, so no /255.0 needed here
    return model.predict(images_for_lime)

# --- 4. DEFINE THE MAIN GRADIO FUNCTION (Predict + Explain) ---
def predict_and_explain(input_image_pil):
    print(f"Received image, starting analysis...")
    
    # 1. Get Predictions
    processed_batch, processed_array = preprocess_image(input_image_pil)
    predictions = model.predict(processed_batch)[0]
    
    # Format as Gradio-friendly {Label: Confidence} dictionary
    confidences = {CLASS_NAMES[i]: float(predictions[i]) for i in range(len(CLASS_NAMES))}
    
    # 2. Get LIME Explanation (The XAI part)
    print("Starting LIME explanation...")
    explainer = lime_image.LimeImageExplainer()
    
    # Run the LIME "probe"
    # This is the step that "wiggles" the image
    explanation = explainer.explain_instance(
        image=processed_array.astype('double'), 
        classifier_fn=model_predict_for_lime,
        top_labels=1, # Only explain the top prediction
        hide_color=0,
        num_samples=1000 # Number of "wiggles"
    )
    
    # Get the top prediction's index (e.g., 0 for COVID, 1 for NORMAL)
    top_prediction_index = predictions.argmax()
    
    # Get the heatmap (mask) for the top prediction
    # This highlights the pixels that mattered *most*
    img_boundry, mask = explanation.get_image_and_mask(
        label=top_prediction_index,
        positive_only=True, 
        num_features=5, # Show top 5 most important areas
        hide_rest=False
    )
    
    # Combine the heatmap with the original image
    # We use mark_boundaries to draw the green areas
    heatmap_image = mark_boundaries(img_boundry, mask)
    
    print("Analysis complete.")
    
    # Return both results to the Gradio interface
    return confidences, heatmap_image

# --- 5. BUILD THE WEB INTERFACE (GRADIO) ---
title = "Smart Lung Disease Detector (VGG16 + XAI)"
description = (
    "A demo for my AI project. This model (VGG16 + Flatten/Deeper Head) was trained on a Kaggle dataset of Chest X-Rays. "
    "It classifies 4 diseases (COVID-19, Normal, Pneumonia, Tuberculosis) and uses LIME (XAI) to "
    "show *why* it made its decision. This is the 'checkmate' project."
)

gr.Interface(
    fn=predict_and_explain,
    inputs=gr.Image(type="pil", label="Upload a Chest X-Ray"), 
    outputs=[
        gr.Label(num_top_classes=4, label="Diagnosis Results"),
        gr.Image(label="Explainability Heatmap (LIME)", type="pil")
    ],
    title=title,
    description=description,
    allow_flagging="never"
).launch(share=False)

