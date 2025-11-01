## **title: Lung Disease XAI Classifier emoji: 🫁🔬 colorFrom: blue colorTo: green sdk: gradio sdk\_version: "3.48.0" \# Use a specific, stable version app\_file: app.py pinned: false**

# **Smart Lung Disease Detector (VGG16 \+ XAI)**

This is the repository for my AI semester project, a deep learning application designed to classify chest X-ray images and provide trustworthy, explainable results.  
This is *not* just a classifier; it's an **Explainable AI (XAI)** tool that shows *why* it makes a decision by generating a heatmap of the most important areas on the X-ray.

### **Live Demo**

[You can find the live, cloud-hosted application here on Hugging Face Spaces.](https://huggingface.co/spaces/aditya-sharma-0/Lung-Diseases-Detector)

## **The Model: "Model 7 \- The Optimized Champion"**

This project involved 7 experiments to find the best, most robust model. The winning architecture is a hybrid "Transfer Learning" model.

* **Eyes (Base):** VGG16 (Frozen, 14.7M params).  
* **Neck (Feature Extractor):** Flatten() (Proven to be better than GlobalAveragePooling2D for this problem, as it retains spatial information).  
* **Brain (Head):** A *deeper, regularized* Dense head (Dense(256) \-\> Dropout(0.5) \-\> Dense(128) \-\> Dropout(0.3) \-\> Dense(4)).  
* **Final Accuracy:** 90.92% (Trained with EarlyStopping and ModelCheckpoint to guarantee the *best* version of the model was saved).

## **The "Checkmate" Feature: Explainable AI (XAI)**

Any model can provide a number. A *research-level* model must provide **proof**.  
This app uses the **LIME** (lime\_image) library to answer the question: "Why did the AI predict 'Pneumonia'?"

1. The user uploads an X-ray.  
2. The app returns the 4-class diagnosis (e.g., PNEUMONIA: 90.92%).  
3. It *also* returns a **heatmap image** that highlights the *exact pixels* in the lung that the AI used to make its decision. This makes the model trustworthy and interpretable.

### **Tech Stack**

* **Model:** TensorFlow / Keras (Python)  
* **XAI Library:** LIME  
* **Web App:** Gradio (Python)  
* **Hosting:** Hugging Face Spaces  
* **Dataset:** Kaggle "Chest X-Ray (Pneumonia,Covid-19,Tuberculosis)"  
* **Version Control:** Git \+ Git LFS (for large model file handling)