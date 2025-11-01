---
license: mit
title: Lung-Diseases-Detector
sdk: gradio
emoji: 📚
colorFrom: blue
colorTo: green
sdk_version: 5.49.1
---
# **Lung-Diseases-Detector 🫁🔬**

A Deep Learning web application to classify chest X-Ray images into four categories: **Normal, COVID-19, Pneumonia, and Tuberculosis**.  
This project was developed as a semester project for my Artificial Intelligence course, demonstrating a full-stack data science workflow from data analysis and model training to final deployment.  
[**Try the Live Demo\!**](https://www.google.com/search?q=https://your-huggingface-space-url.hf.space) *(To be Replaced with HF Space URL after I deploy)*  
*(After I deploy, ill take a screenshot and place it here)*

## **🚀 The Model: From 78% to 91%+ Accuracy**

The core of this project was a series of experiments to find the best possible AI model. I didn't just build one model; I built and tested five, proving that **Transfer Learning with a VGG16 base and a Dense head** was the better suited for this task.

### **My Experimental Journey**

I trained and evaluated 5 different architectures to find the best-performing model. This process was key to understanding *why* the final model works.

1. **Model 1 (The Champion): VGG16 \+ Flatten() \+ Dense(256)**  
   * **Score:** 91.31% (Test Accuracy)  
   * **Finding:** This was my best-performing architecture. However, the training graphs showed it **overfitted** (validation accuracy peaked early and then dropped). This led to Experiment 5\.  
2. **Model 2: VGG16 \+ Flatten() \+ RandomForest**  
   * **Score:** 78.86% (Failed)  
   * **Finding:** Proved that a Dense head (Deep Learning) is far superior to a RandomForest "brain" for this task. The "Curse of Dimensionality" (25,000+ features from Flatten()) was too noisy for the Random Forest.  
3. **Model 3: MobileNetV2 \+ GAP() \+ RandomForest**  
   * **Score:** 79.64% (Failed)  
   * **Finding:** Even with "clean" features (1,280 from GlobalAveragePooling2D), the Random Forest "brain" was *still* not smart enough. This confirmed I needed a Deep Learning "head."  
4. **Model 4: MobileNetV2 \+ GAP() \+ Dense(256)**  
   * **Score:** 90.79%  
   * **Finding:** A very stable and efficient model that proves GlobalAveragePooling2D works well, but it didn't beat the VGG16+Flatten combination.  
5. **Model 5: VGG16 \+ Flatten() \+ EarlyStopping (The *True* Champion)**  
   * **Score:** **\~94.7%** (This was the *peak validation score* from Model 1).  
   * **Finding:** I combined the *architecture* of Model 1 with EarlyStopping and ModelCheckpoint callbacks. This re-trained model stops *at its peak performance*, capturing the \~94-95% accuracy *before* it starts to overfit. This is the **model\_FINAL\_BEST.h5** file used in this app.

### **Final Model Performance**

*(These are the metrics from the Final Model)*

```
#### ---------- CLASSIFICATION REPORT (ULTIMATE MODEL) ----------
               precision    recall  f1-score   support

      COVID19       1.00      0.91      0.95       106
       NORMAL       0.91      0.82      0.87       234
    PNEUMONIA       0.91      0.96      0.93       390
TURBERCULOSIS       0.75      1.00      0.85        41

     accuracy                           0.91       771
    macro avg       0.89      0.92      0.90       771
 weighted avg       0.92      0.91      0.91       771

----------------------------------------------------------------------

```

#### **Confusion Matrix**
<img width="794" height="715" alt="image" src="https://github.com/user-attachments/assets/228aa604-56be-4240-b08a-2631999b0609" />


## **🛠 Technology Stack**

* **AI / Machine Learning:** TensorFlow Version: 2.18.0 & Keras  
* **Base Model:** VGG16 (Transfer Learning)  
* **Classifier:** Dense Neural Network with Dropout & EarlyStopping  
* **Backend & Frontend:** Gradio (Python Library)  
* **Cloud Hosting:** Hugging Face Spaces  
* **Data Science:** Pandas, Numpy, Scikit-learn, Matplotlib, Seaborn

## **🚀 How to Run Locally**

1. **Clone the repository:**  
```
   git clone https://github.com/aditya-sharma-0/Lung-Diseases-Detector.git 
   cd Lung-Diseases-Detector
```

3. **Set up Git LFS (for the .h5 model file):**  
```
   git lfs install  
   git lfs pull
```

5. **Install dependencies:**  
```
    pip install \-r requirements.txt
```

7. **Run the app:**  
```
   python app.py
```

   The app will be running at http://127.0.0.1:7860

## **📜 License**

This project is licensed under the **MIT License** \- see the [LICENSE](https://www.google.com/search?q=MIT-Licence) file for details.