<!-- ## **title: Lung Disease XAI Classifier emoji: 🫁🔬 colorFrom: blue colorTo: green sdk: gradio sdk\_version: "3.48.0" \# Use a specific, stable version app\_file: app.py pinned: false** -->

# **Smart Lung Disease Detector (VGG16 \+ XAI)**

This is the repository for my AI semester project, a deep learning application designed to classify chest X-ray images and provide trustworthy, explainable results.  
This is *not* just a classifier; it's an **Explainable AI (XAI)** tool that shows *why* it makes a decision by generating a heatmap of the most important areas on the X-ray.

### **Live Demo**

[You can find the live, cloud-hosted application here on Hugging Face Spaces.](https://aditya-sharma-0-lung-diseases-detector-xai.hf.space/)

<img width="1316" height="681" alt="image" src="https://github.com/user-attachments/assets/b1fa99da-50e7-498c-aba9-54e2cbd0e21f" />


## **The Model: "Model 7 \- The Optimized Model"**

This project involved 7 experiments to find the best, most robust model. The winning architecture is a hybrid "Transfer Learning" model.

* **Eyes (Base):** VGG16 (Frozen, 14.7M params).  
* **Neck (Feature Extractor):** Flatten() (Proven to be better than GlobalAveragePooling2D for this problem, as it retains spatial information).  
* **Brain (Head):** A *deeper, regularized* Dense head (Dense(256) \-\> Dropout(0.5) \-\> Dense(128) \-\> Dropout(0.3) \-\> Dense(4)).  
* **Final Accuracy:** 90.92% (Trained with EarlyStopping and ModelCheckpoint to guarantee the *best* version of the model was saved).

## **The Differentiating Feature: Explainable AI (XAI)**

Any model can provide a number. A *research-level* model must provide **proof**.  
This app uses the **LIME** (lime\_image) library to answer the question: "Why did the AI predict 'Pneumonia'?"

1. The user uploads an X-ray.  
2. The app returns the 4-class diagnosis (e.g., PNEUMONIA: 90.92%).  
3. It *also* returns a **heatmap image** that highlights the *exact pixels* in the lung that the AI used to make its decision. This makes the model trustworthy and interpretable.

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

<img width="794" height="715" alt="508553195-228aa604-56be-4240-b08a-2631999b0609" src="https://github.com/user-attachments/assets/c1c6774e-2304-4831-a742-af6931467c46" />


### **Tech Stack**

* **Model:** TensorFlow / Keras (Python)  
* **XAI Library:** LIME  
* **Web App:** Gradio (Python)  
* **Hosting:** Hugging Face Spaces  
* **Dataset:** Kaggle "Chest X-Ray (Pneumonia,Covid-19,Tuberculosis)"  
* **Version Control:** Git \+ Git LFS (for large model file handling)

---

## How to Run Locally

To run this application on your local machine, you'll need Python 3.9+ and Git installed.

**1. Install Git LFS (Critical for the Model File)**

This project uses Git LFS (Large File Storage) for the trained model file (`.h5`). You *must* install this *before* cloning.

On Windows: 

Download and run the installer from https://git-lfs.github.com/.

On Mac: 

```
brew install git-lfs
```

On Linux: 

```
sudo apt-get install git-lfs
```


After installing, you must run this command *once* in your terminal to initialize LFS:
```
git lfs install
```

**2. Clone the Repository**
This command will download the project and also use Git LFS to download the 500MB+ model file correctly.
```
git clone https://github.com/aditya-sharma-0/Lung-Diseases-Detector.git
cd Lung-Diseases-Detector
```

**3. Create a Virtual Environment (Best Practice)**
   
This creates a "bubble" for your project so the libraries don't mess up your main computer.

On Windows
```
python -m venv venv
.\venv\Scripts\activate
```
On Mac/Linux
```
python3 -m venv venv
source venv/bin/activate
```

**4. Install Dependencies**

This reads the requirements.txt file and installs all the necessary libraries.
```
pip install -r requirements.txt
```

**5. Run the Application**

This is the final step! This command starts the Gradio web server.
```
python app.py
```

6. Open in Your Browser

Your terminal will print a message like: **Running on local URL: http://127.0.0.1:7860**. Open that URL in your browser to use the app!
