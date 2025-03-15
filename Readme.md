# Lung Cancer Detection using Deep Learning

## Introduction
The dataset is taken from kaggle "https://www.kaggle.com/datasets/paritosh921/adenocarcinoma-lungs-cancer-image-and-mask"
### Problem Statement
Lung cancer is one of the leading causes of cancer-related deaths worldwide, primarily due to late detection. Early and accurate detection significantly improves treatment outcomes. However, traditional diagnostic methods, such as CT scans and X-rays, require expert radiologists and can be time-consuming. Manual diagnosis is also prone to human error, leading to misdiagnosis or delays in treatment. This project aims to automate lung cancer detection using a deep learning-based UNet model embedded in a web application, allowing users to upload medical images and receive instant visualizations of detected cancerous regions.
Goals and Objectives
### Goals
- To develop a deep learning-based lung cancer detection system that can accurately identify cancerous regions in lung images.
- To create a user-friendly web application where users can upload medical images and get real-time predictions.
### Objectives
- Train a UNet model for lung cancer segmentation to highlight affected regions.
- Implement a web-based interface where users can upload images for analysis.
- Provide visual feedback by overlaying the predicted cancer regions on the uploaded image.
- Ensure high accuracy and reliability by validating the model’s performance against real-world data.
### Importance of the Project
- Early Detection Saves Lives – Faster and more accurate detection of lung cancer can significantly improve survival rates.
- Reduces Dependency on Experts – Assists radiologists by providing an automated second opinion, reducing workload and minimizing human error.
- Accessibility and Convenience – A web-based system allows easy access to lung cancer detection without requiring specialized medical software.
- Scalability for Large-Scale Screening – Can be integrated into hospitals and diagnostic centers for mass screening, helping detect lung cancer at an early stage.


This project bridges the gap between AI-driven automation and medical diagnostics, making lung cancer detection faster, more accessible, and more reliable.

---
## 2. Project Structure

```
my_project/
├── models/                  # Contains saved models and preprocessing objects
│   ├── segmentation_model_unet_final.keras   # Trained model file
├── static/                  # Static files (e.g., CSS, JavaScript)
│   └── styles.css           # Styling for the web interface
│   └── results          # folder storing predicted images
│   └── uploads         # folder storing uploaded images
├── templates/               # HTML templates for the web pages
│   └── index.html           # Main HTML template for user input and displaying results
├── app.py                   # Main Flask application file
├── environment.yml         # conda environment file 
├── Lung Cancer Detection using Deep Learning.ipynb      # jupyter notebook contain model building and preprocessing
└── README.md                # Documentation file describing the project
```
## 3. Installation
### 3.1 **Colne the repository**
    git clone <repository-url>
    cd Lung_Cancer_Detection_Using_DeepLearning
### 3.2 **Create the Conda Environment with the yml file**
    conda env create -f environment.yml -n new_env_name
### 3.3 **Activate new environment**
    conda activate new_env_name