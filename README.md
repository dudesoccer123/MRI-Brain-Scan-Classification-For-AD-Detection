# MRI-Brain-Scan-Classification-For-AD-Detection
An AI-powered web application for classifying Alzheimer's Disease from brain scan images using deep learning.

## Features
- Upload NIfTI (.nii) brain scan files
- AI-powered classification into 3 categories:
  - AD (Alzheimer's Disease)
  - CN (Cognitively Normal) 
  - MCI (Mild Cognitive Impairment)
- Confidence scores and detailed analysis
- Interactive visualizations

## How to Run the App

### 1. Clone the Repository

```bash
git clone https://github.com/dudesoccer123/MRI-Brain-Scan-Classification-For-AD-Detection.git
cd MRI-Brain-Scan-Classification-For-AD-Detection
```
### 2. Build and Run Using Docker Compose
```
docker-compose up --build
```
### 3. Use the App
- Open your browser and go to http://localhost:8501
- Upload a .nii brain scan file (MRI)
- Click "Analyze Brain Scan" to get results

## Medical Disclaimer
⚠️ This tool is for research purposes only and should not be used as a substitute for professional medical diagnosis.

## Model
Uses EfficientNetB2 with Squeeze-and-Excitation Attention blocks, trained on brain imaging data.


