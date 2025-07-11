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

- ## How to Run
1. Install requirements: `pip install -r requirements.txt`
2. Run the app: `streamlit run app.py`
3. Upload a .nii brain scan file for MRI
4. Click "Analyze Brain Scan" to get results

## Medical Disclaimer
⚠️ This tool is for research purposes only and should not be used as a substitute for professional medical diagnosis.

## Model
Uses EfficientNetB2 with Squeeze-and-Excitation Attention blocks, trained on brain imaging data.


