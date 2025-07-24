import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# Set page configuration FIRST
st.set_page_config(
    page_title="Alzheimer's Disease Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize with progress bar at start
if 'app_loaded' not in st.session_state:
    st.info("🚀 Initializing Alzheimer's Disease Classification System...")
    progress_bar = st.progress(0)
    
    progress_bar.progress(20)
    import numpy as np
    import nibabel as nib
    import torch
    import torch.nn as nn
    from torchvision import transforms, models
    progress_bar.progress(50)
    import tempfile
    import os
    from PIL import Image
    import matplotlib.pyplot as plt
    import seaborn as sns
    import gdown
    progress_bar.progress(100)
    
    st.session_state['app_loaded'] = True
    progress_bar.empty()
    st.rerun()
else:
    import numpy as np
    import nibabel as nib
    import torch
    import torch.nn as nn
    from torchvision import transforms, models
    import tempfile
    import os
    from PIL import Image
    import matplotlib.pyplot as plt
    import seaborn as sns
    import gdown


st.sidebar.header("📋 Instructions")
st.sidebar.markdown("""
1. **Upload NIfTI File**: Upload a .nii brain scan file
2. **Get Classification**: View the AI-powered diagnosis
3. **Review Results**: Analyze confidence scores
""")

st.sidebar.markdown("---")
st.sidebar.header("⚠️ Medical Disclaimer")
st.sidebar.warning("""
This tool is for research purposes only and should not be used as a substitute for professional medical diagnosis.
""")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CONFIG = {
    'image_size': 224,
    'num_slices': 16,
    'num_classes': 3,
}

GDRIVE_FILE_ID = "1rnTb3iSnPSGykdj7xRSKa7tRaqXG3KGB"

class_names = ['AD (Alzheimer\'s Disease)', 'CN (Cognitively Normal)', 'MCI (Mild Cognitive Impairment)']
class_descriptions = {
    'AD (Alzheimer\'s Disease)': 'A progressive neurodegenerative disorder characterized by memory loss and cognitive decline.',
    'CN (Cognitively Normal)': 'Normal cognitive function with no signs of cognitive impairment.',
    'MCI (Mild Cognitive Impairment)': 'Mild cognitive changes that are greater than normal aging but not severe enough for dementia diagnosis.'
}

# Model Architecture 
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class EfficientNetB2WithSE(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        
        # Load pre-trained EfficientNetB2
        try:
            from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
            self.backbone = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
        except ImportError:
            self.backbone = models.efficientnet_b2(pretrained=True)
        
        # Get features
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        # SE block
        self.se_block = SEBlock(num_features, reduction=8)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # Extract features
        features = self.backbone.features(x)
        
        # Global pooling
        features = nn.AdaptiveAvgPool2d(1)(features)
        
        # SE attention
        features = self.se_block(features)
        
        # Flatten and classify
        features = features.flatten(1)
        return self.classifier(features)

@st.cache_resource
def load_model():
    """Load model from Google Drive"""
    model_path = "alzheimer_model.pth"
    
    if not os.path.exists(model_path):
        with st.spinner("Downloading model from Google Drive..."):
            url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
            gdown.download(url, model_path, quiet=False)
    
    with st.spinner("Loading model..."):
        model = EfficientNetB2WithSE(CONFIG['num_classes'])
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
    return model

def extract_key_slices(img_data, num_slices=16):
    """Extract key slices from 3D volume"""
    total_slices = img_data.shape[2]
    
    # Same slice selection as training
    start_slice = int(total_slices * 0.15)
    end_slice = int(total_slices * 0.85)
    
    slice_indices = np.linspace(start_slice, end_slice-1, num_slices, dtype=int)
    return slice_indices

def preprocess_slice(slice_2d):
    """Preprocess a single slice (same as training)"""
    
    p2, p98 = np.percentile(slice_2d, (2, 98))
    slice_2d = np.clip(slice_2d, p2, p98)
    slice_2d = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min() + 1e-8)
    
    
    slice_2d = np.stack([slice_2d, slice_2d, slice_2d], axis=2)
    slice_2d = (slice_2d * 255).astype(np.uint8)
    
    return slice_2d

def classify_nii_file(model, nii_file_path):
    """Classify a .nii file and return predictions with progress tracking"""
    img = nib.load(nii_file_path)
    img_data = img.get_fdata()
    slice_indices = extract_key_slices(img_data, CONFIG['num_slices'])
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    slice_predictions = []
    processed_slices = []
    
    with torch.no_grad():
        for i, slice_idx in enumerate(slice_indices):
            if slice_idx < img_data.shape[2]:
                slice_2d = img_data[:, :, slice_idx]
            else:
                slice_2d = img_data[:, :, img_data.shape[2]//2]
            
            # Preprocess slice
            processed_slice = preprocess_slice(slice_2d)
            processed_slices.append(processed_slice)
            
            # Convert to tensor
            slice_tensor = transform(processed_slice).unsqueeze(0).to(device)
            
            # Get prediction
            output = model(slice_tensor)
            prob = torch.softmax(output, dim=1).cpu().numpy()[0]
            slice_predictions.append(prob)
    
    
    avg_predictions = np.mean(slice_predictions, axis=0)
    predicted_class = np.argmax(avg_predictions)
    confidence = avg_predictions[predicted_class]
    
    return {
        'predicted_class': predicted_class,
        'class_name': class_names[predicted_class],
        'confidence': confidence,
        'probabilities': avg_predictions,
        'slice_predictions': slice_predictions,
        'processed_slices': processed_slices,
        'total_slices_analyzed': len(slice_indices)
    }

@st.cache_data
def create_visualization(results):
    """Create visualizations for the results"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Probability distribution
    axes[0, 0].bar(range(len(class_names)), results['probabilities'])
    axes[0, 0].set_title('Classification Probabilities')
    axes[0, 0].set_xlabel('Classes')
    axes[0, 0].set_ylabel('Probability')
    axes[0, 0].set_xticks(range(len(class_names)))
    axes[0, 0].set_xticklabels([name.split('(')[0].strip() for name in class_names], rotation=45)
    
    # 2. Confidence heatmap
    conf_matrix = np.array(results['probabilities']).reshape(1, -1)
    sns.heatmap(conf_matrix, annot=True, fmt='.3f', cmap='Blues', 
                xticklabels=[name.split('(')[0].strip() for name in class_names],
                yticklabels=['Confidence'], ax=axes[0, 1])
    axes[0, 1].set_title('Confidence Heatmap')
    
    # 3. Sample processed slices
    if len(results['processed_slices']) >= 4:
        for i in range(4):
            slice_img = results['processed_slices'][i * len(results['processed_slices']) // 4]
            axes[1, i//2 if i < 2 else 1].imshow(slice_img[:, :, 0], cmap='gray')
            axes[1, i//2 if i < 2 else 1].set_title(f'Slice {i+1}')
            axes[1, i//2 if i < 2 else 1].axis('off')
    
    plt.tight_layout()
    return fig

def prepare_classification_data(results):
    """Prepare classification data for LangChain integration"""
    classification_data = {
        'diagnosis': results['class_name'],
        'confidence_score': f"{results['confidence']:.1%}",
        'confidence_level': 'High' if results['confidence'] > 0.8 else 'Medium' if results['confidence'] > 0.6 else 'Low',
        'all_probabilities': {
            'AD': f"{results['probabilities'][0]:.1%}",
            'CN': f"{results['probabilities'][1]:.1%}",
            'MCI': f"{results['probabilities'][2]:.1%}"
        },
        'total_slices_analyzed': results['total_slices_analyzed'],
        'description': class_descriptions[results['class_name']]
    }
    return classification_data

# Main App
def main():
    st.title("🧠 Alzheimer's Disease Classification System")
    st.markdown("---")
    model = load_model()
    st.success("✅ Model loaded successfully!")
    st.header("📁 Upload Brain Scan")
    nii_file = st.file_uploader("Upload brain scan (.nii file)", type=['nii'])
    
    if nii_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nii') as tmp_file:
            tmp_file.write(nii_file.read())
            nii_path = tmp_file.name
        if st.button("🔍 Analyze Brain Scan", type="primary"):
            with st.spinner("Analyzing brain scan..."):
                results = classify_nii_file(model, nii_path)
            st.success("✅ Analysis Complete!")
            st.markdown("---")
            st.header("📊 Classification Results")
            col_result1, col_result2, col_result3 = st.columns(3)
            with col_result1:
                st.metric(
                    "Predicted Diagnosis",
                    results['class_name'].split('(')[0].strip(),
                    f"{results['confidence']:.1%} confidence"
                )
            with col_result2:
                st.metric(
                    "Confidence Level",
                    'High' if results['confidence'] > 0.8 else 'Medium' if results['confidence'] > 0.6 else 'Low',
                    f"{results['confidence']:.3f}"
                )
            
            with col_result3:
                st.metric(
                    "Slices Analyzed",
                    results['total_slices_analyzed'],
                    "brain regions"
                )
            st.subheader("🎯 Detailed Probabilities")
            prob_df = {
                'Diagnosis': [name.split('(')[0].strip() for name in class_names],
                'Probability': [f"{prob:.1%}" for prob in results['probabilities']],
                'Description': list(class_descriptions.values())
            }
            st.dataframe(prob_df, use_container_width=True)
            st.subheader("📈 Analysis Visualization")
            fig = create_visualization(results)
            st.pyplot(fig, use_container_width=True)
            classification_data = prepare_classification_data(results)
            st.session_state['classification_results'] = classification_data
        os.unlink(nii_path)

if __name__ == "__main__":
    main()