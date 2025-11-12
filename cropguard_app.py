import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import gdown
import os

st.set_page_config(
    page_title="CropGuard AI - Plant Disease Detection",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; background-color: #f0f0f0; color: #333; border: 1px solid #ddd; border-radius: 6px; padding: 10px 20px; font-size: 14px; margin-top: 10px; }
    .stButton>button:hover { background-color: #e0e0e0; border-color: #bbb; }
    .analyze-button>button { background-color: #ff6b6b !important; color: white !important; font-weight: 500; }
    .analyze-button>button:hover { background-color: #ff5252 !important; }
    .header-title { text-align: center; color: #2d6a4f; font-size: 48px; font-weight: bold; margin-bottom: 10px; }
    .upload-box { background-color: #e8f5e9; border: 2px dashed #95d5b2; border-radius: 10px; padding: 40px; text-align: center; margin: 20px 0; }
    .diagnosis-box { background-color: #d1f4d1; border-left: 4px solid #52b788; padding: 20px; border-radius: 8px; margin: 20px 0; }
    .remedy-box { background-color: #d4e9f7; border-left: 4px solid #4a90e2; padding: 20px; border-radius: 8px; margin: 20px 0; }
    .fertilizer-box { background-color: #fff4d1; border-left: 4px solid #f4a261; padding: 20px; border-radius: 8px; margin: 20px 0; }
    .confidence-text { font-size: 14px; color: #555; margin-top: 5px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="header-title">🌿 CropGuard AI 🌿</div>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_path = 'plant_disease_prediction_model.h5'
    if not os.path.exists(model_path):
        st.info('📥 Downloading model... (~500MB)')
        progress_bar = st.progress(0)
        try:
            file_id = '1znEHh0QFjRQp_CihCu5CHQZKmCLwYU_p'
            url = f'https://drive.google.com/uc?id={file_id}&export=download&confirm=t'
            output = gdown.download(url, model_path, quiet=False)
            progress_bar.progress(100)
            if output:
                st.success('✅ Model downloaded!')
            else:
                st.error('❌ Download failed')
                return None
        except Exception as e:
            st.error(f'Error: {str(e)}')
            return None
    try:
        model = keras.models.load_model(model_path, compile=False)
        st.success('✅ Model loaded!')
        return model
    except Exception as e:
        st.error(f'Error loading: {str(e)}')
        return None

CLASS_NAMES = {0:"Apple___Apple_scab",1:"Apple___Black_rot",2:"Apple___Cedar_apple_rust",3:"Apple___healthy",4:"Blueberry___healthy",5:"Cherry_(including_sour)___Powdery_mildew",6:"Cherry_(including_sour)___healthy",7:"Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",8:"Corn_(maize)___Common_rust_",9:"Corn_(maize)___Northern_Leaf_Blight",10:"Corn_(maize)___healthy",11:"Grape___Black_rot",12:"Grape___Esca_(Black_Measles)",13:"Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",14:"Grape___healthy",15:"Orange___Haunglongbing_(Citrus_greening)",16:"Peach___Bacterial_spot",17:"Peach___healthy",18:"Pepper,_bell___Bacterial_spot",19:"Pepper,_bell___healthy",20:"Potato___Early_blight",21:"Potato___Late_blight",22:"Potato___healthy",23:"Raspberry___healthy",24:"Soybean___healthy",25:"Squash___Powdery_mildew",26:"Strawberry___Leaf_scorch",27:"Strawberry___healthy",28:"Tomato___Bacterial_spot",29:"Tomato___Early_blight",30:"Tomato___Late_blight",31:"Tomato___Leaf_Mold",32:"Tomato___Septoria_leaf_spot",33:"Tomato___Spider_mites Two-spotted_spider_mite",34:"Tomato___Target_Spot",35:"Tomato___Tomato_Yellow_Leaf_Curl_Virus",36:"Tomato___Tomato_mosaic_virus",37:"Tomato___healthy"}

DISEASE_INFO = {
    "Tomato___Early_blight": {"remedy": ["Apply chlorothalonil fungicide", "Remove infected leaves", "Proper spacing"], "fertilizer": ["NPK 10-10-10", "Add compost", "Apply calcium"]},
    "Tomato___Late_blight": {"remedy": ["Apply copper fungicide", "Remove infected plants", "Avoid overhead water"], "fertilizer": ["NPK 5-10-10", "Increase potassium", "Reduce nitrogen"]},
    "Tomato___healthy": {"remedy": ["Continue monitoring", "Good practices", "Regular watering"], "fertilizer": ["NPK 10-10-10", "Switch to 5-10-10 when flowering", "Add compost"]},
    "Potato___Early_blight": {"remedy": ["Remove infected leaves", "Apply copper fungicide", "Proper spacing"], "fertilizer": ["NPK 10-10-10", "Add potassium", "Maintain moisture"]},
    "Potato___Late_blight": {"remedy": ["Apply metalaxyl", "Remove infected material", "Better circulation"], "fertilizer": ["NPK 5-10-10", "More potassium", "Less nitrogen"]},
    "Potato___healthy": {"remedy": ["Monitor pests", "Regular irrigation", "Remove weeds"], "fertilizer": ["NPK 10-10-10", "Add compost", "Apply sulfur if needed"]},
    "Apple___Apple_scab": {"remedy": ["Apply captan or sulfur", "Remove infected leaves", "Good air flow"], "fertilizer": ["NPK 10-10-10", "Add calcium", "Avoid excess nitrogen"]},
    "Apple___healthy": {"remedy": ["Keep monitoring", "Good practices", "Regular spraying"], "fertilizer": ["NPK 10-10-10", "Monthly feeding", "Spring compost"]},
    "Corn_(maize)___healthy": {"remedy": ["Watch for pests", "Proper spacing", "Water regularly"], "fertilizer": ["NPK 20-10-10", "Side dress nitrogen", "Add micronutrients"]},
    "Grape___healthy": {"remedy": ["Regular pruning", "Monitor pests", "Remove dead wood"], "fertilizer": ["NPK 10-10-10", "Organic matter", "Sulfur if needed"]}
}

DEFAULT_INFO = {"remedy": ["Consult expert", "Remove affected parts", "Proper spacing"], "fertilizer": ["Balanced NPK", "Organic matter", "Maintain pH"]}

def preprocess_image(image):
    try:
        img = image.resize((128, 128))
        img_array = np.array(img)
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]
        img_array = img_array.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        st.error(f"Preprocessing error: {str(e)}")
        return None

def predict_disease(model, image):
    try:
        processed_image = preprocess_image(image)
        if processed_image is None:
            raise Exception("Preprocessing failed")
        predictions = model.predict(processed_image, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class]) * 100
        disease_name = CLASS_NAMES.get(predicted_class, "Unknown")
        return disease_name, confidence, predictions[0]
    except Exception as e:
        raise Exception(f"Prediction failed: {str(e)}")

st.markdown("### Upload Leaf Image")

uploaded_file = st.file_uploader("Drag and drop", type=['png','jpg','jpeg','webp'], label_visibility="collapsed")

if uploaded_file:
    try:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns([1,1])
        with col1:
            st.image(image, caption="Uploaded", use_column_width=True)
        with col2:
            st.write(f"**File:** {uploaded_file.name}")
            st.write(f"**Size:** {uploaded_file.size/1024:.2f} KB")
            st.write(f"**Dimensions:** {image.size[0]} x {image.size[1]} px")
        
        st.markdown('<div class="analyze-button">', unsafe_allow_html=True)
        if st.button("🔬 Analyze Leaf", use_container_width=True):
            model = load_model()
            if model:
                with st.spinner("🔍 Analyzing..."):
                    try:
                        disease_name, confidence, all_predictions = predict_disease(model, image)
                        parts = disease_name.split("___")
                        plant_name = parts[0].replace("_", " ")
                        disease = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
                        
                        st.markdown(f"""<div class="diagnosis-box"><h3>✅ Diagnosis: {plant_name}</h3><p><strong>Disease:</strong> {disease}</p><p class="confidence-text">Confidence: {confidence:.2f}%</p></div>""", unsafe_allow_html=True)
                        
                        info = DISEASE_INFO.get(disease_name, DEFAULT_INFO)
                        st.markdown("""<div class="remedy-box"><h4>🌱 Remedy:</h4></div>""", unsafe_allow_html=True)
                        for r in info["remedy"]:
                            st.markdown(f"• {r}")
                        
                        st.markdown(f"""<div class="fertilizer-box"><h4>🌾 Fertilizer for {plant_name}:</h4></div>""", unsafe_allow_html=True)
                        for f in info["fertilizer"]:
                            st.markdown(f"• {f}")
                        
                        st.markdown("---\n### 📊 Top 3 Predictions:")
                        top_3 = np.argsort(all_predictions)[-3:][::-1]
                        for i, idx in enumerate(top_3, 1):
                            name = CLASS_NAMES.get(idx, "Unknown").replace("___", " - ").replace("_", " ")
                            conf = float(all_predictions[idx]) * 100
                            st.write(f"{i}. {name}: {conf:.2f}%")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        st.markdown('</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error: {str(e)}")
else:
    st.markdown("""<div class="upload-box"><h3>☁️ Drag and drop here</h3><p style="color:#666;font-size:14px;">Max 20MB • PNG, JPG, JPEG, WEBP</p></div>""", unsafe_allow_html=True)
    st.info("👆 Upload a leaf image!")

st.markdown("---")
st.markdown("""<div style="text-align:center;color:#666;font-size:12px;"><p>🌿 CropGuard AI</p><p style="font-size:11px;color:#999;">256x256 images • Google Drive enabled</p></div>""", unsafe_allow_html=True)

