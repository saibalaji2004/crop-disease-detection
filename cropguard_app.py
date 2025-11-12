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
    .header-title { text-align: center; color: #2d6a4f; font-size: 48px; font-weight: bold; margin-bottom: 10px; display: flex; align-items: center; justify-content: center; gap: 10px; }
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
        st.info('📥 Downloading model from Google Drive... (first time only, ~500MB)')
        progress_bar = st.progress(0)
        try:
            file_id = '1znEHh0QFjRQp_CihCu5CHQZKmCLwYU_p'
            url = f'https://drive.google.com/uc?id={file_id}&export=download&confirm=t'
            output = gdown.download(url, model_path, quiet=False)
            progress_bar.progress(100)
            if output:
                st.success('✅ Model downloaded successfully!')
            else:
                st.error('❌ Download failed.')
                return None
        except Exception as e:
            st.error(f'❌ Error: {str(e)}')
            return None
    try:
        model = keras.models.load_model(model_path, compile=False)
        st.success('✅ Model loaded successfully!')
        return model
    except Exception as e:
        st.error(f'❌ Error loading model: {str(e)}')
        return None

CLASS_NAMES = {0:"Apple___Apple_scab",1:"Apple___Black_rot",2:"Apple___Cedar_apple_rust",3:"Apple___healthy",4:"Blueberry___healthy",5:"Cherry_(including_sour)___Powdery_mildew",6:"Cherry_(including_sour)___healthy",7:"Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",8:"Corn_(maize)___Common_rust_",9:"Corn_(maize)___Northern_Leaf_Blight",10:"Corn_(maize)___healthy",11:"Grape___Black_rot",12:"Grape___Esca_(Black_Measles)",13:"Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",14:"Grape___healthy",15:"Orange___Haunglongbing_(Citrus_greening)",16:"Peach___Bacterial_spot",17:"Peach___healthy",18:"Pepper,_bell___Bacterial_spot",19:"Pepper,_bell___healthy",20:"Potato___Early_blight",21:"Potato___Late_blight",22:"Potato___healthy",23:"Raspberry___healthy",24:"Soybean___healthy",25:"Squash___Powdery_mildew",26:"Strawberry___Leaf_scorch",27:"Strawberry___healthy",28:"Tomato___Bacterial_spot",29:"Tomato___Early_blight",30:"Tomato___Late_blight",31:"Tomato___Leaf_Mold",32:"Tomato___Septoria_leaf_spot",33:"Tomato___Spider_mites Two-spotted_spider_mite",34:"Tomato___Target_Spot",35:"Tomato___Tomato_Yellow_Leaf_Curl_Virus",36:"Tomato___Tomato_mosaic_virus",37:"Tomato___healthy"}

DISEASE_INFO = {
    "Tomato___Early_blight": {"remedy": ["Apply fungicides containing chlorothalonil", "Remove infected leaves", "Ensure proper spacing"], "fertilizer": ["Use balanced NPK 10-10-10", "Add organic compost", "Apply calcium"]},
    "Tomato___Late_blight": {"remedy": ["Apply copper-based fungicides", "Remove infected plants", "Avoid overhead watering"], "fertilizer": ["Use NPK 5-10-10", "Increase potassium", "Avoid excessive nitrogen"]},
    "Tomato___healthy": {"remedy": ["Continue monitoring", "Maintain good practices", "Keep watering schedule"], "fertilizer": ["Use NPK 10-10-10", "Switch to 5-10-10 during flowering", "Add compost"]},
    "Potato___Early_blight": {"remedy": ["Remove infected leaves", "Apply copper fungicides", "Proper spacing"], "fertilizer": ["Use NPK 10-10-10", "Add potassium", "Maintain moisture"]},
    "Potato___Late_blight": {"remedy": ["Apply metalaxyl fungicides", "Remove infected material", "Improve circulation"], "fertilizer": ["Use NPK 5-10-10", "Increase potassium", "Reduce nitrogen"]},
    "Potato___healthy": {"remedy": ["Monitor for pests", "Maintain irrigation", "Remove weeds"], "fertilizer": ["Use NPK 10-10-10", "Add compost", "Apply sulfur if needed"]},
    "Apple___Apple_scab": {"remedy": ["Apply captan or sulfur fungicides", "Remove infected leaves", "Good air circulation"], "fertilizer": ["Use balanced NPK 10-10-10", "Add calcium", "Avoid excess nitrogen"]},
    "Apple___healthy": {"remedy": ["Continue monitoring", "Maintain practices", "Regular spray schedule"], "fertilizer": ["Use NPK 10-10-10", "Apply monthly", "Add compost in spring"]},
    "Corn_(maize)___healthy": {"remedy": ["Monitor for pests", "Maintain spacing", "Water regularly"], "fertilizer": ["Use NPK 20-10-10", "Side dress with nitrogen", "Add micronutrients"]},
    "Grape___healthy": {"remedy": ["Prune regularly", "Monitor pests", "Remove dead wood"], "fertilizer": ["Use NPK 10-10-10", "Add organic matter", "Apply sulfur if needed"]}
}

DEFAULT_INFO = {"remedy": ["Consult agricultural extension", "Remove affected parts", "Proper spacing"], "fertilizer": ["Balanced NPK fertilizer", "Add organic matter", "Maintain pH"]}

def preprocess_image(image):
    try:
        target_size = 256
        img = image.resize((target_size, target_size))
        img_array = np.array(img)
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]
        img_array = img_array.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def predict_disease(model, image):
    try:
        processed_image = preprocess_image(image)
        if processed_image is None:
            raise Exception("Preprocessing failed")
        predictions = model.predict(processed_image, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class]) * 100
        disease_name = CLASS_NAMES.get(predicted_class, "Unknown Disease")
        return disease_name, confidence, predictions[0]
    except Exception as e:
        raise Exception(f"Prediction failed: {str(e)}")

st.markdown("### Upload Leaf Image")

uploaded_file = st.file_uploader("Drag and drop file here", type=['png', 'jpg', 'jpeg', 'webp'], help="Limit 20MB per file • PNG, JPG, JPEG, WEBP", label_visibility="collapsed")

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption="Uploaded Leaf Image", use_column_width=True)
        with col2:
            st.write(f"**Filename:** {uploaded_file.name}")
            st.write(f"**Size:** {uploaded_file.size / 1024:.2f} KB")
            st.write(f"**Dimensions:** {image.size[0]} x {image.size[1]} px")
        st.markdown('<div class="analyze-button">', unsafe_allow_html=True)
        analyze_button = st.button("🔬 Analyze Leaf", use_container_width=True, key="analyze_button")
        st.markdown('</div>', unsafe_allow_html=True)
        if analyze_button:
            model = load_model()
            if model is not None:
                with st.spinner("🔍 Analyzing..."):
                    try:
                        disease_name, confidence, all_predictions = predict_disease(model, image)
                        parts = disease_name.split("___")
                        plant_name = parts[0].replace("_", " ")
                        disease = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
                        st.markdown(f"""<div class="diagnosis-box"><h3>✅ Diagnosis: {plant_name}</h3><p><strong>Disease/Condition:</strong> {disease}</p><p class="confidence-text">Confidence: {confidence:.2f}%</p></div>""", unsafe_allow_html=True)
                        info = DISEASE_INFO.get(disease_name, DEFAULT_INFO)
                        st.markdown(f"""<div class="remedy-box"><h4>🌱 Suggested Remedy:</h4></div>""", unsafe_allow_html=True)
                        for remedy in info["remedy"]:
                            st.markdown(f"• {remedy}")
                        st.markdown(f"""<div class="fertilizer-box"><h4>🌾 Fertilizer Recommendations for {plant_name}:</h4></div>""", unsafe_allow_html=True)
                        for fertilizer in info["fertilizer"]:
                            st.markdown(f"• {fertilizer}")
                        st.markdown("---")
                        st.markdown("### 📊 Other Possible Predictions:")
                        top_3_indices = np.argsort(all_predictions)[-3:][::-1]
                        for idx, class_idx in enumerate(top_3_indices, 1):
                            pred_name = CLASS_NAMES.get(class_idx, "Unknown")
                            pred_confidence = float(all_predictions[class_idx]) * 100
                            pred_name_display = pred_name.replace("___", " - ").replace("_", " ")
                            st.write(f"{idx}. {pred_name_display}: {pred_confidence:.2f}%")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.info("💡 Try a different image or check model compatibility")
            else:
                st.error("❌ Model could not be loaded")
    except Exception as e:
        st.error(f"Error: {str(e)}")
else:
    st.markdown("""<div class="upload-box"><h3>☁️ Drag and drop file here</h3><p style="color: #666; font-size: 14px;">Limit 20MB per file • PNG, JPG, JPEG, WEBP</p></div>""", unsafe_allow_html=True)
    st.info("👆 Upload a leaf image to get started!")

st.markdown("---")
st.markdown("""<div style="text-align: center; color: #666; font-size: 12px;"><p>🌿 CropGuard AI - Protecting your crops with artificial intelligence</p><p style="font-size: 11px; color: #999;">Model expects 256x256 pixel images • Google Drive integration enabled</p></div>""", unsafe_allow_html=True)
