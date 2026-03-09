import streamlit as st
import cv2
import numpy as np
from PIL import Image
import joblib
from main import preprocess_and_align, extract_rois, get_fused_vector

st.set_page_config(page_title="Visual Forensics ID Auth", layout="wide")

st.title("🔍 Visual Forensics for Identity Document Authentication")
st.markdown("**(Fonts, Textures, Seals & Holograms)** - *Multi-Feature Analysis*")

uploaded_file = st.file_uploader("Upload an ID Document (Smartphone Camera or Scanner)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Original Image (Raw Capture)")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
        
    with st.spinner('Applying Perspective Correction & Feature Extraction...'):
        aligned_img = preprocess_and_align(image)
        rois = extract_rois(aligned_img)
        features = get_fused_vector(rois)
        
        # Load Model
        try:
            model = joblib.load("models/authenticity_model.pkl")
            prediction = model.predict([features])[0]
            confidence = model.predict_proba([features])[0].max() * 100
        except:
            st.error("Model not trained yet. Please run train.py first.")
            st.stop()

    with col2:
        st.subheader("2. Pre-processed & Aligned")
        st.image(cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB), use_container_width=True)

    st.markdown("---")
    st.subheader("3. Multi-Feature ROI Extraction (Explainability)")
    
    roi1, roi2, roi3 = st.columns(3)
    with roi1:
        st.image(cv2.cvtColor(rois["text_region"], cv2.COLOR_BGR2RGB), caption="Font & Typography (LBP Analysis)")
    with roi2:
        st.image(cv2.cvtColor(rois["background_texture"], cv2.COLOR_BGR2RGB), caption="Micro-Textures (LBP Analysis)")
    with roi3:
        st.image(cv2.cvtColor(rois["seal_hologram"], cv2.COLOR_BGR2RGB), caption="Seals & Holograms (CNN ResNet18)")

    st.markdown("---")
    st.subheader("4. Final Authentication Decision")
    
    if prediction == 1:
        st.success(f"✅ GENUINE DOCUMENT DETECTED (Confidence: {confidence:.2f}%)")
    else:
        st.error(f"🚨 COUNTERFEIT / FORGERY DETECTED (Confidence: {confidence:.2f}%)")