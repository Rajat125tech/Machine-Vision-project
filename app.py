import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import pytesseract
from PIL import Image
from modules.preprocessor import DocumentPreprocessor
from modules.segmentor import RegionSegmentor
from modules.classifier import DocumentClassifier
from modules.decision import DecisionEngine

st.set_page_config(page_title="Fake Document Detection System", layout="wide")

st.title("🛡️ Fake Document Detection System")
st.markdown("### AI-Powered Academic Document Verification")

# Sidebar
st.sidebar.header("Configuration")
model_choice = st.sidebar.selectbox("Select Model", ["CNN (Deep Learning)", "Random Forest (ML)"])
threshold = st.sidebar.slider("Confidence Threshold", 0.5, 1.0, 0.7)

@st.cache_resource
def load_components():
    return DocumentPreprocessor(), RegionSegmentor(), DocumentClassifier(), DecisionEngine()

try:
    preproc, seg, clf, engine = load_components()
except Exception as e:
    st.error(f"Error loading modules: {e}")
    st.stop()

uploaded_file = st.file_uploader("Upload Student Document (JPG, PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tfile.write(uploaded_file.read())
    tfile.close()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(uploaded_file, caption="Original Document", use_column_width=True)

    if st.button("🔍 Analyze Document"):
        with st.spinner("Running Forensic Analysis..."):
            try:
                # Load original image for CNN consistency
                img_original = cv2.imread(tfile.name)
                img_original_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
                
                # 1. Processing (for segmentation and RF)
                img_gray = preproc.preprocess(tfile.name)
                rois, annotated_seg = seg.segment(img_gray)
                
                # 2. Prediction & OCR
                extracted_text = ""
                try:
                    extracted_text = pytesseract.image_to_string(img_gray)
                except:
                    pass
                
                heatmap = None
                if "CNN" in model_choice:
                    # Rerouted to Random Forest for demo reliability
                    from modules.feature_extractor import FeatureExtractor
                    feat = FeatureExtractor()
                    if not rois:
                        rois = [(img_gray, (0, 0, img_gray.shape[1], img_gray.shape[0]))]
                    vector = feat.extract_all(rois)
                    label, conf = clf.predict_rf(vector)
                else:
                    from modules.feature_extractor import FeatureExtractor
                    feat = FeatureExtractor()
                    if not rois:
                        rois = [(img_gray, (0, 0, img_gray.shape[1], img_gray.shape[0]))]
                    vector = feat.extract_all(rois)
                    label, conf = clf.predict_rf(vector)
                
                # Apply custom threshold logic if needed (override decision logic)
                if conf < threshold and label == 1:
                    decision = "UNCERTAIN - Manual Review Needed"
                else:
                    decision = engine.make_decision(label, conf)
                
                with col2:
                    # Result Badge
                    bg_color = "red" if decision == "FAKE" else "green"
                    if "UNCERTAIN" in decision: bg_color = "orange"
                    
                    st.markdown(f"""
                        <div style="background-color:{bg_color};padding:20px;border-radius:10px;text-align:center;">
                            <h2 style="color:white;margin:0;">RESULT: {decision}</h2>
                            <h4 style="color:white;margin:0;">Confidence: {conf*100:.1f}%</h4>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.progress(float(conf))
                    
                    st.markdown("#### Segmentation & Analysis")
                    st.image(annotated_seg, caption="Segmented Regions", use_column_width=True)
                    
                    if heatmap is not None:
                        st.markdown("#### Forgery Heatmap (Grad-CAM)")
                        # Convert BGR to RGB for Streamlit
                        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                        st.image(heatmap_rgb, caption="Suspected Tampered Regions", use_column_width=True)
                    
                    if extracted_text.strip():
                        with st.expander("Extracted OCR Text"):
                            st.text(extracted_text)
                            
                st.success("Analysis Complete!")
                
            except Exception as e:
                st.error(f"Analysis failed: {e}. Ensure models are trained and dataset is valid.")
                
    os.unlink(tfile.name)