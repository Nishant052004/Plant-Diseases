import numpy as np
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from disease_info import disease_info

# Page Configuration
st.set_page_config(
    page_title="AI Plant Disease Detection Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hardcoded Mappings for Slide 3 Alignment
CLASS_DISPLAY_NAMES = {
    "Pepper__bell___Bacterial_spot": "Pepper Bacterial Spot",
    "Pepper__bell___healthy": "Pepper Healthy",
    "Potato___Early_blight": "Potato Early Blight",
    "Potato___Late_blight": "Potato Late Blight",
    "Potato___healthy": "Potato Healthy",
    "Tomato_Bacterial_spot": "Tomato Bacterial Spot",
    "Tomato_Early_blight": "Tomato Early Blight",
    "Tomato_Late_blight": "Tomato Late Blight",
    "Tomato_Leaf_Mold": "Tomato Leaf Mold",
    "Tomato_Septoria_leaf_spot": "Tomato Septoria Spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Tomato Spider Mites",
    "Tomato__Target_Spot": "Tomato Target Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Tomato Yellow Leaf Curl",
    "Tomato__Tomato_mosaic_virus": "Tomato Mosaic Virus",
    "Tomato_healthy": "Tomato Healthy"
}

CLASS_CATEGORIES = {
    "Pepper__bell___Bacterial_spot": "Bacterial",
    "Pepper__bell___healthy": "Healthy",
    "Potato___Early_blight": "Fungal",
    "Potato___Late_blight": "Fungal",
    "Potato___healthy": "Healthy",
    "Tomato_Bacterial_spot": "Bacterial",
    "Tomato_Early_blight": "Fungal",
    "Tomato_Late_blight": "Fungal",
    "Tomato_Leaf_Mold": "Fungal",
    "Tomato_Septoria_leaf_spot": "Fungal",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Pest",
    "Tomato__Target_Spot": "Fungal",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Viral",
    "Tomato__Tomato_mosaic_virus": "Viral",
    "Tomato_healthy": "Healthy"
}

# Custom CSS for Slide Design Consistency
st.markdown("""
<style>
/* Font import */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

/* Global Style Modifications */
html, body, [data-testid="stAppViewContainer"], .stApp {
    background-color: #080C14 !important;
    font-family: 'Outfit', sans-serif !important;
    color: #E2E8F0 !important;
}

[data-testid="stHeader"] {
    background-color: transparent !important;
}

/* Hide streamlit headers/footers */
header, footer {
    visibility: hidden !important;
    height: 0px !important;
}

/* Custom styled badge */
.badge-header {
    background: rgba(0, 255, 153, 0.08);
    border: 1px solid #00FF99;
    color: #00FF99;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    display: inline-block;
    margin-bottom: 18px;
    text-transform: uppercase;
}

/* Title & Subtitle */
.main-title {
    font-size: 42px;
    font-weight: 700;
    color: #FFFFFF;
    margin-bottom: 5px;
    line-height: 1.2;
}

.main-title span {
    color: #00FF99;
}

.main-subtitle {
    font-size: 16px;
    color: #94A3B8;
    margin-bottom: 30px;
}

/* KPI Metrics cards */
.metric-row {
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    margin-bottom: 30px;
}

.metric-card {
    flex: 1;
    min-width: 200px;
    background: #111625;
    border: 1px solid #1E293B;
    border-radius: 16px;
    padding: 24px;
    text-align: left;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    transition: transform 0.3s ease, border-color 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-3px);
    border-color: #334155;
}

.metric-card .icon {
    font-size: 28px;
    margin-bottom: 12px;
}

.metric-card .value {
    font-size: 32px;
    font-weight: 700;
    margin-bottom: 4px;
}

.metric-card.classes .value { color: #00FF99; }
.metric-card.accuracy .value { color: #00B0FF; }
.metric-card.crops .value { color: #FFC107; }

.metric-card .label {
    font-size: 14px;
    color: #64748B;
    font-weight: 600;
}

/* Section cards */
.panel-card {
    background: #111625;
    border: 1px solid #1E293B;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
}

.panel-title {
    font-size: 20px;
    font-weight: 600;
    color: #FFFFFF;
    margin-bottom: 20px;
    border-bottom: 1px solid #1E293B;
    padding-bottom: 10px;
}

/* Crop badges */
.crop-badge-container {
    display: flex;
    justify-content: center;
    gap: 15px;
    margin-top: 20px;
}

.crop-badge {
    background: #0D111A;
    border: 1px solid #1E293B;
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 14px;
    font-weight: 600;
    color: #00FF99;
    border-color: #00FF99;
    background: rgba(0, 255, 153, 0.05);
}

/* Custom styled output headers */
.predicted-header {
    background: rgba(0, 255, 153, 0.05);
    border: 1px solid #00FF99;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 20px;
}

.predicted-label {
    font-size: 12px;
    color: #00FF99;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.predicted-value {
    font-size: 24px;
    font-weight: 700;
    color: #FFFFFF;
    margin-top: 5px;
}

.confidence-header {
    background: rgba(0, 176, 255, 0.05);
    border: 1px solid #00B0FF;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 20px;
}

.confidence-label {
    font-size: 12px;
    color: #00B0FF;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.confidence-value {
    font-size: 24px;
    font-weight: 700;
    color: #FFFFFF;
    margin-top: 5px;
}

/* Disease Info sections */
.info-section {
    background: #0D111A;
    border: 1px solid #1E293B;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
    display: flex;
    gap: 15px;
    align-items: flex-start;
}

.info-icon {
    font-size: 24px;
    min-width: 32px;
    height: 32px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.info-content h4 {
    margin: 0 0 5px 0;
    font-size: 15px;
    font-weight: 600;
    color: #FFFFFF;
}

.info-content p {
    margin: 0;
    font-size: 14px;
    color: #94A3B8;
    line-height: 1.5;
}

/* Badges for Disease Catalog classes */
.badge-pill {
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    display: inline-block;
    margin-bottom: 10px;
}

.badge-pill.fungal { background: rgba(255, 193, 7, 0.1); border: 1px solid #FFC107; color: #FFC107; }
.badge-pill.healthy { background: rgba(0, 255, 153, 0.1); border: 1px solid #00FF99; color: #00FF99; }
.badge-pill.bacterial { background: rgba(239, 68, 68, 0.1); border: 1px solid #EF4444; color: #EF4444; }
.badge-pill.viral { background: rgba(59, 130, 246, 0.1); border: 1px solid #3B82F6; color: #3B82F6; }
.badge-pill.pest { background: rgba(168, 85, 247, 0.1); border: 1px solid #A855F7; color: #A855F7; }

/* Grid Layout for Catalog */
.catalog-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 16px;
    margin-top: 20px;
}

.catalog-card {
    background: #111625;
    border: 1px solid #1E293B;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

.catalog-card h5 {
    margin: 5px 0 0 0;
    font-size: 14px;
    font-weight: 600;
    color: #FFFFFF;
}

/* Timeline/Pipeline layout */
.pipeline-container {
    display: flex;
    flex-direction: column;
    gap: 16px;
}

.pipeline-step {
    background: #111625;
    border: 1px solid #1E293B;
    border-radius: 12px;
    padding: 18px;
    display: flex;
    align-items: center;
    gap: 20px;
}

.pipeline-number {
    font-size: 20px;
    font-weight: 700;
    min-width: 44px;
    height: 44px;
    border-radius: 22px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.pipeline-step.step1 .pipeline-number { background: rgba(0, 255, 153, 0.08); border: 1px solid #00FF99; color: #00FF99; }
.pipeline-step.step2 .pipeline-number { background: rgba(0, 176, 255, 0.08); border: 1px solid #00B0FF; color: #00B0FF; }
.pipeline-step.step3 .pipeline-number { background: rgba(168, 85, 247, 0.08); border: 1px solid #A855F7; color: #A855F7; }
.pipeline-step.step4 .pipeline-number { background: rgba(239, 68, 68, 0.08); border: 1px solid #EF4444; color: #EF4444; }
.pipeline-step.step5 .pipeline-number { background: rgba(255, 193, 7, 0.08); border: 1px solid #FFC107; color: #FFC107; }

.pipeline-icon {
    font-size: 26px;
}

.pipeline-content h4 {
    margin: 0 0 4px 0;
    font-size: 16px;
    font-weight: 600;
    color: #FFFFFF;
}

.pipeline-content p {
    margin: 0;
    font-size: 13px;
    color: #94A3B8;
    line-height: 1.4;
}

/* Customizing tab elements */
div[data-testid="stHorizontalBlock"] button[role="tab"] {
    font-size: 16px !important;
    font-weight: 600 !important;
    color: #94A3B8 !important;
}

div[data-testid="stHorizontalBlock"] button[role="tab"][aria-selected="true"] {
    color: #00FF99 !important;
    border-bottom: 2px solid #00FF99 !important;
}
</style>
""", unsafe_allow_html=True)

# Cache Model Load
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_model.h5")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model plant_disease_model.h5: {e}")

# Read Disease Labels
with open("labels.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Top Dashboard Header
st.markdown('<div class="badge-header">AI · COMPUTER VISION · DEEP LEARNING</div>', unsafe_allow_html=True)
st.markdown('<h1 class="main-title">AI Plant Disease <span>Detection Dashboard</span></h1>', unsafe_allow_html=True)
st.markdown('<p class="main-subtitle">Upload a leaf image · Get instant disease diagnosis · Receive treatment & prevention tips</p>', unsafe_allow_html=True)

# KPI Metrics Cards (Slide 1)
st.markdown("""
<div class="metric-row">
    <div class="metric-card classes">
        <div class="icon">🌱</div>
        <div class="value">15</div>
        <div class="label">Disease Classes</div>
    </div>
    <div class="metric-card accuracy">
        <div class="icon">🧠</div>
        <div class="value">95%+</div>
        <div class="label">Detection Accuracy</div>
    </div>
    <div class="metric-card crops">
        <div class="icon">🌾</div>
        <div class="value">3</div>
        <div class="label">Crop Types Covered</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Main Navigation Tabs
tab1, tab2, tab3 = st.tabs(["🌱 Detection Dashboard", "🔄 5-Step AI Pipeline", "📋 Disease Catalog"])

# TAB 1: DETECTION DASHBOARD
with tab1:
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Upload Leaf Image</div>', unsafe_allow_html=True)
        
        uploader_file = st.file_uploader(
            "Upload leaf image file",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        
        # Crop cover indicators (Slide 1)
        st.markdown("""
        <div class="crop-badge-container">
            <div class="crop-badge">✓ Tomato</div>
            <div class="crop-badge">✓ Potato</div>
            <div class="crop-badge">✓ Pepper</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display Uploaded Image inside a Card if available
        if uploader_file is not None:
            image = Image.open(uploader_file).convert("RGB")
            st.markdown('<div class="panel-card">', unsafe_allow_html=True)
            st.markdown('<div class="panel-title">Uploaded Leaf Image</div>', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
    with col2:
        if uploader_file is not None:
            # Model Preprocessing & Prediction
            img = image.resize((224, 224))
            img_arr = np.array(img) / 255.0
            img_arr = np.expand_dims(img_arr, axis=0)
            
            with st.spinner("Running deep learning model inference..."):
                prediction = model.predict(img_arr, verbose=0)
                predicted_index = np.argmax(prediction)
                predicted_class = classes[predicted_index]
                confidence = float(np.max(prediction)) * 100
                
            display_name = CLASS_DISPLAY_NAMES.get(predicted_class, predicted_class)
            category = CLASS_CATEGORIES.get(predicted_class, "Unknown")
            
            # Prediction Results Card (Slide 2 layout)
            st.markdown(f"""
            <div class="predicted-header">
                <div class="predicted-label">Predicted Disease</div>
                <div class="predicted-value">{display_name}</div>
            </div>
            <div class="confidence-header">
                <div class="confidence-label">Confidence Score</div>
                <div class="confidence-value">{confidence:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence Status Banner
            if confidence > 90:
                st.success("✓ High Confidence Prediction")
            elif confidence > 70:
                st.warning("⚠ Medium Confidence Prediction")
            else:
                st.error("✗ Low Confidence Prediction")
                
            # Disease Info details (Slide 2 magnifying glass, flask, shield icons)
            if predicted_class in disease_info:
                info = disease_info[predicted_class]
                st.markdown(f"""
                <div class="panel-card">
                    <div class="panel-title">Disease Info</div>
                    <div class="info-section">
                        <div class="info-icon">🔍</div>
                        <div class="info-content">
                            <h4>Description</h4>
                            <p>{info['description']}</p>
                        </div>
                    </div>
                    <div class="info-section">
                        <div class="info-icon">🧪</div>
                        <div class="info-content">
                            <h4>Treatment</h4>
                            <p>{info['treatment']}</p>
                        </div>
                    </div>
                    <div class="info-section">
                        <div class="info-icon">🛡️</div>
                        <div class="info-content">
                            <h4>Prevention</h4>
                            <p>{info['prevention']}</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Disease description and treatment guides not available for this class.")
                
            # Top 3 Predictions Progress Bars
            st.markdown('<div class="panel-card">', unsafe_allow_html=True)
            st.markdown('<div class="panel-title">Top 3 Predictions</div>', unsafe_allow_html=True)
            
            top3 = np.argsort(prediction[0])[-3:][::-1]
            for idx in top3:
                cls_name = classes[idx]
                cls_display = CLASS_DISPLAY_NAMES.get(cls_name, cls_name)
                score = prediction[0][idx] * 100
                
                st.write(f"**{cls_display}** : {score:.2f}%")
                st.progress(float(score / 100))
            st.markdown('</div>', unsafe_allow_html=True)
            
        else:
            # Placeholder State
            st.markdown("""
            <div class="panel-card" style="text-align: center; padding: 60px 20px; color: #64748B;">
                <div style="font-size: 50px; margin-bottom: 20px;">🌱</div>
                <h3 style="color: #FFFFFF;">Awaiting Leaf Image</h3>
                <p>Upload a photograph of a plant leaf in the panel to the left to receive diagnostic recommendations and insights.</p>
            </div>
            """, unsafe_allow_html=True)


# TAB 2: 5-STEP AI PIPELINE
with tab2:
    st.markdown('<h2 style="color: #FFFFFF; font-weight: 700; margin-bottom: 5px;">5-Step AI Pipeline</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color: #94A3B8; margin-bottom: 30px;">From leaf image to diagnosis in milliseconds</p>', unsafe_allow_html=True)
    
    col_pipe_left, col_pipe_right = st.columns([1.2, 1])
    
    with col_pipe_left:
        st.markdown("""
        <div class="pipeline-container">
            <div class="pipeline-step step1">
                <div class="pipeline-number">01</div>
                <div class="pipeline-icon">📤</div>
                <div class="pipeline-content">
                    <h4>Upload Image</h4>
                    <p>User uploads JPG/PNG leaf photo via Streamlit file uploader.</p>
                </div>
            </div>
            <div style="height: 1px;"></div>
            <div class="pipeline-step step2">
                <div class="pipeline-number">02</div>
                <div class="pipeline-icon">⚙️</div>
                <div class="pipeline-content">
                    <h4>Preprocessing</h4>
                    <p>Resize image to 224×224 pixels, normalize pixel intensities (divide by 255), and expand batch dimensions.</p>
                </div>
            </div>
            <div style="height: 1px;"></div>
            <div class="pipeline-step step3">
                <div class="pipeline-number">03</div>
                <div class="pipeline-icon">🧠</div>
                <div class="pipeline-content">
                    <h4>CNN Inference</h4>
                    <p>Deep learning Convolutional Neural Network (VGG16-based) predicts probability distribution across 15 classes.</p>
                </div>
            </div>
            <div style="height: 1px;"></div>
            <div class="pipeline-step step4">
                <div class="pipeline-number">04</div>
                <div class="pipeline-icon">⚠️</div>
                <div class="pipeline-content">
                    <h4>Disease Identified</h4>
                    <p>Top predicted class index and softmax confidence score extracted via mathematical argmax functions.</p>
                </div>
            </div>
            <div style="height: 1px;"></div>
            <div class="pipeline-step step5">
                <div class="pipeline-number">05</div>
                <div class="pipeline-icon">🧪</div>
                <div class="pipeline-content">
                    <h4>Treatment Delivered</h4>
                    <p>Targeted descriptions, medicinal treatments, and preventive agronomic guidelines returned to dashboard UI.</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_pipe_right:
        st.markdown('<h3 style="color: #FFFFFF; margin-bottom: 20px; font-weight: 600;">Live Dashboard Preview</h3>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="panel-card">
            <div class="panel-title" style="margin-bottom: 10px;">Uploaded Image</div>
            <div style="text-align: center; background: #0D111A; border: 1px dashed #1E293B; border-radius: 12px; padding: 40px 20px; margin-bottom: 20px;">
                <div style="font-size: 40px; color: #00FF99; margin-bottom: 10px;">🌱</div>
                <div style="color: #94A3B8; font-size: 14px; font-weight: 600;">Leaf Image Preview</div>
            </div>
            
            <div class="predicted-header" style="margin-bottom: 12px; padding: 12px 16px;">
                <div class="predicted-label" style="font-size: 10px;">Predicted Disease</div>
                <div class="predicted-value" style="font-size: 18px; margin-top: 2px;">Tomato Early Blight</div>
            </div>
            
            <div class="confidence-header" style="margin-bottom: 15px; padding: 12px 16px;">
                <div class="confidence-label" style="font-size: 10px;">Confidence Score</div>
                <div class="confidence-value" style="font-size: 18px; margin-top: 2px;">94.3%</div>
            </div>
            
            <div style="text-align: center; padding: 8px; background: rgba(0, 255, 153, 0.1); border-radius: 6px; color: #00FF99; font-size: 12px; font-weight: 600; margin-bottom: 20px;">
                ✓ High Confidence Prediction
            </div>
            
            <div>
                <div style="font-size: 13px; font-weight: 600; color: #FFFFFF; margin-bottom: 8px;">Top 3 Predictions</div>
                
                <div style="font-size: 12px; color: #94A3B8; display: flex; justify-content: space-between; margin-bottom: 2px;">
                    <span>Early Blight</span><span>94.3%</span>
                </div>
                <div style="background: #0D111A; height: 6px; border-radius: 3px; margin-bottom: 10px;">
                    <div style="background: #00FF99; width: 94.3%; height: 100%; border-radius: 3px;"></div>
                </div>
                
                <div style="font-size: 12px; color: #94A3B8; display: flex; justify-content: space-between; margin-bottom: 2px;">
                    <span>Septoria Spot</span><span>3.5%</span>
                </div>
                <div style="background: #0D111A; height: 6px; border-radius: 3px; margin-bottom: 10px;">
                    <div style="background: #00B0FF; width: 3.5%; height: 100%; border-radius: 3px;"></div>
                </div>
                
                <div style="font-size: 12px; color: #94A3B8; display: flex; justify-content: space-between; margin-bottom: 2px;">
                    <span>Leaf Mold</span><span>2.2%</span>
                </div>
                <div style="background: #0D111A; height: 6px; border-radius: 3px;">
                    <div style="background: #A855F7; width: 2.2%; height: 100%; border-radius: 3px;"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# TAB 3: DISEASE CATALOG & TECH STACK
with tab3:
    st.markdown('<h2 style="color: #FFFFFF; font-weight: 700; margin-bottom: 5px;">15 Disease Classes Detected</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color: #94A3B8; margin-bottom: 30px;">Search catalog and review underlying system parameters</p>', unsafe_allow_html=True)
    
    col_cat_left, col_cat_right = st.columns([1.5, 1])
    
    with col_cat_left:
        # Search & Filters
        fcol1, fcol2 = st.columns(2)
        with fcol1:
            search_query = st.text_input("🔍 Search Catalog", "", placeholder="Search crop or disease name...")
        with fcol2:
            selected_cat = st.selectbox("📂 Filter by Category", ["All", "Fungal", "Healthy", "Viral", "Bacterial", "Pest"])
            
        # Filter Logic
        filtered_classes = {}
        for filename, display_name in CLASS_DISPLAY_NAMES.items():
            category = CLASS_CATEGORIES[filename]
            # Category filter
            if selected_cat != "All" and category != selected_cat:
                continue
            # Search filter
            if search_query.lower() not in display_name.lower():
                continue
            filtered_classes[filename] = display_name
            
        # Catalog Grid Representation (Slide 3)
        grid_html = '<div class="catalog-grid">'
        for filename, display_name in filtered_classes.items():
            category = CLASS_CATEGORIES[filename]
            badge_class = category.lower()
            info = disease_info.get(filename, {"description": "Information pending."})
            desc = info["description"]
            
            grid_html += f"""
            <div class="catalog-card">
                <span class="badge-pill {badge_class}">{category}</span>
                <h5>{display_name}</h5>
                <p style="font-size: 12px; color: #94A3B8; margin: 8px 0 0 0; line-height: 1.4;">{desc}</p>
            </div>
            """
        grid_html += '</div>'
        
        if not filtered_classes:
            st.warning("No disease classes matched your search parameters.")
        else:
            st.markdown(grid_html, unsafe_allow_html=True)
            
    with col_cat_right:
        # Tech Stack details
        st.markdown("""
        <div class="panel-card">
            <div class="panel-title">Tech Stack</div>
            <div style="display: flex; flex-direction: column; gap: 20px;">
                <div style="display: flex; align-items: flex-start; gap: 15px;">
                    <div style="font-size: 20px; background: rgba(168, 85, 247, 0.08); width: 40px; height: 40px; border-radius: 20px; display: flex; align-items: center; justify-content: center; border: 1px solid #A855F7; color: #A855F7;">🧠</div>
                    <div>
                        <div style="font-weight: 600; color: #FFFFFF; font-size: 14px;">TensorFlow / Keras</div>
                        <div style="font-size: 12px; color: #94A3B8; margin-top: 2px;">CNN model inference (VGG16 architecture)</div>
                    </div>
                </div>
                <div style="display: flex; align-items: flex-start; gap: 15px;">
                    <div style="font-size: 20px; background: rgba(0, 255, 153, 0.08); width: 40px; height: 40px; border-radius: 20px; display: flex; align-items: center; justify-content: center; border: 1px solid #00FF99; color: #00FF99;">💻</div>
                    <div>
                        <div style="font-weight: 600; color: #FFFFFF; font-size: 14px;">Streamlit</div>
                        <div style="font-size: 12px; color: #94A3B8; margin-top: 2px;">Interactive web dashboard interface</div>
                    </div>
                </div>
                <div style="display: flex; align-items: flex-start; gap: 15px;">
                    <div style="font-size: 20px; background: rgba(0, 176, 255, 0.08); width: 40px; height: 40px; border-radius: 20px; display: flex; align-items: center; justify-content: center; border: 1px solid #00B0FF; color: #00B0FF;">⚙️</div>
                    <div>
                        <div style="font-weight: 600; color: #FFFFFF; font-size: 14px;">NumPy + PIL</div>
                        <div style="font-size: 12px; color: #94A3B8; margin-top: 2px;">Image processing & dimension normalization</div>
                    </div>
                </div>
                <div style="display: flex; align-items: flex-start; gap: 15px;">
                    <div style="font-size: 20px; background: rgba(255, 193, 7, 0.08); width: 40px; height: 40px; border-radius: 20px; display: flex; align-items: center; justify-content: center; border: 1px solid #FFC107; color: #FFC107;">📂</div>
                    <div>
                        <div style="font-weight: 600; color: #FFFFFF; font-size: 14px;">disease_info.py</div>
                        <div style="font-size: 12px; color: #94A3B8; margin-top: 2px;">15-class description & treatment database</div>
                    </div>
                </div>
                <div style="display: flex; align-items: flex-start; gap: 15px;">
                    <div style="font-size: 20px; background: rgba(239, 68, 68, 0.08); width: 40px; height: 40px; border-radius: 20px; display: flex; align-items: center; justify-content: center; border: 1px solid #EF4444; color: #EF4444;">💾</div>
                    <div>
                        <div style="font-weight: 600; color: #FFFFFF; font-size: 14px;">plant_disease_model</div>
                        <div style="font-size: 12px; color: #94A3B8; margin-top: 2px;">H5 pre-trained CNN model — 95%+ accuracy</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Disease breakdown charts (Slide 3)
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Disease Breakdown</div>', unsafe_allow_html=True)
        
        # Pie chart layout matching slide 3 colors & categories
        categories_pie = ['Fungal (7)', 'Healthy (3)', 'Viral (2)', 'Bacterial (2)', 'Pest (1)']
        counts_pie = [7, 3, 2, 2, 1]
        colors_pie = ['#FFC107', '#00FF99', '#00B0FF', '#EF4444', '#A855F7']
        
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        fig.patch.set_facecolor('#111625')
        ax.set_facecolor('#111625')
        
        wedges, texts, autotexts = ax.pie(
            counts_pie, 
            labels=categories_pie, 
            autopct='%1.0f%%', 
            colors=colors_pie, 
            startangle=140,
            textprops=dict(color="#94A3B8", fontsize=9, weight="bold")
        )
        
        # Change color of labels inside wedges to dark theme highlight
        for autotext in autotexts:
            autotext.set_color('#080C14')
            autotext.set_fontsize(9)
            
        ax.axis('equal')
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)