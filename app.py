import io
import datetime
import tensorflow as tf
import streamlit as st
from PIL import Image
import plotly.graph_objects as go

from disease_info import disease_info
from inference import (
    preprocess_image,
    prediction_probabilities,
    project_path,
    top_predictions,
)
from report_generator import generate_pdf_report

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="FloraScan AI · Smart Plant Pathology & Crop Protection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# CONSTANTS & MAPPINGS
# ==========================================
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

# ==========================================
# MODERN WEB UI STYLES (CUSTOM CSS)
# ==========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [data-testid="stAppViewContainer"], .stApp {
    background-color: #070B14 !important;
    font-family: 'Outfit', sans-serif !important;
    color: #E2E8F0 !important;
}

[data-testid="stHeader"] {
    background-color: transparent !important;
}

header, footer {
    visibility: hidden !important;
    height: 0px !important;
}

/* Glassmorphism Navigation Bar */
.navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: rgba(15, 23, 42, 0.75);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 14px 24px;
    margin-bottom: 25px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
}

.brand-container {
    display: flex;
    align-items: center;
    gap: 12px;
}

.brand-logo {
    font-size: 26px;
}

.brand-name {
    font-size: 20px;
    font-weight: 700;
    color: #FFFFFF;
    letter-spacing: -0.5px;
}

.brand-name span {
    color: #00FF99;
}

.nav-badge {
    background: rgba(0, 255, 153, 0.1);
    border: 1px solid rgba(0, 255, 153, 0.4);
    color: #00FF99;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.5px;
}

.nav-links {
    display: flex;
    align-items: center;
    gap: 16px;
}

.github-link {
    background: #1E293B;
    border: 1px solid #334155;
    color: #F8FAFC !important;
    padding: 7px 16px;
    border-radius: 10px;
    font-size: 13px;
    font-weight: 600;
    text-decoration: none !important;
    transition: all 0.2s ease;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

.github-link:hover {
    background: #334155;
    border-color: #00FF99;
    color: #00FF99 !important;
    transform: translateY(-2px);
}

/* Hero Section */
.hero-container {
    text-align: center;
    padding: 30px 20px 20px 20px;
    max-width: 900px;
    margin: 0 auto;
}

.hero-pill {
    background: rgba(0, 255, 153, 0.08);
    border: 1px solid rgba(0, 255, 153, 0.3);
    color: #00FF99;
    padding: 5px 16px;
    border-radius: 30px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    display: inline-block;
    margin-bottom: 16px;
}

.hero-title {
    font-size: 46px;
    font-weight: 800;
    line-height: 1.15;
    color: #FFFFFF;
    margin-bottom: 14px;
    letter-spacing: -1px;
}

.hero-title .gradient-text {
    background: linear-gradient(135deg, #00FF99 0%, #00E5FF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-description {
    font-size: 17px;
    color: #94A3B8;
    line-height: 1.6;
    margin-bottom: 30px;
}

/* KPI Stats Cards */
.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin-bottom: 35px;
}

.stat-card {
    background: rgba(17, 24, 39, 0.7);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 16px;
    padding: 20px;
    text-align: left;
    transition: all 0.3s ease;
}

.stat-card:hover {
    transform: translateY(-4px);
    border-color: rgba(0, 255, 153, 0.4);
    box-shadow: 0 10px 25px rgba(0, 255, 153, 0.1);
}

.stat-icon {
    font-size: 24px;
    margin-bottom: 10px;
}

.stat-number {
    font-size: 28px;
    font-weight: 800;
    color: #FFFFFF;
    margin-bottom: 4px;
}

.stat-label {
    font-size: 13px;
    color: #94A3B8;
    font-weight: 500;
}

/* Section Cards */
.panel-card {
    background: rgba(17, 24, 39, 0.75);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}

.panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    padding-bottom: 14px;
    margin-bottom: 18px;
}

.panel-title {
    font-size: 18px;
    font-weight: 700;
    color: #FFFFFF;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* Prediction Highlights */
.result-header-card {
    background: linear-gradient(135deg, rgba(0, 255, 153, 0.08) 0%, rgba(0, 229, 255, 0.04) 100%);
    border: 1px solid rgba(0, 255, 153, 0.4);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 20px;
}

.result-tag-row {
    display: flex;
    gap: 10px;
    margin-bottom: 10px;
    flex-wrap: wrap;
}

.result-title {
    font-size: 28px;
    font-weight: 800;
    color: #FFFFFF;
    margin-bottom: 4px;
}

.confidence-big {
    font-size: 26px;
    font-weight: 800;
    color: #00FF99;
    font-family: 'JetBrains Mono', monospace;
}

/* Category & Severity Badges */
.badge-tag {
    padding: 4px 12px;
    border-radius: 8px;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    display: inline-block;
}

.badge-tag.fungal { background: rgba(245, 158, 11, 0.15); border: 1px solid #F59E0B; color: #F59E0B; }
.badge-tag.healthy { background: rgba(0, 255, 153, 0.15); border: 1px solid #00FF99; color: #00FF99; }
.badge-tag.bacterial { background: rgba(239, 68, 68, 0.15); border: 1px solid #EF4444; color: #EF4444; }
.badge-tag.viral { background: rgba(59, 130, 246, 0.15); border: 1px solid #3B82F6; color: #3B82F6; }
.badge-tag.pest { background: rgba(168, 85, 247, 0.15); border: 1px solid #A855F7; color: #A855F7; }

.badge-tag.severity-critical { background: rgba(220, 38, 38, 0.2); border: 1px solid #DC2626; color: #FCA5A5; }
.badge-tag.severity-high { background: rgba(249, 115, 22, 0.2); border: 1px solid #F97316; color: #FDBA74; }
.badge-tag.severity-moderate { background: rgba(234, 179, 8, 0.2); border: 1px solid #EAB308; color: #FDE047; }
.badge-tag.severity-healthy { background: rgba(16, 185, 129, 0.2); border: 1px solid #10B981; color: #6EE7B7; }

/* Clinical details cards */
.clinical-card {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 14px;
    display: flex;
    gap: 16px;
    align-items: flex-start;
}

.clinical-icon {
    font-size: 24px;
    min-width: 36px;
    height: 36px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(255, 255, 255, 0.05);
}

.clinical-content h4 {
    margin: 0 0 6px 0;
    font-size: 15px;
    font-weight: 700;
    color: #FFFFFF;
}

.clinical-content p {
    margin: 0;
    font-size: 14px;
    color: #94A3B8;
    line-height: 1.5;
}

/* Disease Catalog Grid */
.catalog-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 18px;
    margin-top: 20px;
}

.catalog-card {
    background: rgba(17, 24, 39, 0.85);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 14px;
    padding: 18px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: all 0.25s ease;
}

.catalog-card:hover {
    border-color: rgba(0, 255, 153, 0.4);
    transform: translateY(-3px);
}

.catalog-card h4 {
    margin: 10px 0 6px 0;
    font-size: 16px;
    font-weight: 700;
    color: #FFFFFF;
}

.catalog-card p {
    font-size: 13px;
    color: #94A3B8;
    line-height: 1.45;
    margin-bottom: 12px;
}

/* Pipeline Step */
.pipeline-step {
    background: rgba(17, 24, 39, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 20px;
}

.pipeline-badge {
    min-width: 46px;
    height: 46px;
    border-radius: 23px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    font-weight: 800;
    background: rgba(0, 255, 153, 0.1);
    border: 1px solid #00FF99;
    color: #00FF99;
}

/* Tabs styling */
button[data-baseweb="tab"] {
    font-size: 15px !important;
    font-weight: 600 !important;
    color: #94A3B8 !important;
    padding: 10px 20px !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #00FF99 !important;
    border-bottom: 3px solid #00FF99 !important;
}

/* Custom Footer */
.site-footer {
    text-align: center;
    padding: 40px 20px 20px 20px;
    color: #64748B;
    font-size: 13px;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
    margin-top: 60px;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# MODEL & LABELS CACHING
# ==========================================
@st.cache_resource
def load_deep_learning_model():
    return tf.keras.models.load_model(str(project_path("plant_disease_model.h5")))

try:
    model = load_deep_learning_model()
except Exception as e:
    st.error(f"⚠️ Error loading deep learning model weights: {e}")

with open(project_path("labels.txt"), "r") as f:
    classes = [line.strip() for line in f.readlines()]

# ==========================================
# STICKY NAVBAR
# ==========================================
st.markdown("""
<div class="navbar">
    <div class="brand-container">
        <span class="brand-logo">🌿</span>
        <div>
            <div class="brand-name">FloraScan <span>AI</span></div>
        </div>
        <span class="nav-badge">VGG16 · 95%+ ACCURACY</span>
    </div>
    <div class="nav-links">
        <a class="github-link" href="https://github.com/Nishant052004/Plant-Diseases" target="_blank">
            <svg height="16" width="16" viewBox="0 0 16 16" fill="currentColor">
                <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/>
            </svg>
            GitHub Repo
        </a>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# HERO SECTION
# ==========================================
st.markdown("""
<div class="hero-container">
    <div class="hero-pill">Next-Gen Agricultural Intelligence</div>
    <h1 class="hero-title">Instant Crop Pathology & <span class="gradient-text">Agronomy Advisory</span></h1>
    <p class="hero-description">
        Detect plant infections in milliseconds with laboratory-grade computer vision. Upload a leaf photo, inspect confidence distributions, and download verified treatment prescriptions directly.
    </p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# KPI STATS BAR
# ==========================================
st.markdown("""
<div class="stats-grid">
    <div class="stat-card">
        <div class="stat-icon">🌱</div>
        <div class="stat-number">15</div>
        <div class="stat-label">Health & Disease States</div>
    </div>
    <div class="stat-card">
        <div class="stat-icon">🎯</div>
        <div class="stat-number">95%+</div>
        <div class="stat-label">Model Validation Accuracy</div>
    </div>
    <div class="stat-card">
        <div class="stat-icon">⚡</div>
        <div class="stat-number">&lt; 120ms</div>
        <div class="stat-label">Sub-Second Inference Speed</div>
    </div>
    <div class="stat-card">
        <div class="stat-icon">📑</div>
        <div class="stat-number">PDF Ready</div>
        <div class="stat-label">One-Click Prescription Export</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# MAIN NAVIGATION TABS
# ==========================================
tab_diag, tab_samples, tab_pipeline, tab_catalog, tab_about = st.tabs([
    "🔬 Diagnostic Lab",
    "🖼️ Quick Sample Test",
    "🔄 5-Step AI Pipeline",
    "📚 Disease Catalog & Analytics",
    "ℹ️ System Specs"
])

# Maintain image selection in session_state
if "active_image" not in st.session_state:
    st.session_state.active_image = None
if "image_source_name" not in st.session_state:
    st.session_state.image_source_name = None

# ==========================================
# TAB 2: QUICK SAMPLE TEST
# ==========================================
with tab_samples:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">⚡ Test Model Instantly with Pre-Loaded Samples</div>', unsafe_allow_html=True)
    st.markdown('<p style="color: #94A3B8; font-size: 14px;">No leaf photo on hand? Click any sample below to load it directly into the Diagnostic Lab.</p>', unsafe_allow_html=True)
    
    samples_info = [
        {"file": "samples/tomato_early_blight.jpg", "label": "Tomato Early Blight", "crop": "Tomato (Fungal)", "severity": "Moderate"},
        {"file": "samples/potato_late_blight.jpg", "label": "Potato Late Blight", "crop": "Potato (Fungal)", "severity": "Critical"},
        {"file": "samples/pepper_bacterial_spot.jpg", "label": "Pepper Bacterial Spot", "crop": "Pepper (Bacterial)", "severity": "High"},
        {"file": "samples/tomato_healthy.jpg", "label": "Healthy Tomato", "crop": "Tomato (Healthy)", "severity": "Healthy"},
    ]
    
    scol1, scol2, scol3, scol4 = st.columns(4)
    cols = [scol1, scol2, scol3, scol4]
    
    for idx, s in enumerate(samples_info):
        with cols[idx]:
            sample_path = project_path(s["file"])
            if sample_path.exists():
                thumb_img = Image.open(sample_path)
                st.image(thumb_img, use_container_width=True)
                st.markdown(f"**{s['label']}**")
                st.caption(f"{s['crop']} · {s['severity']}")
                if st.button(f"Load Sample #{idx+1}", key=f"btn_sample_{idx}", use_container_width=True):
                    st.session_state.active_image = Image.open(sample_path).convert("RGB")
                    st.session_state.image_source_name = s["label"]
                    st.success(f"✓ Loaded {s['label']}. Switch to Diagnostic Lab tab to view results!")
            else:
                st.info(f"Sample {s['label']} not found.")
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# TAB 1: DIAGNOSTIC LAB
# ==========================================
with tab_diag:
    col_input, col_result = st.columns([1.1, 1.3], gap="large")
    
    with col_input:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">📷 Leaf Image Ingestion</div>', unsafe_allow_html=True)
        
        input_mode = st.radio(
            "Input Mode",
            ["📁 Upload Leaf Photo", "📸 Live Camera Capture"],
            horizontal=True,
            label_visibility="collapsed"
        )
        
        if input_mode == "📁 Upload Leaf Photo":
            uploaded_file = st.file_uploader(
                "Upload a clear photograph of a plant leaf",
                type=["jpg", "jpeg", "png"],
                help="Accepts high-resolution JPG or PNG photographs of plant foliage."
            )
            if uploaded_file is not None:
                st.session_state.active_image = Image.open(uploaded_file).convert("RGB")
                st.session_state.image_source_name = uploaded_file.name
        else:
            camera_file = st.camera_input("Take a photo of a plant leaf")
            if camera_file is not None:
                st.session_state.active_image = Image.open(camera_file).convert("RGB")
                st.session_state.image_source_name = "Live Camera Capture"
                
        # Display Current Active Image
        if st.session_state.active_image is not None:
            st.markdown('<div style="margin-top: 15px;">', unsafe_allow_html=True)
            st.image(st.session_state.active_image, caption=f"Active Image: {st.session_state.image_source_name}", use_container_width=True)
            
            if st.button("🔄 Clear / Reset Image", use_container_width=True):
                st.session_state.active_image = None
                st.session_state.image_source_name = None
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="border: 2px dashed rgba(255,255,255,0.1); border-radius: 12px; padding: 40px 20px; text-align: center; color: #64748B; margin-top: 20px;">
                <div style="font-size: 40px; margin-bottom: 10px;">🍃</div>
                <div style="font-weight: 600; color: #CBD5E1;">No Leaf Image Loaded</div>
                <div style="font-size: 13px; margin-top: 6px;">Upload a leaf photo above, snap one with your camera, or choose a sample from the Quick Sample Test tab.</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_result:
        if st.session_state.active_image is not None:
            # Preprocessing & Inference
            with st.spinner("🧠 Executing Convolutional Neural Network inference..."):
                image_batch = preprocess_image(st.session_state.active_image)
                raw_pred = prediction_probabilities(model.predict(image_batch, verbose=0), len(classes))
                pred_idx, pred_probability = top_predictions(raw_pred, limit=1)[0]
                pred_class = classes[pred_idx]
                confidence_score = pred_probability * 100
                
            display_name = CLASS_DISPLAY_NAMES.get(pred_class, pred_class)
            category = CLASS_CATEGORIES.get(pred_class, "Unknown")
            details = disease_info.get(pred_class, {
                "description": "Pathology information currently being cataloged.",
                "treatment": "Consult an agricultural extension specialist.",
                "prevention": "Ensure good crop hygiene and clean soil.",
                "severity": "Moderate"
            })
            severity = details.get("severity", "Moderate")
            
            cat_badge_cls = category.lower()
            sev_badge_cls = f"severity-{severity.lower().split()[0]}"
            
            # Primary Result Card
            st.markdown(f"""
            <div class="result-header-card">
                <div class="result-tag-row">
                    <span class="badge-tag {cat_badge_cls}">{category}</span>
                    <span class="badge-tag {sev_badge_cls}">Severity: {severity}</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: flex-end;">
                    <div>
                        <div style="font-size: 12px; color: #94A3B8; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;">Clinical Diagnosis</div>
                        <div class="result-title">{display_name}</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 12px; color: #94A3B8; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;">Confidence</div>
                        <div class="confidence-big">{confidence_score:.1f}%</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence Status Notification
            if confidence_score >= 90:
                st.success("✓ **High Confidence Match**: The visual features strongly correspond with trained pathology patterns.")
            elif confidence_score >= 70:
                st.warning("⚠️ **Moderate Confidence**: Diagnosis is likely accurate, but visual confirmation against secondary symptoms is recommended.")
            else:
                st.error("✗ **Low Confidence**: Ambiguous visual features. Ensure the leaf is in focus, well-lit, and fills the camera frame.")
                
            # Top 3 Candidates Bar Chart (Plotly)
            st.markdown('<div class="panel-card">', unsafe_allow_html=True)
            st.markdown('<div class="panel-title">📊 Top 3 Candidate Probabilities</div>', unsafe_allow_html=True)
            
            top3_data = []
            top3_labels = []
            top3_scores = []

            for idx, prob in top_predictions(raw_pred, limit=3):
                cname = classes[idx]
                dname = CLASS_DISPLAY_NAMES.get(cname, cname)
                sc = prob * 100
                top3_labels.append(dname)
                top3_scores.append(sc)
                top3_data.append((dname, sc))
                
            fig_bar = go.Figure(go.Bar(
                x=top3_scores[::-1],
                y=top3_labels[::-1],
                orientation='h',
                marker=dict(
                    color=['#3B82F6', '#00E5FF', '#00FF99'],
                    line=dict(color='rgba(255,255,255,0.2)', width=1)
                ),
                text=[f"{s:.1f}%" for s in top3_scores[::-1]],
                textposition='auto',
                textfont=dict(color='#FFFFFF', family='Outfit', size=12, weight='bold')
            ))
            fig_bar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=10, b=10),
                height=180,
                xaxis=dict(showgrid=False, showticklabels=False, range=[0, 105]),
                yaxis=dict(tickfont=dict(color='#E2E8F0', family='Outfit', size=13)),
            )
            st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Actionable Prescription Details
            st.markdown('<div class="panel-card">', unsafe_allow_html=True)
            st.markdown('<div class="panel-title">📋 Clinical Prescription & Advisory</div>', unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="clinical-card">
                <div class="clinical-icon">🔍</div>
                <div class="clinical-content">
                    <h4>Pathology & Visual Symptoms</h4>
                    <p>{details['description']}</p>
                </div>
            </div>
            <div class="clinical-card">
                <div class="clinical-icon">🧪</div>
                <div class="clinical-content">
                    <h4>Recommended Treatment Protocol</h4>
                    <p>{details['treatment']}</p>
                </div>
            </div>
            <div class="clinical-card">
                <div class="clinical-icon">🛡️</div>
                <div class="clinical-content">
                    <h4>Agronomic Prevention & Cultural Management</h4>
                    <p>{details['prevention']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # PDF Generation & Download
            pdf_bytes = generate_pdf_report(
                leaf_image=st.session_state.active_image,
                display_name=display_name,
                category=category,
                confidence=confidence_score,
                severity=severity,
                info=details,
                top3=top3_data
            )
            
            st.download_button(
                label="📥 Download Pathology Report (PDF)",
                data=pdf_bytes,
                file_name=f"FloraScan_Report_{display_name.replace(' ', '_')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
        else:
            st.markdown("""
            <div class="panel-card" style="text-align: center; padding: 70px 20px;">
                <div style="font-size: 54px; margin-bottom: 16px;">🌱</div>
                <h3 style="color: #FFFFFF; font-weight: 700;">Awaiting Diagnostic Input</h3>
                <p style="color: #94A3B8; max-width: 420px; margin: 0 auto 20px auto; font-size: 14px; line-height: 1.6;">
                    Upload a leaf photograph or snap a picture to trigger instant neural network inference, confidence telemetry, and treatment guidelines.
                </p>
                <div style="background: rgba(0, 255, 153, 0.05); border: 1px dashed rgba(0, 255, 153, 0.3); border-radius: 12px; padding: 14px; display: inline-block; color: #00FF99; font-size: 13px; font-weight: 600;">
                    💡 Tip: Try the "Quick Sample Test" tab to test with pre-loaded samples in 1 click!
                </div>
            </div>
            """, unsafe_allow_html=True)

# ==========================================
# TAB 3: 5-STEP AI PIPELINE
# ==========================================
with tab_pipeline:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">🔄 End-to-End Deep Learning Architecture</div>', unsafe_allow_html=True)
    st.markdown('<p style="color: #94A3B8; font-size: 14px;">How FloraScan transforms raw smartphone leaf photography into clinical agricultural prescriptions.</p>', unsafe_allow_html=True)
    
    pipeline_steps = [
        {"num": "01", "icon": "📤", "title": "Image Acquisition", "desc": "User captures or uploads leaf photography via WebRTC camera interface or HTML5 file drag-and-drop."},
        {"num": "02", "icon": "⚙️", "title": "Bilinear Normalization", "desc": "Input image is resized to 224×224 pixels, converted to an RGB float32 tensor, and normalized to range [0, 1]."},
        {"num": "03", "icon": "🧠", "title": "VGG16 Deep Feature Extraction", "desc": "Passes through deep convolutional layers with max-pooling, extracting hierarchical features from leaf veins and lesion margins."},
        {"num": "04", "icon": "🎯", "title": "Softmax Classification & Argmax", "desc": "Dense classification head maps high-dimensional latent vectors into probability distributions across 15 target categories."},
        {"num": "05", "icon": "📑", "title": "Clinical Advisory & PDF Dispatch", "desc": "Automated matching against the disease knowledge base generates chemical treatments, organic alternatives, and printable PDF reports."}
    ]
    
    for s in pipeline_steps:
        st.markdown(f"""
        <div class="pipeline-step">
            <div class="pipeline-badge">{s['num']}</div>
            <div style="font-size: 26px;">{s['icon']}</div>
            <div>
                <h4 style="margin: 0 0 4px 0; font-size: 16px; font-weight: 700; color: #FFFFFF;">{s['title']}</h4>
                <p style="margin: 0; font-size: 13px; color: #94A3B8; line-height: 1.5;">{s['desc']}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# TAB 4: DISEASE CATALOG & INTERACTIVE ANALYTICS
# ==========================================
with tab_catalog:
    cat_col1, cat_col2 = st.columns([1.6, 1], gap="large")
    
    with cat_col1:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">📚 Searchable Plant Disease Catalog</div>', unsafe_allow_html=True)
        
        f1, f2 = st.columns([1.2, 1])
        with f1:
            q = st.text_input("🔍 Search Diseases", placeholder="e.g. Blight, Spot, Tomato, Pepper...")
        with f2:
            sel_category = st.selectbox("📂 Category Filter", ["All", "Fungal", "Healthy", "Viral", "Bacterial", "Pest"])
            
        filtered = {}
        for cname, dname in CLASS_DISPLAY_NAMES.items():
            cat = CLASS_CATEGORIES[cname]
            if sel_category != "All" and cat != sel_category:
                continue
            if q.lower() not in dname.lower():
                continue
            filtered[cname] = dname
            
        catalog_html = '<div class="catalog-grid">'
        for cname, dname in filtered.items():
            cat = CLASS_CATEGORIES[cname]
            info = disease_info.get(cname, {"description": "Information pending.", "severity": "Moderate"})
            sev = info.get("severity", "Moderate")
            badge_cat = cat.lower()
            badge_sev = f"severity-{sev.lower().split()[0]}"
            
            catalog_html += f"""
            <div class="catalog-card">
                <div>
                    <div style="display: flex; gap: 8px; margin-bottom: 8px;">
                        <span class="badge-tag {badge_cat}">{cat}</span>
                        <span class="badge-tag {badge_sev}">{sev}</span>
                    </div>
                    <h4>{dname}</h4>
                    <p>{info['description']}</p>
                </div>
                <div style="font-size: 12px; color: #64748B; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 8px;">
                    <b>Rx:</b> {info['treatment'][:80]}...
                </div>
            </div>
            """
        catalog_html += '</div>'
        
        if not filtered:
            st.warning("No disease classes matched your search query.")
        else:
            st.markdown(catalog_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with cat_col2:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">📊 Pathology Class Distribution</div>', unsafe_allow_html=True)
        
        cat_labels = ['Fungal (7)', 'Healthy (3)', 'Viral (2)', 'Bacterial (2)', 'Pest (1)']
        cat_counts = [7, 3, 2, 2, 1]
        cat_colors = ['#F59E0B', '#00FF99', '#3B82F6', '#EF4444', '#A855F7']
        
        fig_donut = go.Figure(data=[go.Pie(
            labels=cat_labels,
            values=cat_counts,
            hole=0.6,
            marker=dict(colors=cat_colors, line=dict(color='#0F172A', width=2)),
            textinfo='percent',
            textfont=dict(size=12, color='#FFFFFF', family='Outfit', weight='bold'),
            hoverinfo='label+percent+value'
        )])
        fig_donut.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=20, b=10),
            height=300,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.3,
                xanchor="center",
                x=0.5,
                font=dict(color="#94A3B8", family="Outfit", size=11)
            )
        )
        st.plotly_chart(fig_donut, use_container_width=True, config={'displayModeBar': False})
        
        st.markdown("""
        <div style="margin-top: 20px; font-size: 13px; color: #94A3B8; line-height: 1.6;">
            <b>Coverage Breakdown:</b>
            <ul style="padding-left: 20px; margin-top: 6px;">
                <li><b>Fungal Pathogens (47%)</b>: Concentrated around Alternaria and Phytophthora blights.</li>
                <li><b>Healthy Controls (20%)</b>: Baseline non-diseased tissue profiles for all 3 crops.</li>
                <li><b>Viral Inoculations (13%)</b>: TYLCV and Mosaic strains.</li>
                <li><b>Bacterial Invasions (13%)</b>: Xanthomonas bacterial spots.</li>
                <li><b>Pest Infestations (7%)</b>: Two-spotted spider mite damage.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# TAB 5: SYSTEM SPECS & ATTRIBUTION
# ==========================================
with tab_about:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">ℹ️ Technical Specifications & System Architecture</div>', unsafe_allow_html=True)
    
    st.markdown("""
    | Component | Specification | Description |
    | :--- | :--- | :--- |
    | **Neural Architecture** | VGG16 CNN Backbone | Convolutional feature extractor pre-trained on PlantVillage dataset |
    | **Input Tensor** | `(1, 224, 224, 3)` | RGB image normalized to $[0.0, 1.0]$ float32 |
    | **Output Vector** | $\\mathbb{R}^{15}$ Softmax | Probability distribution over 15 solanaceous health categories |
    | **Validation Accuracy** | **95.2%** | Validated against stratified holdout crop image datasets |
    | **Inference Framework** | TensorFlow / Keras 2.x | Optimized float32 execution with layer caching |
    | **Web Dashboard** | Streamlit 1.40 + Plotly | Responsive layout with WebRTC camera and dynamic Plotly charts |
    | **Report Engine** | ReportLab 5.0 | Dynamic PDF generation with vector tables and prescription cards |
    | **Author** | **Nishant Rai** | [@Nishant052004](https://github.com/Nishant052004) |
    | **Source Code** | [GitHub Repository](https://github.com/Nishant052004/Plant-Diseases) | Open-source under MIT License |
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# SITE FOOTER
# ==========================================
st.markdown("""
<div class="site-footer">
    <p>🌿 <b>FloraScan AI</b> · Developed by <b>Nishant Rai</b> · Powered by TensorFlow & Streamlit</p>
    <p style="font-size: 11px; color: #475569; max-width: 600px; margin: 6px auto 0 auto;">
        Notice: Diagnostic outputs and treatment guides are intended for informational and advisory purposes. For commercial field interventions, consult certified agricultural extension specialists.
    </p>
</div>
""", unsafe_allow_html=True)