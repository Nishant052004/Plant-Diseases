# 🌿 FloraScan AI · Smart Plant Pathology & Crop Protection

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75.svg?logo=plotly&logoColor=white)](https://plotly.com/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

An interactive computer vision web application built to help farmers, gardeners, and agronomists diagnose crop leaf diseases instantly from photos. Powered by a deep Convolutional Neural Network (CNN) and deployed with a modern dark-mode Streamlit web dashboard, this tool doesn't just name the disease — it evaluates infection severity, displays interactive telemetry, provides practical treatments, and exports printable clinical PDF pathology reports.

---

## 📌 Why This Project?

Early detection of crop diseases is critical. In many farming communities, getting an expert agronomist to visit a field can take days or weeks — by which time fungal blights or viral infections may have already devastated an entire crop yield.

This project bridges that gap by putting a diagnostic laboratory in your browser:
- **Instant diagnosis:** Upload a leaf photo or snap one directly using your camera and get predictions in milliseconds.
- **1-Click sample testing:** Test the engine immediately using bundled high-resolution leaf samples without needing your own photos.
- **Actionable agronomic advice:** Every diagnosis comes paired with targeted descriptions, disease severity indicators, chemical/organic treatment guidelines, and cultural prevention practices.
- **Printable PDF reports:** Generate and download a formatted pathology report complete with timestamps, thumbnail, and treatment checklists.

---

## ✨ Features

- **⚡ Real-Time Leaf Diagnosis:** Upload photos in JPG, JPEG, or PNG formats or use the **live camera capture** mode.
- **🖼️ 1-Click Sample Gallery:** Pre-loaded samples (Tomato Early Blight, Potato Late Blight, Pepper Spot, Healthy Tomato) to test the app with a single click.
- **📊 Interactive Plotly Telemetry:** Real-time horizontal bar charts and distribution donut charts replacing static images.
- **📑 Clinical PDF Prescription Export:** Generates vector PDF diagnostic summaries with one click via ReportLab.
- **🛡️ Actionable Treatment & Prevention Guides:** Clear, practical recommendations for fungicides, bactericides, organic controls (like neem oil), and farm management practices.
- **🎯 Dynamic Confidence Indicator:**
  - **High Confidence (>90%):** Strong match with trained patterns.
  - **Moderate Confidence (70% - 90%):** Reliable, but warrants close inspection.
  - **Low Confidence (<70%):** Prompts the user to re-check photo quality or angle.
- **🔄 Interactive 5-Step AI Pipeline:** An educational visual walkthrough explaining how the image transforms from raw pixels to diagnosis.
- **📋 Searchable Disease Catalog:** Browse all 15 covered disease classes with real-time text search and category filtering (Fungal, Bacterial, Viral, Pest, Healthy).
- **🎨 Modern Glassmorphic Web UI:** Styled with frosted glass navbar, responsive KPI metrics, and high-contrast clinical tags.

---

## 🌾 Supported Crops & Diseases (15 Classes)

The model covers three of the most widely grown vegetable crops across 15 distinct health states:

| Crop | Class Name | Category | Severity | Primary Symptoms / Characteristics |
| :--- | :--- | :--- | :--- | :--- |
| **Pepper (Bell)** | Bacterial Spot | Bacterial | High | Dark, water-soaked spots with yellow halos |
| **Pepper (Bell)** | Healthy | Healthy | None | Vibrant green foliage, normal growth |
| **Potato** | Early Blight (*Alternaria solani*) | Fungal | Moderate | Dark concentric rings ("target" spots) on older leaves |
| **Potato** | Late Blight (*Phytophthora infestans*) | Fungal | Critical | Rapidly spreading dark lesions with white mold in humid conditions |
| **Potato** | Healthy | Healthy | None | Clean, vigorous leaf tissue |
| **Tomato** | Bacterial Spot | Bacterial | High | Small, dark brown-black specks with yellow halos |
| **Tomato** | Early Blight | Fungal | Moderate | Brown concentric ring spots starting on lower foliage |
| **Tomato** | Late Blight | Fungal | Critical | Water-soaked patches causing rapid leaf wilting and collapse |
| **Tomato** | Leaf Mold | Fungal | Moderate | Pale greenish-yellow spots turning into velvety olive mold |
| **Tomato** | Septoria Leaf Spot | Fungal | Moderate | Small circular spots with grayish-white centers and dark borders |
| **Tomato** | Spider Mites | Pest Damage | Moderate | Yellow stippling, fine webbing, and leaf bronzing |
| **Tomato** | Target Spot | Fungal | Moderate | Concentric lesions resembling targets on leaves and stems |
| **Tomato** | Yellow Leaf Curl Virus | Viral | High | Upward leaf curling, stunting, and yellowing margins (transmitted by whiteflies) |
| **Tomato** | Mosaic Virus | Viral | Critical | Mottled light and dark green patterns, distorted leaves |
| **Tomato** | Healthy | Healthy | None | Intact foliage, balanced coloration |

### Category Breakdown
- **Fungal Diseases:** 7 classes (47%)
- **Healthy Leaves:** 3 classes (20%)
- **Viral Infections:** 2 classes (13%)
- **Bacterial Infections:** 2 classes (13%)
- **Pest Infestations:** 1 class (7%)

---

## 🧠 How It Works (The 5-Step Pipeline)

```
[ Upload / Camera Leaf Photo ]
               │
               ▼
[ Bilinear Preprocessing (224×224, Normalized /255) ]
               │
               ▼
[ Deep CNN Feature Extraction (VGG16 Backbone) ]
               │
               ▼
[ Argmax & Softmax Confidence Scoring ]
               │
               ▼
[ Clinical Advisory + Printable PDF Dispatch ]
```

1. **Acquisition:** User provides a leaf image via drag-and-drop, camera capture, or the 1-click sample gallery.
2. **Preprocessing:** Resized to `224 × 224` pixels, converted to RGB tensor, normalized to $[0, 1]$, and expanded into batch format `(1, 224, 224, 3)`.
3. **Inference:** Pre-trained VGG16 CNN extracts deep visual spatial hierarchies.
4. **Identification:** Softmax output yields probability distribution across 15 target categories.
5. **Guidance & Export:** Verified treatments rendered on screen, with a one-click downloadable PDF report.

---

## 📁 Project Structure

```text
Plant-Diseases/
├── app.py                      # Main Streamlit web application & modern dashboard
├── disease_info.py             # Disease descriptions, severity, treatments & prevention
├── report_generator.py         # PDF prescription and diagnostic report generator
├── labels.txt                  # List of the 15 classification category labels
├── plant_disease_model.h5      # Trained deep learning CNN model weights
├── requirements.txt            # Python dependencies (Streamlit, TF, Plotly, ReportLab)
├── samples/                    # Pre-loaded sample leaf photos for instant testing
│   ├── pepper_bacterial_spot.jpg
│   ├── potato_late_blight.jpg
│   ├── tomato_early_blight.jpg
│   └── tomato_healthy.jpg
├── PlantDisease_LinkedIn.pptx  # Project presentation deck & architecture slides
├── .gitignore                  # Git exclusion rules
└── README.md                   # Project documentation
```

---

## 🚀 Getting Started

Follow these steps to run the application on your local machine.

### 1. Prerequisites
- Python 3.9, 3.10, or 3.11 installed.
- Git installed on your system.

### 2. Clone the Repository
```bash
git clone https://github.com/Nishant052004/Plant-Diseases.git
cd Plant-Diseases
```

### 3. Create a Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Launch the Dashboard
```bash
streamlit run app.py
```

Once started, open your browser and navigate to:
```
http://localhost:8501
```

---

## 🛠️ Tech Stack & Tools

- **Core Language:** [Python](https://www.python.org/)
- **Deep Learning Framework:** [TensorFlow / Keras](https://www.tensorflow.org/) (VGG16 CNN Architecture)
- **Web Application & UI:** [Streamlit](https://streamlit.io/)
- **Interactive Visualizations:** [Plotly](https://plotly.com/)
- **PDF Report Generation:** [ReportLab](https://www.reportlab.com/)
- **Image Processing:** [Pillow (PIL)](https://python-pillow.org/) & [NumPy](https://numpy.org/)
- **Styling:** Custom glassmorphism CSS with Google Outfit & JetBrains Mono typography

---

## 🔮 Roadmap & Future Improvements

- [ ] **Expand Crop Spectrum:** Add support for Apple, Corn, Grape, and Rice crops.
- [ ] **Grad-CAM Visual Heatmaps:** Highlight exactly which regions of the leaf triggered the disease prediction.
- [ ] **Mobile & Offline Deployment:** Convert the model to **TensorFlow Lite (`.tflite`)** for edge execution on low-cost smartphones without internet access.
- [ ] **Multilingual Support:** Add regional languages (Hindi, Spanish, etc.) so local farmers can understand treatment instructions directly.
- [ ] **Weather & Soil Integration:** Correlate local humidity and rainfall data with disease likelihood for proactive warnings.

---

## 👤 Author

**Nishant Rai**
- GitHub: [@Nishant052004](https://github.com/Nishant052004)
- Repository: [Plant-Diseases](https://github.com/Nishant052004/Plant-Diseases)

If you find this project helpful, feel free to give it a ⭐ on GitHub!
