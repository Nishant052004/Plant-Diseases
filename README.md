# 🌱 AI Plant Disease Detection Dashboard

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Accuracy](https://img.shields.io/badge/Model%20Accuracy-95%25%2B-00FF99.svg)](#model-and-performance)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

An interactive computer vision dashboard built to help farmers, gardeners, and agronomists diagnose crop leaf diseases instantly from photos. Powered by a deep Convolutional Neural Network (CNN) and deployed with a clean Streamlit interface, this tool doesn't just name the disease — it gives you practical treatment steps and preventive measures to protect your harvest.

---

## 📌 Why This Project?

Early detection of crop diseases is critical. In many farming communities, getting an expert agronomist to visit a field can take days or weeks — by which time fungal blights or viral infections may have already devastated an entire crop yield.

This project bridges that gap by putting a diagnostic laboratory in your browser:
- **Instant diagnosis:** Upload a simple photo from your phone or camera and get predictions in milliseconds.
- **Actionable agronomic advice:** Every diagnosis comes paired with targeted descriptions, chemical/organic treatment guidelines, and cultural prevention practices.
- **Transparency:** The model provides confidence scores and top-3 probable predictions so users can make informed decisions rather than blindly trusting an output.

---

## ✨ Features

- **⚡ Real-Time Leaf Diagnosis:** Upload photos in JPG, JPEG, or PNG formats for immediate analysis.
- **📊 Multi-Class Probabilities:** View top-3 predicted diseases with visual probability bars to inspect borderline cases.
- **🛡️ Actionable Treatment & Prevention Guides:** Clear, practical recommendations for fungicides, bactericides, organic controls (like neem oil), and farm management practices.
- **🎯 Dynamic Confidence Indicator:**
  - **High Confidence (>90%):** Strong match with trained patterns.
  - **Moderate Confidence (70% - 90%):** Reliable, but warrants close inspection.
  - **Low Confidence (<70%):** Prompts the user to re-check photo quality or angle.
- **🔄 Interactive 5-Step AI Pipeline:** An educational visual walkthrough explaining how the image transforms from raw pixels to diagnosis.
- **📋 Searchable Disease Catalog:** Browse all 15 covered disease classes with real-time text search and category filtering (Fungal, Bacterial, Viral, Pest, Healthy).
- **🎨 Modern Dark Dashboard UI:** Designed with custom styling, high-contrast metrics, and responsive cards for a seamless user experience.

---

## 🌾 Supported Crops & Diseases (15 Classes)

The model covers three of the most widely grown vegetable crops across 15 distinct health states:

| Crop | Class Name | Category | Primary Symptoms / Characteristics |
| :--- | :--- | :--- | :--- |
| **Pepper (Bell)** | Bacterial Spot | Bacterial | Dark, water-soaked spots on leaves and fruit |
| **Pepper (Bell)** | Healthy | Healthy | Vibrant green foliage, normal growth |
| **Potato** | Early Blight (*Alternaria solani*) | Fungal | Dark concentric rings ("target" spots) on older leaves |
| **Potato** | Late Blight (*Phytophthora infestans*) | Fungal | Rapidly spreading dark lesions with white mold in humid conditions |
| **Potato** | Healthy | Healthy | Clean, vigorous leaf tissue |
| **Tomato** | Bacterial Spot | Bacterial | Small, dark brown-black specks with yellow halos |
| **Tomato** | Early Blight | Fungal | Brown concentric ring spots starting on lower foliage |
| **Tomato** | Late Blight | Fungal | Water-soaked patches causing rapid leaf wilting and collapse |
| **Tomato** | Leaf Mold | Fungal | Pale greenish-yellow spots turning into velvety olive mold |
| **Tomato** | Septoria Leaf Spot | Fungal | Small circular spots with grayish-white centers and dark borders |
| **Tomato** | Spider Mites | Pest Damage | Yellow stippling, fine webbing, and leaf bronzing |
| **Tomato** | Target Spot | Fungal | Concentric lesions resembling targets on leaves and stems |
| **Tomato** | Yellow Leaf Curl Virus | Viral | Upward leaf curling, stunting, and yellowing margins (transmitted by whiteflies) |
| **Tomato** | Mosaic Virus | Viral | Mottled light and dark green patterns, distorted leaves |
| **Tomato** | Healthy | Healthy | Intact foliage, balanced coloration |

### Category Breakdown
- **Fungal Diseases:** 7 classes
- **Healthy Leaves:** 3 classes
- **Viral Infections:** 2 classes
- **Bacterial Infections:** 2 classes
- **Pest Infestations:** 1 class

---

## 🧠 How It Works (The 5-Step Pipeline)

```
[ Upload Leaf Image ]
         │
         ▼
[ Preprocessing (224×224, Normalized /255) ]
         │
         ▼
[ Deep CNN Inference (VGG16 Backbone) ]
         │
         ▼
[ Argmax & Softmax Confidence Scoring ]
         │
         ▼
[ Diagnosis + Treatment & Prevention Advice ]
```

1. **Upload:** User provides a leaf image via the drag-and-drop file uploader.
2. **Preprocessing:** The image is resized to `224 × 224` pixels, converted to an RGB array, normalized by scaling pixel values to `[0, 1]`, and expanded into batch format `(1, 224, 224, 3)`.
3. **Inference:** A pre-trained VGG16-based Convolutional Neural Network extracts deep visual features across multiple convolutional and pooling blocks.
4. **Identification:** Softmax output yields probabilities across all 15 classes, selecting the highest-confidence match.
5. **Guidance:** The dashboard fetches verified treatment and prevention recommendations from `disease_info.py` and renders them on screen.

---

## 📁 Project Structure

```text
Plant-Diseases/
├── app.py                      # Main Streamlit web application & dashboard
├── disease_info.py             # Disease descriptions, treatments & prevention database
├── labels.txt                  # List of the 15 classification category labels
├── plant_disease_model.h5      # Trained deep learning CNN model weights
├── requirements.txt            # Python dependencies
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
- **Image Processing:** [Pillow (PIL)](https://python-pillow.org/) & [NumPy](https://numpy.org/)
- **Visualization:** [Matplotlib](https://matplotlib.org/)
- **Styling:** Custom dark theme CSS with Google Outfit typography

---

## 🔮 Roadmap & Future Improvements

- [ ] **Expand Crop Spectrum:** Add support for Apple, Corn, Grape, and Rice crops.
- [ ] **Grad-CAM Visual Heatmaps:** Highlight exactly which regions of the leaf triggered the disease prediction.
- [ ] **Mobile & Offline Deployment:** Convert the model to **TensorFlow Lite (`.tflite`)** for edge execution on low-cost smartphones without internet access.
- [ ] **Multilingual Support:** Add regional languages (Hindi, Spanish, etc.) so local farmers can understand treatment instructions directly.
- [ ] **Weather & Soil Integration:** Correlate local humidity and rainfall data with disease likelihood for proactive warnings.

---

## 👤 Author

**Nishant**
- GitHub: [@Nishant052004](https://github.com/Nishant052004)
- Repository: [Plant-Diseases](https://github.com/Nishant052004/Plant-Diseases)

If you find this project helpful, feel free to give it a ⭐ on GitHub!
