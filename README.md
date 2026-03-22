# 🧠 Brain Tumor Detection System
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)

An **AI-powered web application** that detects brain tumors from MRI scans using a **multi-model ensemble architecture** combining CNN texture analysis, SAM segmentation, and GNN geometric reasoning.

Users upload MRI images through a clinical dark-theme web interface. The system runs a **7-stage pipeline** and returns the tumor classification with visual evidence at every stage.

---

## 🏗️ Architecture Overview

```
MRI Image
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                  Pipeline                               │
│                                                         │
│  ┌──────────────┐        ┌────────────────────────────┐ │
│  │  CNN Branch  │        │      SAM + GNN Branch      │ │
│  │ MobileNetV2  │        │  Segment Anything Model    │ │
│  │  (Texture)   │        │  + Graph Neural Network    │ │
│  │              │        │     (Geometry/Shape)       │ │
│  └──────┬───────┘        └────────────┬───────────────┘ │
│         │                             │                 │
│         ▼                             ▼                 │
│  ┌──────────────────────────────────────────────────┐   │
│  │   Clinical Geometry Veto System                  │   │
│  │   Spatial zone boosts · No-Tumor suppression     │   │
│  └──────────────────────────────────────────────────┘   │
│                          │                              │
│              CNN 60% + GNN 40%                          │
└─────────────────────────────────────────────────────────┘
    │
    ▼
Final Diagnosis + 6 Visual Pipeline Stages
```

---

```
## 📂 Project Structure

```text
Brain-Tumor-Detection/
│
├── backend/
│   ├── app.py
│   ├── train_model.py
│   ├── download_sam.py
│   ├── requirements.txt
│   │
│   ├── models/
│   │   ├── brain_tumor_model.h5
│   │   ├── gnn_tumor_model_smart.pth
│   │   └── sam_vit_h_4b8939.pth
│   │
│   ├── src/
│   │   └── predict.py
│   │
│   └── uploads/
│
├── frontend/
│
├── dataset/
│   ├── train/
│   └── test/
│
├── README.md
└── .gitignore

```
---

## 📥 Clone the Repository

```bash
git clone https://github.com/ArpanMoharana/Brain-Tumor-Detection.git
cd Brain-Tumor-Detection
```

---



---

## 📦 Download Dataset (Required)

Dataset:

https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

---

## 📁 Add Dataset to the Project

```
dataset/
├── train/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── notumor/
│
└── test/
    ├── glioma/
    ├── meningioma/
    ├── pituitary/
    └── notumor/
```



---

## 🧠 How to Train the Model (Local Training)

### Step 1

The project includes a local training script:

```
backend/train_model.py
```


---

### Step 2 — Activate Environment

# Mac/Linux
```bash
cd backend


python3 -m venv venv
```
```
source venv/bin/activate
```

# Windows
```
cd backend

venv\Scripts\activate
```


---

### Step 3 — Run Training

## FOR MAC

```bash
python3 train_model.py
```

## FOR WINDOWS

```bash
python train_model.py
```

---

### Step 4 — Training Process

* Loads MRI images
* Resizes to 224×224
* Normalizes pixel values
* Trains MobileNetV2 CNN
* Validates performance

---

### Step 5 — Output

Model saved at:

```
backend/models/brain_tumor_model.h5
```

---

### Notes

* Takes 10–30 minutes
* CPU works, GPU recommended
* Overwrites existing model

---
# 📦 Download SAM Model (REQUIRED)

- You have 2 options to download the Segment Anything (SAM) weights:

## 🔹 Option 1: Automatic (Recommended)
```
cd backend
python3 download_sam.py
```
## 🔹 Option 2: Manual Download
```
cd backend/models
wget [https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
```
---

## 🚀 Running the Project

### Backend

```bash
cd backend
source venv/bin/activate
pip install -r requirements.txt
python3 app.py
```

---

### Frontend

```bash
cd frontend
npm install
npm start
```

---

## 🧪 How to Use
- Upload an MRI image via the web interface.

- Click Analyze.

- View the detection results, class predictions, and visual overlays.

- Click Download Report to save a PDF of the results.

---

## 🔬 7-Stage Pipeline

| Stage | Description      |
| ----- | ---------------- |
| 1     | Preprocessing    |
| 2     | SAM Segmentation |
| 3     | Superpixels      |
| 4     | Graph            |
| 5     | Mask             |
| 6     | GNN              |
| 7     | GradCAM          |

---

## 📊 Classes

* Glioma
* Meningioma
* No Tumor
* Pituitary

---

## ⚙️ Tech Stack

* TensorFlow / Keras
* PyTorch / GNN
* SAM
* OpenCV
* React
* Flask

---

## Notes!
⚠️ Training is mandatory before running the application.

⚠️ The SAM model must be downloaded before running inference.

💡 Hardware: CPU works, but a GPU is highly recommended for faster performance.

---


# 💻 System Requirements

### Minimum Requirements
- OS: Windows / macOS / Linux
- RAM: 8 GB
- Storage: 5–10 GB free space
- Python: 3.9 – 3.11
- Node.js: v16+

---

### Recommended (for better performance)

- RAM: 16 GB
- GPU: NVIDIA GPU (for faster training/inference)
- CUDA (optional, for GPU acceleration)

---

### Notes
- CPU execution is fully supported (default)
- Mac users can use **MPS acceleration**
- GPU is **not mandatory**, but speeds up training significantly
- SAM model (~2.5GB) requires stable internet for first download

---

## 📜 License

MIT License
