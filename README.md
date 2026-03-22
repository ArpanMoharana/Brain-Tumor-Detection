# 🧠 Brain Tumor Detection System

An **AI-powered web application** that detects brain tumors from MRI scans using a **multi-model ensemble architecture** combining CNN texture analysis, SAM segmentation, and GNN geometric reasoning.

Users upload MRI images through a clinical dark-theme web interface. The system runs a **7-stage pipeline** and returns the tumor classification with visual evidence at every stage.

---

## 🏗️ Architecture Overview

```
MRI Image
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                  Dual-Brain Pipeline                    │
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


Brain-Tumor-Detection/
│
├── backend/
│   ├── app.py
│   ├── train_model.py
│   ├── requirements.txt
│   │
│   ├── models/
│   │   ├── .gitkeep
│   │   ├── brain_tumor_model.h5
│   │   ├── gnn_tumor_model_smart.pth
│   │   └── sam_vit_h.pth
│   │
│   ├── src/
│   │   └── predict.py
│   │
│   └── uploads/
│
├── frontend/
│
├── dataset/                  ✅ ← ADD THIS FOLDER
│   ├── train/
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── pituitary/
│   │   └── notumor/
│   │
│   └── test/
│       ├── glioma/
│       ├── meningioma/
│       ├── pituitary/
│       └── notumor/
│
├── research/
├── README.md
├── LICENSE
└── .gitignore

```

---

## 📥 Clone the Repository

```bash
git clone https://github.com/ArpanMoharana/Brain-Tumor-Detection.git
cd Brain-Tumor-Detection
```

---

## 📦 Download Model Weights

Model files are **not included in this repository** due to size limits.

| File                      | Size   | Source            |
| ------------------------- | ------ | ----------------- |
| brain_tumor_model.h5      | ~14 MB | Present with Code |
| gnn_tumor_model_smart.pth | ~1 MB  | Present with Code |
| `sam_vit_h.pth`           | 2.4 GB | [Meta AI](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) 

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O backend/models/sam_vit_h.pth
```

---

## 📦 Download Dataset (Required)

Dataset:

https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

---

## 📁 Add Dataset to the Project

```
dataset/
├── train/
├── test/
```

Each contains:

```
glioma/
meningioma/
pituitary/
notumor/
```

---

## 🧠 How to Train the Model (Local Training)

The project includes a local training script:

```
backend/train_model.py
```

### Step 1 — Prepare Dataset

```
Brain-Tumor-Detection/dataset/train
Brain-Tumor-Detection/dataset/test
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

```bash
python3 train_model.py
```

(Windows)

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

## 🧪 Usage

1. Upload MRI
2. Click Analyze
3. View stages
4. Get prediction

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

## 📜 License

MIT License
