# 🧠 Dual-Brain Brain Tumor Detection System

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

## 📂 Project Structure

```
Brain-Tumor-Detection/
│
├── backend/
│   ├── app.py                        # Flask API server
│   ├── train_model.py                # CNN training script (local)
│   ├── requirements.txt
│   │
│   ├── models/                       # Model weights (download separately)
│   │   ├── .gitkeep
│   │   ├── brain_tumor_model.h5      # Retrained MobileNetV2
│   │   ├── gnn_tumor_model_smart.pth # Trained 3-layer GNN
│   │   └── sam_vit_h.pth             # Meta SAM ViT-H
│   │
│   ├── src/
│   │   └── predict.py                # Core 7-stage pipeline engine
│   │
│   └── uploads/                      # Temp storage for user images
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── App.js                    # React clinical dashboard
│   │   └── App.css                  # Dark theme stylesheet
│   ├── package.json
│   └── package-lock.json
│
├── research/                         # Colab training notebooks
│   ├── CNN_Retrain_BrainTumor.ipynb
│   ├── CNN_BoostGlioma.ipynb
│   └── GNN_Train_Real_Data.ipynb
│
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

Model files are **not included in this repository** due to size limits. Download and place them in `backend/models/`:

| File | Size | Source |
|------|------|--------|
| `brain_tumor_model.h5` | ~14 MB | Present with Code |
| `gnn_tumor_model_smart.pth` | ~1 MB | Present with Code |
| `sam_vit_h.pth` | 2.4 GB | Need To download [Meta AI](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) |

```bash
# Download SAM directly
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O backend/models/sam_vit_h.pth
```

---

## 🤖 Retrain Models (Optional — Colab)

If you want to retrain from scratch, use the notebooks in `research/`. They require a Kaggle API key (`kaggle.json`).

**Step 1 — Retrain CNN:**
Open `research/CNN_Retrain_BrainTumor.ipynb` in Google Colab (T4 GPU).
Upload `kaggle.json`, run all cells. Downloads Msoud dataset automatically.

**Step 2 — Boost Glioma recall:**
Open `research/CNN_BoostGlioma.ipynb` in the same Colab session.
Loads `best_p2.keras` and fine-tunes with focal loss + doubled Glioma weight.

**Step 3 — Train GNN:**
Open `research/GNN_Train_Real_Data.ipynb` in Google Colab.
Builds ~4760 superpixel graphs from real MRI images and trains the GNN.

After training, download and place `.h5` and `.pth` files into `backend/models/`.

---

## 🚀 Running the Project

Both backend and frontend must run simultaneously in separate terminals.

---

### 🧠 Backend Setup

```bash
cd backend
```

**Mac / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 app.py
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Backend runs at: `http://127.0.0.1:5000`

---

### 🌐 Frontend Setup

Open a new terminal:

```bash
cd frontend
npm install
npm start
```

Frontend runs at: `http://localhost:3000`

---

## 🧪 Using the Application

1. Open `http://localhost:3000`
2. Upload an MRI image (JPEG/PNG)
3. Click **Analyze MRI**
4. View all 6 pipeline stages as they process
5. Read the final diagnosis, confidence score, and per-class probabilities
6. Download the PDF report

---

## 🤖 Retrain Models (Optional — Colab)

If you want to retrain from scratch, use the notebooks in `research/`. They require a Kaggle API key (`kaggle.json`).

**Step 1 — Retrain CNN:**
Open `research/CNN_Retrain_BrainTumor.ipynb` in Google Colab (T4 GPU).
Upload `kaggle.json`, run all cells. Downloads Msoud dataset automatically.

**Step 2 — Boost Glioma recall:**
Open `research/CNN_BoostGlioma.ipynb` in the same Colab session.
Loads `best_p2.keras` and fine-tunes with focal loss + doubled Glioma weight.

**Step 3 — Train GNN:**
Open `research/GNN_Train_Real_Data.ipynb` in Google Colab.
Builds ~4760 superpixel graphs from real MRI images and trains the GNN.

After training, download and place `.h5` and `.pth` files into `backend/models/`.

---

## 🔬 The 7-Stage Pipeline

| Stage | Name | Description |
|-------|------|-------------|
| 1 | Pre-processing | ROI crop, denoise, resize to 224×224 |
| 2 | SAM segmentation | Meta SAM generates initial tumor mask with skull stripping |
| 3 | Superpixel (SLIC) | 200 organic superpixel regions with color overlay |
| 4 | Graph construction | Superpixel RAG graph with 16 regionprop node features |
| 5 | Refined segmentation | SAM mask applied as red overlay on MRI |
| 6 | GNN inference | 3-layer GCNConv classifies tumor by shape geometry |
| 7 | Grad-CAM overlay | CNN attention heatmap with SAM target point |

---

## 📊 Tumor Classes

| Index | Class | Description |
|-------|-------|-------------|
| 0 | Glioma | Tumor in brain glial cells — irregular, infiltrating |
| 1 | Meningioma | Tumor in brain membranes — near skull surface |
| 2 | No Tumor | Healthy MRI scan |
| 3 | Pituitary | Tumor in pituitary gland — lower/center brain |

---

## 📈 Model Performance

| Model | Architecture | Dataset | Test Accuracy |
|-------|-------------|---------|---------------|
| CNN | MobileNetV2 (fine-tuned) | Msoud Brain Tumor MRI | 88% |
| GNN | 3-layer GCNConv | Msoud (graph features) | ~75% |

**Per-class CNN recall:**

| Class | Recall |
|-------|--------|
| Glioma | ~0.75+ (after focal loss boost) |
| Meningioma | 0.82 |
| No Tumor | 0.99 |
| Pituitary | 0.99 |

---

## ⚙️ Tech Stack

**AI / ML**
- TensorFlow 2.15 · Keras · MobileNetV2
- PyTorch · PyTorch Geometric · GCNConv
- Meta Segment Anything Model (SAM ViT-H)
- scikit-image (SLIC superpixels · regionprops)
- OpenCV · NumPy · NetworkX

**Backend**
- Python · Flask · Flask-CORS

**Frontend**
- React.js · CSS · jsPDF

**Training**
- Google Colab (T4 GPU)
- Kaggle Msoud Brain Tumor MRI dataset

---

## 📜 License

This project is licensed under the **MIT License**.
