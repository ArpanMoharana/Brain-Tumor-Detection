# 🧠 Brain Tumor Detection System

An **AI-powered web application** that detects brain tumors from MRI scans using **Deep Learning with MobileNetV2 Transfer Learning**.

The system allows users to upload MRI images and receive **tumor classification predictions, confidence scores, and probability visualizations** through an interactive web interface.

This project demonstrates an **end-to-end AI system** combining:

* Computer Vision
* Deep Learning
* Backend API development
* Frontend visualization

---

# 📌 Features

* MRI image upload and analysis
* Multi-class tumor classification
* Binary classification (Tumor / No Tumor)
* Probability visualization using dynamic bars
* Confusion matrix visualization
* Downloadable AI diagnostic report (PDF)
* Full-stack architecture (Flask + React)

---

# 🧠 AI Model

The classification model uses **Transfer Learning with MobileNetV2** to detect brain tumors from MRI scans.

MobileNetV2 is a lightweight convolutional neural network optimized for image classification and mobile devices.

## Model Pipeline

MRI Image
↓
Image Preprocessing
↓
MobileNetV2 Feature Extraction
↓
Dense Layers
↓
Softmax Classification

---

## Model Architecture

Input Image: **224 × 224 × 3**

```
MobileNetV2 (Pretrained on ImageNet)
↓
GlobalAveragePooling2D
↓
Dense (128 units, ReLU)
↓
Dropout (0.5)
↓
Dense (4 units, Softmax)
```

---

## Classes

| Class      | Description                |
| ---------- | -------------------------- |
| Glioma     | Tumor in glial brain cells |
| Meningioma | Tumor in brain membranes   |
| Pituitary  | Tumor in pituitary gland   |
| No Tumor   | Healthy MRI                |

---

## Binary Classification Mapping

Binary output is derived from multiclass predictions:

| Binary Label | Meaning                                 |
| ------------ | --------------------------------------- |
| 0            | No Tumor                                |
| 1            | Tumor (Glioma / Meningioma / Pituitary) |

---

# ⚙️ Tech Stack

## Machine Learning

* TensorFlow
* Keras
* MobileNetV2
* NumPy
* OpenCV
* Scikit-learn

## Backend

* Python
* Flask
* Flask-CORS

## Frontend

* React.js
* JavaScript
* CSS

## Visualization

* Seaborn
* Matplotlib

## Other Tools

* Git
* GitHub
* GitHub Desktop
* jsPDF (PDF report generation)

---

# 📂 Project Structure

```
Brain-Tumor-Detection
│
├── backend
│   ├── app.py                # Flask API server
│   ├── train_model.py        # Model training script
│   ├── requirements.txt      # Python dependencies
│   │
│   ├── models
│   │   └── brain_tumor_model.h5
│   │
│   └── src
│       └── predict.py        # Prediction logic
│
├── frontend
│   ├── public
│   │   └── confusion_matrix.png
│   │
│   ├── src
│   │   └── App.js            # React interface
│   │
│   ├── package.json
│   └── package-lock.json
│
├── README.md
├── LICENSE
└── .gitignore
```

---

# 📊 Dataset

Dataset used:

**Brain MRI Images for Brain Tumor Detection**

Source:

https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Dataset classes:

* Glioma
* Meningioma
* Pituitary
* No Tumor

Dataset structure used during training:

```
dataset
│
├── train
│   ├── glioma
│   ├── meningioma
│   ├── pituitary
│   └── notumor
│
└── test
    ├── glioma
    ├── meningioma
    ├── pituitary
    └── notumor
```

---

# 📈 Model Performance

| Metric       | Value       |
| ------------ | ----------- |
| Accuracy     | ~94%        |
| Architecture | MobileNetV2 |
| Input Size   | 224 × 224   |
| Classes      | 4           |

Model performance is visualized using a **Confusion Matrix**.

---

# 🚀 Installation & Running the Project

Follow the steps below to run the project locally.

---

# 1️⃣ Clone the Repository

```
git clone https://github.com/ArpanMoharana/Brain-Tumor-Detection.git
```

```
cd Brain-Tumor-Detection
```

---

# 🧠 Backend Setup (Flask + AI Model)

Navigate to backend folder:

```
cd backend
```

---

## 🖥️ Mac / Linux

### Create Virtual Environment

```
python3 -m venv venv
```

### Activate Virtual Environment

```
source venv/bin/activate
```

### Install Dependencies

```
pip install -r requirements.txt
```

### Run Backend Server

```
python3 app.py
```

Backend runs on:

```
http://127.0.0.1:5000
```

---

## 🪟 Windows

### Create Virtual Environment

```
python -m venv venv
```

### Activate Virtual Environment

```
venv\Scripts\activate
```

### Install Dependencies

```
pip install -r requirements.txt
```

### Run Backend Server

```
python app.py
```

Backend runs on:

```
http://127.0.0.1:5000
```

---

# 🌐 Frontend Setup (React)

Open a new terminal and navigate to the frontend folder:

```
cd frontend
```

Install dependencies:

```
npm install
```

Run the React application:

```
npm start
```

Frontend runs on:

```
http://localhost:3000
```

---

# 🧪 Using the Application

1. Open the React interface in your browser:

```
http://localhost:3000
```

2. Upload an MRI image

3. Click **Analyze MRI**

4. The system will display:

* Predicted tumor class
* Confidence score
* Class probabilities
* Confusion matrix
* AI report download option

---

# ⚠️ Important Notes

The following folders are **not included in the repository** and should not be uploaded:

```
venv
node_modules
dataset
uploads
```

These are generated locally and can be recreated.

---

# 📜 License

This project is licensed under the **MIT License**.
