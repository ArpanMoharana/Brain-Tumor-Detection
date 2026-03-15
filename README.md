# 🧠 Brain Tumor Detection System

An **AI-powered web application** that detects brain tumors from MRI scans using **Deep Learning with MobileNetV2 Transfer Learning**.

Users can upload MRI images and the system will classify them into different tumor types while showing prediction confidence and probability distribution.

---

# 📂 Project Structure

After cloning the repository, the folder structure should look like this:

```
Brain-Tumor-Detection
│
├── backend
│   ├── app.py
│   ├── train_model.py
│   ├── requirements.txt
│   │
│   ├── models
│   │   └── brain_tumor_model.h5
│   │
│   └── src
│       └── predict.py
│
├── frontend
│   ├── public
│   │   └── confusion_matrix.png
│   │
│   ├── src
│   │   └── App.js
│   │
│   ├── package.json
│   └── package-lock.json
│
├── README.md
├── LICENSE
└── .gitignore
```

---

# 📥 Clone the Repository

Clone the project using Git:

```
git clone https://github.com/ArpanMoharana/Brain-Tumor-Detection.git
```

Navigate into the project folder:

```
cd Brain-Tumor-Detection
```

---

# 📦 Download Dataset (Required)

The dataset is **not included in this repository** because large datasets exceed GitHub storage limits.

Download the dataset from Kaggle:

https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Download and extract the dataset.

---

# 📁 Add Dataset to the Project

After extracting the dataset, create this folder inside the project:

```
dataset
```

Place the dataset like this:

```
Brain-Tumor-Detection
│
├── dataset
│   ├── train
│   │   ├── glioma
│   │   ├── meningioma
│   │   ├── pituitary
│   │   └── notumor
│   │
│   └── test
│       ├── glioma
│       ├── meningioma
│       ├── pituitary
│       └── notumor
```

The final paths should be:

```
Brain-Tumor-Detection/dataset/train
Brain-Tumor-Detection/dataset/test
```

---

# 🚀 Running the Project

The project contains two parts:

• Backend (Flask + AI Model)
• Frontend (React Interface)

Both must be started separately.

---

# 🧠 Backend Setup

Navigate to backend folder:

```
cd backend
```
Then Run the following code for respective OS
---

## 💻 Mac / Linux

Create virtual environment:

```
python3 -m venv venv
```

Activate environment:

```
source venv/bin/activate
```

Install dependencies:

```
pip install -r requirements.txt
```

Run the backend:

```
python3 app.py
```

Backend will run on:

```
http://127.0.0.1:5000
```

---

## 🪟 Windows

Create virtual environment:

```
python -m venv venv
```

Activate environment:

```
venv\Scripts\activate
```

Install dependencies:

```
pip install -r requirements.txt
```

Run backend:

```
python app.py
```

Backend will run on:

```
http://127.0.0.1:5000
```

---

# 🌐 Frontend Setup

Open a **new terminal** and navigate to the frontend folder:

```
cd frontend
```

Install dependencies:

```
npm install
```

Start the React app:

```
npm start
```

Frontend runs at:

```
http://localhost:3000
```

---

# 🧪 Using the Application

1. Open the web interface
2. Upload an MRI image
3. Click **Analyze MRI**
4. The AI model predicts tumor type
5. View probability bars and confidence score
6. Download AI report (PDF)

---

# 🧠 About the Project

This project implements a **Deep Learning based medical image classification system** for detecting brain tumors from MRI scans.

The system uses **Transfer Learning with MobileNetV2** to achieve high accuracy while keeping the model lightweight.

The project demonstrates a **complete AI pipeline**:

MRI Image → Preprocessing → Deep Learning Model → Prediction API → Web Interface

---

# 🤖 AI Model

Model Architecture:

```
Input Image (224x224x3)
↓
MobileNetV2 (Pretrained on ImageNet)
↓
GlobalAveragePooling2D
↓
Dense Layer (128 units)
↓
Dropout
↓
Softmax Output Layer
```

---

# 📊 Classification Classes

| Class      | Description                |
| ---------- | -------------------------- |
| Glioma     | Tumor in brain glial cells |
| Meningioma | Tumor in brain membranes   |
| Pituitary  | Tumor in pituitary gland   |
| No Tumor   | Healthy MRI                |

Binary classification mapping:

```
0 → No Tumor
1 → Tumor
```

---

# ⚙️ Tech Stack

Machine Learning:

* TensorFlow
* Keras
* MobileNetV2
* NumPy
* OpenCV
* Scikit-learn

Backend:

* Python
* Flask
* Flask-CORS

Frontend:

* React.js
* JavaScript
* CSS

Visualization:

* Matplotlib
* Seaborn

---

# 📈 Model Performance

| Metric       | Value       |
| ------------ | ----------- |
| Accuracy     | ~94%        |
| Architecture | MobileNetV2 |
| Classes      | 4           |
| Input Size   | 224 × 224   |

---

# ⚠️ Important Notes

The following folders are **not included in the repository**:

```
dataset
venv
node_modules
uploads
```

They must be created locally.

---

# 📜 License

This project is licensed under the  **MIT License**.
