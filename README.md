# 🧠 Brain Tumor Detection System

An **AI-powered web application** that detects brain tumors from MRI scans using **Deep Learning with MobileNetV2 Transfer Learning**.

Users can upload MRI images through a web interface and the system will classify them into different tumor types while displaying prediction confidence and probability distribution.

This project demonstrates a **complete AI pipeline** combining:

* Computer Vision
* Deep Learning
* Backend API development
* Frontend visualization

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
├── dataset
│   ├── train
│   └── test
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

# 📦 Download Dataset

The dataset is **not included in this repository** because large datasets exceed GitHub storage limits.

Download the dataset from Kaggle:

https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Extract the dataset.

---

# 📁 Add Dataset to the Project

Place the extracted dataset inside the project like this:

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

Final paths should be:

```
Brain-Tumor-Detection/dataset/train
Brain-Tumor-Detection/dataset/test
```

---

# 🤖 Train the Model (Required if model file is missing/Untrained- one time only)

If the file below does not exist:

```
backend/models/brain_tumor_model.h5
```

You must train the model first(one time only).

Navigate to the backend folder:

### Mac / Linux

```
cd backend
python3 train_model.py
```

### Windows

```
cd backend
python train_model.py
```

After training completes, the model will be saved as:

```
backend/models/brain_tumor_model.h5
```

---

# 🚀 Running the Project

The project has two parts:

• Backend (Flask + AI Model)
• Frontend (React Interface)

Both must be started separately.

---

# 🧠 Backend Setup

Navigate to backend folder:

```
cd backend
```
Then Run the following commands for respective OS :-

---

## Mac / Linux

Create virtual environment

```
python3 -m venv venv
```

Activate environment

```
source venv/bin/activate
```

Install dependencies

```
pip install -r requirements.txt
```

Run backend server

```
python3 app.py
```

Backend runs on:

```
http://127.0.0.1:5000
```

---

## Windows

Create virtual environment

```
python -m venv venv
```

Activate environment

```
venv\Scripts\activate
```

Install dependencies

```
pip install -r requirements.txt
```

Run backend

```
python app.py
```

Backend runs on:

```
http://127.0.0.1:5000
```

---

# 🌐 Frontend Setup

Open another terminal and navigate to frontend:

```
cd frontend
```

Install dependencies:

```
npm install
```

Run React application:

```
npm start
```

Frontend runs at:

```
http://localhost:3000
```

---

# 🧪 Using the Application

1. Open the React interface
2. Upload an MRI image
3. Click **Analyze MRI**
4. The AI model predicts tumor type
5. View probability bars and confidence score
6. Download AI report (PDF)

---

# 🧠 About the Project

This project implements a **Deep Learning based medical image classification system** for detecting brain tumors from MRI scans.

The model uses **Transfer Learning with MobileNetV2**, a lightweight CNN architecture pretrained on ImageNet, which allows high accuracy with limited medical data.

The application demonstrates an **end-to-end AI deployment pipeline**:

MRI Image → Preprocessing → Deep Learning Model → Prediction API → Web Interface

---

# 📊 Tumor Classes

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

Machine Learning

* TensorFlow
* Keras
* MobileNetV2
* NumPy
* OpenCV
* Scikit-learn

Backend

* Python
* Flask
* Flask-CORS

Frontend

* React.js
* JavaScript
* CSS

Visualization

* Matplotlib
* Seaborn

---

# 📈 Model Performance

| Metric       | Value       |
| ------------ | ----------- |
| Accuracy     | ~94%        |
| Architecture | MobileNetV2 |
| Input Size   | 224 × 224   |
| Classes      | 4           |

---

# ⚠️ Important Notes

The following folders are **not included in the repository** and must be created locally:

```
dataset
venv
node_modules
uploads
```


---

# 📜 License

This project is licensed under the **MIT License**.
