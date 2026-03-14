from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from src.predict import predict_image

app = Flask(__name__)

# Allow React frontend to connect
CORS(app, resources={r"/*": {"origins": "*"}})

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ---------------------------
# ROOT ROUTE (for testing)
# ---------------------------
@app.route("/")
def home():
    return "Brain Tumor Detection Backend Running ✅"


# ---------------------------
# PREDICT ROUTE
# ---------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        predicted_class, confidence, probabilities, gradcam_path, binary_prediction, binary_label = predict_image(
            filepath)

        return jsonify({
            "success": True,
            "prediction": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities,
            "binary_prediction": binary_prediction,
            "binary_label": binary_label,
            "gradcam": None
        })

    except Exception as e:
        print("🔥 ERROR:", e)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ---------------------------
# SERVE UPLOADED FILES
# ---------------------------
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


# ---------------------------
# START SERVER
# ---------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
