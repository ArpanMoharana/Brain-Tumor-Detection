from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from src.predict import predict_image

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return "Brain Tumor Detection Backend Running ✅"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"success": False, "error": "No image uploaded"}), 400

        file = request.files["image"]
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Get the structured data from the 9-stage pipeline
        result_data = predict_image(filepath)

        # Convert filenames into full URLs for the React frontend
        base_url = "http://127.0.0.1:5000/uploads/"
        result_data["stage_images"] = {
            k: base_url + v for k, v in result_data["stage_images"].items()
        }

        return jsonify({
            "success": True,
            **result_data
        })

    except Exception as e:
        print("🔥 ERROR:", e)
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)