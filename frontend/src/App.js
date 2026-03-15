//import React, { useState } from "react";
//import "./App.css";
//import jsPDF from "jspdf";
//
//function App() {
//  const [selectedImage, setSelectedImage] = useState(null);
//  const [preview, setPreview] = useState(null);
//  const [result, setResult] = useState(null);
//  const [loading, setLoading] = useState(false);
//
//  const handleImageChange = (event) => {
//    const file = event.target.files[0];
//    setSelectedImage(file);
//    setPreview(URL.createObjectURL(file));
//    setResult(null);
//  };
//
//  const handleSubmit = async () => {
//    if (!selectedImage) {
//      alert("Please select an MRI image first.");
//      return;
//    }
//
//    const formData = new FormData();
//    formData.append("image", selectedImage);
//
//    try {
//      setLoading(true);
//
//      const response = await fetch("http://127.0.0.1:5000/predict", {
//        method: "POST",
//        body: formData,
//      });
//
//      const data = await response.json();
//
//      if (!data.success) {
//        alert("Prediction failed: " + data.error);
//        setLoading(false);
//        return;
//      }
//
//      setResult(data);
//      setLoading(false);
//
//    } catch (error) {
//      console.error(error);
//      alert("Backend not connected");
//      setLoading(false);
//    }
//  };
//
//  const getPredictionColor = (prediction) => {
//    return prediction === "No Tumor" ? "#22c55e" : "#ef4444";
//  };
//
//  const downloadReport = () => {
//    const doc = new jsPDF();
//
//    doc.setFontSize(18);
//    doc.text("Brain Tumor Detection Report", 20, 20);
//
//    doc.setFontSize(12);
//    doc.text(`Date: ${new Date().toLocaleString()}`, 20, 35);
//
//    doc.text(
//      `Binary Detection: ${result.binary_prediction} (${result.binary_label})`,
//      20,
//      50
//    );
//
//    doc.text(`Predicted Tumor Type: ${result.prediction}`, 20, 60);
//
//    doc.text(`Confidence: ${result.confidence.toFixed(2)}%`, 20, 70);
//
//    doc.text("Class Probabilities:", 20, 85);
//
//    let y = 95;
//    Object.entries(result.probabilities).forEach(([label, value]) => {
//      doc.text(`${label}: ${value.toFixed(2)}%`, 25, y);
//      y += 10;
//    });
//
//    doc.text("Model: MobileNetV2 (Transfer Learning)", 20, y + 10);
//    doc.text("Framework: TensorFlow + Keras", 20, y + 20);
//
//    doc.save("Brain_Tumor_Report.pdf");
//  };
//
//  return (
//    <div className="container">
//      <h1>🧠 Brain Tumor Detection System</h1>
//
//      <div className="model-info">
//        <h3>Model Information</h3>
//        <p><strong>Architecture:</strong> MobileNetV2 (Transfer Learning)</p>
//        <p><strong>Input Size:</strong> 224 × 224</p>
//        <p><strong>Framework:</strong> TensorFlow + Keras</p>
//        <p><strong>Dataset:</strong> Brain MRI Dataset (Kaggle)</p>
//        <p><strong>Validation Accuracy:</strong> 94.8%</p>
//      </div>
//
//      <div className="upload-box">
//        <input type="file" accept="image/*" onChange={handleImageChange} />
//        <button onClick={handleSubmit}>Analyze MRI</button>
//      </div>
//
//      {preview && (
//        <div className="preview-section">
//          <h3>Uploaded MRI</h3>
//          <img src={preview} alt="MRI Preview" className="preview-img" />
//        </div>
//      )}
//
//      {loading && <p>Analyzing...</p>}
//
//      {result && (
//        <div className="result-section">
//
//          <h2>Prediction Result</h2>
//
//          {/* Binary Classification */}
//          <p style={{ fontSize: "18px" }}>
//            <strong>Tumor Detection (Binary):</strong>{" "}
//            <span
//              style={{
//                color: result.binary_prediction === 0 ? "#22c55e" : "#ef4444",
//                fontWeight: "bold",
//              }}
//            >
//              {result.binary_prediction} — {result.binary_label}
//            </span>
//          </p>
//
//          {/* Multi-class classification */}
//          <p style={{ fontSize: "20px" }}>
//            <strong>Predicted Tumor Type:</strong>{" "}
//            <span
//              style={{
//                color: getPredictionColor(result.prediction),
//                fontWeight: "bold",
//              }}
//            >
//              {result.prediction}
//            </span>
//          </p>
//
//          <p style={{ fontSize: "18px" }}>
//            <strong>Confidence:</strong>{" "}
//            <span style={{ color: "#38bdf8", fontWeight: "bold" }}>
//              {result.confidence.toFixed(2)}%
//            </span>
//          </p>
//
//          <h3>Class Probabilities</h3>
//
//          <div className="probability-container">
//            {Object.entries(result.probabilities).map(([label, value]) => (
//              <div key={label} className="probability-bar">
//                <span>{label}</span>
//                <div className="bar">
//                  <div
//                    className="fill"
//                    style={{ width: `${value}%` }}
//                  ></div>
//                </div>
//                <span>{value.toFixed(2)}%</span>
//              </div>
//            ))}
//          </div>
//
//          <button
//            onClick={downloadReport}
//            style={{ marginTop: "20px", backgroundColor: "#3b82f6" }}
//          >
//            Download AI Report (PDF)
//          </button>
//
//          <div className="confusion-section">
//            <h3>Model Evaluation (Confusion Matrix)</h3>
//            <img
//              src="/confusion_matrix.png"
//              alt="Confusion Matrix"
//              className="confusion-img"
//            />
//          </div>
//
//        </div>
//      )}
//    </div>
//  );
//}
//
//export default App;



import React, { useState } from "react";
import "./App.css";
import jsPDF from "jspdf";

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleImageChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedImage(file);
      setPreview(URL.createObjectURL(file));
      setResult(null);
    }
  };

  const handleSubmit = async () => {
    if (!selectedImage) return alert("Select an MRI first.");
    const formData = new FormData();
    formData.append("image", selectedImage);

    try {
      setLoading(true);
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      setResult(data);
      setLoading(false);
    } catch (error) {
      alert("Error contacting backend.");
      setLoading(false);
    }
  };

  const downloadReport = () => {
    const doc = new jsPDF();
    doc.text("Brain Tumor AI Analysis", 20, 20);
    doc.text(`Diagnosis: ${result.prediction}`, 20, 40);
    doc.text(`Confidence: ${result.confidence.toFixed(2)}%`, 20, 50);
    doc.save("Report.pdf");
  };

  return (
    <div className="container">
      <h1>🧠 Brain Tumor Detection System</h1>

      <div className="upload-box">
        <input type="file" accept="image/*" onChange={handleImageChange} />
        <button onClick={handleSubmit} disabled={loading}>
          {loading ? "Analyzing..." : "Analyze MRI"}
        </button>
      </div>

      {/* Stage 1: Image Acquisition */}
      {preview && !result && (
        <div className="stage-card" style={{ maxWidth: "300px", margin: "auto" }}>
          <span className="stage-label">Stage 1</span>
          <p>Input Image Acquired</p>
          <img src={preview} alt="Input" />
        </div>
      )}

      {result && (
        <div className="result-section">
          <h2>9-Stage Implementation Results</h2>
          <div className="pipeline-grid">
            <div className="stage-card">
              <span className="stage-label">Stage 2</span>
              <p>ROI Extraction</p>
              <img src={result.stage_images.roi} alt="ROI" />
            </div>
            <div className="stage-card">
              <span className="stage-label">Stage 3</span>
              <p>Pre-Processing</p>
              <img src={result.stage_images.processed} alt="P" />
            </div>
            <div className="stage-card">
              <span className="stage-label">Stage 4/5</span>
              <p>Segmentation Mask</p>
              <img src={result.stage_images.mask} alt="M" />
            </div>
            <div className="stage-card highlight">
              <span className="stage-label">Stage 8/9</span>
              <p>Grad-CAM Overlay</p>
              <img src={result.stage_images.gradcam} alt="G" />
            </div>
          </div>

          <div className="final-prediction">
            <h2>Result: <span className="highlight-text">{result.prediction}</span></h2>
            <p>Confidence Score: <strong>{result.confidence.toFixed(2)}%</strong></p>
            <button onClick={downloadReport} style={{ background: "#3b82f6" }}>Download PDF</button>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;