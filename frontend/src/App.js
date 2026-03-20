////import React, { useState } from "react";
////import "./App.css";
////import jsPDF from "jspdf";
////
////function App() {
////  const [selectedImage, setSelectedImage] = useState(null);
////  const [preview, setPreview] = useState(null);
////  const [result, setResult] = useState(null);
////  const [loading, setLoading] = useState(false);
////
////  const handleImageChange = (event) => {
////    const file = event.target.files[0];
////    setSelectedImage(file);
////    setPreview(URL.createObjectURL(file));
////    setResult(null);
////  };
////
////  const handleSubmit = async () => {
////    if (!selectedImage) {
////      alert("Please select an MRI image first.");
////      return;
////    }
////
////    const formData = new FormData();
////    formData.append("image", selectedImage);
////
////    try {
////      setLoading(true);
////
////      const response = await fetch("http://127.0.0.1:5000/predict", {
////        method: "POST",
////        body: formData,
////      });
////
////      const data = await response.json();
////
////      if (!data.success) {
////        alert("Prediction failed: " + data.error);
////        setLoading(false);
////        return;
////      }
////
////      setResult(data);
////      setLoading(false);
////
////    } catch (error) {
////      console.error(error);
////      alert("Backend not connected");
////      setLoading(false);
////    }
////  };
////
////  const getPredictionColor = (prediction) => {
////    return prediction === "No Tumor" ? "#22c55e" : "#ef4444";
////  };
////
////  const downloadReport = () => {
////    const doc = new jsPDF();
////
////    doc.setFontSize(18);
////    doc.text("Brain Tumor Detection Report", 20, 20);
////
////    doc.setFontSize(12);
////    doc.text(`Date: ${new Date().toLocaleString()}`, 20, 35);
////
////    doc.text(
////      `Binary Detection: ${result.binary_prediction} (${result.binary_label})`,
////      20,
////      50
////    );
////
////    doc.text(`Predicted Tumor Type: ${result.prediction}`, 20, 60);
////
////    doc.text(`Confidence: ${result.confidence.toFixed(2)}%`, 20, 70);
////
////    doc.text("Class Probabilities:", 20, 85);
////
////    let y = 95;
////    Object.entries(result.probabilities).forEach(([label, value]) => {
////      doc.text(`${label}: ${value.toFixed(2)}%`, 25, y);
////      y += 10;
////    });
////
////    doc.text("Model: MobileNetV2 (Transfer Learning)", 20, y + 10);
////    doc.text("Framework: TensorFlow + Keras", 20, y + 20);
////
////    doc.save("Brain_Tumor_Report.pdf");
////  };
////
////  return (
////    <div className="container">
////      <h1>🧠 Brain Tumor Detection System</h1>
////
////      <div className="model-info">
////        <h3>Model Information</h3>
////        <p><strong>Architecture:</strong> MobileNetV2 (Transfer Learning)</p>
////        <p><strong>Input Size:</strong> 224 × 224</p>
////        <p><strong>Framework:</strong> TensorFlow + Keras</p>
////        <p><strong>Dataset:</strong> Brain MRI Dataset (Kaggle)</p>
////        <p><strong>Validation Accuracy:</strong> 94.8%</p>
////      </div>
////
////      <div className="upload-box">
////        <input type="file" accept="image/*" onChange={handleImageChange} />
////        <button onClick={handleSubmit}>Analyze MRI</button>
////      </div>
////
////      {preview && (
////        <div className="preview-section">
////          <h3>Uploaded MRI</h3>
////          <img src={preview} alt="MRI Preview" className="preview-img" />
////        </div>
////      )}
////
////      {loading && <p>Analyzing...</p>}
////
////      {result && (
////        <div className="result-section">
////
////          <h2>Prediction Result</h2>
////
////          {/* Binary Classification */}
////          <p style={{ fontSize: "18px" }}>
////            <strong>Tumor Detection (Binary):</strong>{" "}
////            <span
////              style={{
////                color: result.binary_prediction === 0 ? "#22c55e" : "#ef4444",
////                fontWeight: "bold",
////              }}
////            >
////              {result.binary_prediction} — {result.binary_label}
////            </span>
////          </p>
////
////          {/* Multi-class classification */}
////          <p style={{ fontSize: "20px" }}>
////            <strong>Predicted Tumor Type:</strong>{" "}
////            <span
////              style={{
////                color: getPredictionColor(result.prediction),
////                fontWeight: "bold",
////              }}
////            >
////              {result.prediction}
////            </span>
////          </p>
////
////          <p style={{ fontSize: "18px" }}>
////            <strong>Confidence:</strong>{" "}
////            <span style={{ color: "#38bdf8", fontWeight: "bold" }}>
////              {result.confidence.toFixed(2)}%
////            </span>
////          </p>
////
////          <h3>Class Probabilities</h3>
////
////          <div className="probability-container">
////            {Object.entries(result.probabilities).map(([label, value]) => (
////              <div key={label} className="probability-bar">
////                <span>{label}</span>
////                <div className="bar">
////                  <div
////                    className="fill"
////                    style={{ width: `${value}%` }}
////                  ></div>
////                </div>
////                <span>{value.toFixed(2)}%</span>
////              </div>
////            ))}
////          </div>
////
////          <button
////            onClick={downloadReport}
////            style={{ marginTop: "20px", backgroundColor: "#3b82f6" }}
////          >
////            Download AI Report (PDF)
////          </button>
////
////          <div className="confusion-section">
////            <h3>Model Evaluation (Confusion Matrix)</h3>
////            <img
////              src="/confusion_matrix.png"
////              alt="Confusion Matrix"
////              className="confusion-img"
////            />
////          </div>
////
////        </div>
////      )}
////    </div>
////  );
////}
////
////export default App;
//
//
//
////import React, { useState } from "react";
////import "./App.css";
////import jsPDF from "jspdf";
////
////function App() {
////  const [selectedImage, setSelectedImage] = useState(null);
////  const [preview, setPreview] = useState(null);
////  const [result, setResult] = useState(null);
////  const [loading, setLoading] = useState(false);
////
////  const handleImageChange = (event) => {
////    const file = event.target.files[0];
////    if (file) {
////      setSelectedImage(file);
////      setPreview(URL.createObjectURL(file));
////      setResult(null);
////    }
////  };
////
////  const handleSubmit = async () => {
////    if (!selectedImage) return alert("Select an MRI first.");
////    const formData = new FormData();
////    formData.append("image", selectedImage);
////
////    try {
////      setLoading(true);
////      const response = await fetch("http://127.0.0.1:5000/predict", {
////        method: "POST",
////        body: formData,
////      });
////      const data = await response.json();
////      setResult(data);
////      setLoading(false);
////    } catch (error) {
////      alert("Error contacting backend.");
////      setLoading(false);
////    }
////  };
////
////  const downloadReport = () => {
////    const doc = new jsPDF();
////    doc.text("Brain Tumor AI Analysis", 20, 20);
////    doc.text(`Diagnosis: ${result.prediction}`, 20, 40);
////    doc.text(`Confidence: ${result.confidence.toFixed(2)}%`, 20, 50);
////    doc.save("Report.pdf");
////  };
////
////  return (
////    <div className="container">
////      <h1>🧠 Brain Tumor Detection System</h1>
////
////      <div className="upload-box">
////        <input type="file" accept="image/*" onChange={handleImageChange} />
////        <button onClick={handleSubmit} disabled={loading}>
////          {loading ? "Analyzing..." : "Analyze MRI"}
////        </button>
////      </div>
////
////      {/* Stage 1: Image Acquisition */}
////      {preview && !result && (
////        <div className="stage-card" style={{ maxWidth: "300px", margin: "auto" }}>
////          <span className="stage-label">Stage 1</span>
////          <p>Input Image Acquired</p>
////          <img src={preview} alt="Input" />
////        </div>
////      )}
////
////      {result && (
////        <div className="result-section">
////          <h2>9-Stage Implementation Results</h2>
////          <div className="pipeline-grid">
////            <div className="stage-card">
////              <span className="stage-label">Stage 2</span>
////              <p>ROI Extraction</p>
////              <img src={result.stage_images.roi} alt="ROI" />
////            </div>
////            <div className="stage-card">
////              <span className="stage-label">Stage 3</span>
////              <p>Pre-Processing</p>
////              <img src={result.stage_images.processed} alt="P" />
////            </div>
////            <div className="stage-card">
////              <span className="stage-label">Stage 4/5</span>
////              <p>Segmentation Mask</p>
////              <img src={result.stage_images.mask} alt="M" />
////            </div>
////            <div className="stage-card highlight">
////              <span className="stage-label">Stage 8/9</span>
////              <p>Grad-CAM Overlay</p>
////              <img src={result.stage_images.gradcam} alt="G" />
////            </div>
////          </div>
////
////          <div className="final-prediction">
////            <h2>Result: <span className="highlight-text">{result.prediction}</span></h2>
////            <p>Confidence Score: <strong>{result.confidence.toFixed(2)}%</strong></p>
////            <button onClick={downloadReport} style={{ background: "#3b82f6" }}>Download PDF</button>
////          </div>
////        </div>
////      )}
////    </div>
////  );
////}
////
////export default App;
//
//
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
//    if (file) {
//      setSelectedImage(file);
//      setPreview(URL.createObjectURL(file));
//      setResult(null);
//    }
//  };
//
//  const handleSubmit = async () => {
//    if (!selectedImage) return alert("Select an MRI first.");
//    const formData = new FormData();
//    formData.append("image", selectedImage);
//
//    try {
//      setLoading(true);
//      const response = await fetch("http://127.0.0.1:5000/predict", {
//        method: "POST",
//        body: formData,
//      });
//      const data = await response.json();
//      setResult(data);
//      setLoading(false);
//    } catch (error) {
//      alert("Error contacting backend.");
//      setLoading(false);
//    }
//  };
//
//  const downloadReport = () => {
//    const doc = new jsPDF();
//    doc.text("Brain Tumor AI Analysis", 20, 20);
//    doc.text(`Diagnosis: ${result.prediction}`, 20, 40);
//    doc.text(`Confidence: ${result.confidence.toFixed(2)}%`, 20, 50);
//    doc.save("Report.pdf");
//  };
//
//  return (
//    <div className="container">
//      <h1>🧠 Brain Tumor Detection System</h1>
//
//      <div className="upload-box">
//        <input type="file" accept="image/*" onChange={handleImageChange} />
//        <button onClick={handleSubmit} disabled={loading}>
//          {loading ? "Analyzing..." : "Analyze MRI"}
//        </button>
//      </div>
//
//      {/* Stage 1: Image Acquisition */}
//      {preview && !result && (
//        <div className="stage-card" style={{ maxWidth: "300px", margin: "auto" }}>
//          <span className="stage-label">Input</span>
//          <p>MRI Acquired</p>
//          <img src={preview} alt="Input" />
//        </div>
//      )}
//
//      {result && (
//        <div className="result-section">
//          <h2>SAM-GNN Framework Pipeline</h2>
//
//          <div className="pipeline-grid">
//            {/* Step 1: Preprocessing */}
//            <div className="stage-card">
//              <span className="stage-label">Step 1</span>
//              <p>Pre-Processing</p>
//              <img src={result.stage_images.processed} alt="Pre-processed" />
//            </div>
//
//            {/* Step 2: SAM Mask */}
//            <div className="stage-card">
//              <span className="stage-label">Step 2</span>
//              <p>Segment Anything (SAM)</p>
//              <img src={result.stage_images.sam_mask} alt="SAM Mask" />
//            </div>
//
//            {/* Step 3: Superpixels */}
//            <div className="stage-card">
//              <span className="stage-label">Step 3</span>
//              <p>Superpixel (SLIC)</p>
//              <img src={result.stage_images.superpixel} alt="Superpixels" />
//            </div>
//
//            {/* Step 4: Graph Construction */}
//            <div className="stage-card">
//              <span className="stage-label">Step 4</span>
//              <p>Graph Construction</p>
//              <img src={result.stage_images.graph} alt="Graph Network" />
//            </div>
//
//            {/* Step 5: Refined Segmentation (Highlighted) */}
//            <div className="stage-card highlight">
//              <span className="stage-label">Step 5</span>
//              <p>Refined Segmentation</p>
//              <img src={result.stage_images.refined} alt="Refined Tumor" />
//            </div>
//
//            {/* Bonus: CNN Grad-CAM */}
//            <div className="stage-card">
//              <span className="stage-label">CNN Branch</span>
//              <p>Grad-CAM Overlay</p>
//              <img src={result.stage_images.gradcam} alt="Grad-CAM" />
//            </div>
//          </div>
//
//          <div className="final-prediction">
//            <h2>Result: <span className="highlight-text">{result.prediction}</span></h2>
//            <p>Confidence Score: <strong>{result.confidence.toFixed(2)}%</strong></p>
//            <button onClick={downloadReport} style={{ background: "#3b82f6" }}>Download PDF</button>
//          </div>
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

const STAGES = [
  { key: "processed",  step: "Step 1", label: "Pre-Processing",         sub: "Skull strip · Denoise · Resize" },
  { key: "sam_mask",   step: "Step 2", label: "Segment Anything (SAM)", sub: "Initial tumor mask" },
  { key: "superpixel", step: "Step 3", label: "Superpixel (SLIC)",      sub: "Organic region patches" },
  { key: "graph",      step: "Step 4", label: "Graph Construction",     sub: "Nodes = superpixels · Edges = neighbors" },
  { key: "refined",    step: "Step 5", label: "Refined Segmentation",   sub: "GNN + Veto fusion", highlight: true },
  { key: "gradcam",    step: "CNN",    label: "Grad-CAM Overlay",       sub: "Texture attention heatmap" },
];

const CLASS_COLORS = {
  Glioma:      "#f87171",
  Meningioma:  "#fb923c",
  Pituitary:   "#a78bfa",
  "No Tumor":  "#34d399",
};

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [preview, setPreview]             = useState(null);
  const [result, setResult]               = useState(null);
  const [loading, setLoading]             = useState(false);
  const [error, setError]                 = useState(null);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setSelectedImage(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
    setError(null);
  };

  const handleSubmit = async () => {
    if (!selectedImage) return alert("Please select an MRI image first.");
    const formData = new FormData();
    formData.append("image", selectedImage);

    try {
      setLoading(true);
      setError(null);
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (!data.success) throw new Error(data.error || "Prediction failed");
      setResult(data);
    } catch (err) {
      console.error(err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const downloadReport = () => {
    const doc = new jsPDF();
    const now = new Date().toLocaleString();

    doc.setFillColor(15, 17, 23);
    doc.rect(0, 0, 210, 297, "F");

    doc.setTextColor(255, 255, 255);
    doc.setFontSize(20);
    doc.text("Brain Tumor Detection Report", 20, 25);

    doc.setFontSize(10);
    doc.setTextColor(150, 160, 180);
    doc.text(`Generated: ${now}`, 20, 35);
    doc.text("System: Dual-Brain SAM-GNN Framework", 20, 42);

    doc.setDrawColor(50, 60, 80);
    doc.line(20, 48, 190, 48);

    doc.setFontSize(14);
    doc.setTextColor(255, 255, 255);
    doc.text("Diagnosis", 20, 60);

    const predColor = CLASS_COLORS[result.prediction] || "#ffffff";
    const hex = predColor.replace("#", "");
    const r = parseInt(hex.slice(0, 2), 16);
    const g = parseInt(hex.slice(2, 4), 16);
    const b = parseInt(hex.slice(4, 6), 16);
    doc.setTextColor(r, g, b);
    doc.setFontSize(18);
    doc.text(result.prediction, 20, 72);

    doc.setTextColor(255, 255, 255);
    doc.setFontSize(12);
    doc.text(`Confidence Score: ${result.confidence.toFixed(2)}%`, 20, 85);

    doc.line(20, 92, 190, 92);

    doc.setFontSize(13);
    doc.text("Class Probabilities", 20, 103);
    doc.setFontSize(11);
    doc.setTextColor(180, 190, 210);

    let y = 115;
    Object.entries(result.probabilities).forEach(([label, val]) => {
      doc.text(`${label}`, 20, y);
      doc.setFillColor(40, 50, 70);
      doc.roundedRect(70, y - 4, 80, 7, 2, 2, "F");
      doc.setFillColor(r, g, b);
      doc.roundedRect(70, y - 4, Math.min(80, 80 * val / 100), 7, 2, 2, "F");
      doc.setTextColor(255, 255, 255);
      doc.text(`${val.toFixed(1)}%`, 156, y);
      doc.setTextColor(180, 190, 210);
      y += 14;
    });

    doc.line(20, y + 4, 190, y + 4);
    doc.setTextColor(100, 120, 150);
    doc.setFontSize(9);
    doc.text("Architecture: MobileNetV2 (CNN) + SAM + PyTorch Geometric (GNN)", 20, y + 14);
    doc.text("Pipeline: ROI Crop → Denoise → SAM → SLIC → Graph → GNN Ensemble", 20, y + 21);
    doc.text("Note: GNN weight is muted (0%) until trained on BraTS dataset.", 20, y + 28);

    doc.save(`BrainTumor_Report_${result.prediction.replace(" ", "_")}.pdf`);
  };

  const predColor = result ? (CLASS_COLORS[result.prediction] || "#ffffff") : "#ffffff";

  return (
    <div className="app">

      {/* ── Header ── */}
      <header className="header">
        <div className="header-inner">
          <div className="logo">
            <span className="logo-icon">⬡</span>
            <span className="logo-text">Dual<span className="logo-accent">Brain</span></span>
          </div>
          <h1 className="site-title">Brain Tumor Detection System</h1>
          <div className="header-badge">SAM · GNN · CNN</div>
        </div>
      </header>

      <main className="main">

        {/* ── Upload Panel ── */}
        <section className="upload-panel">
          <div className="upload-inner">
            <label className="file-label">
              <input
                type="file"
                accept="image/*,.nii,.gz"
                onChange={handleImageChange}
                className="file-input"
              />
              <span className="file-btn">
                {selectedImage ? selectedImage.name : "Choose MRI file"}
              </span>
            </label>
            <button
              className={`analyze-btn ${loading ? "loading" : ""}`}
              onClick={handleSubmit}
              disabled={loading}
            >
              {loading ? (
                <>
                  <span className="spinner" />
                  Analyzing…
                </>
              ) : (
                "Analyze MRI"
              )}
            </button>
          </div>

          {preview && (
            <div className="preview-wrap">
              <span className="preview-label">Input MRI</span>
              <img src={preview} alt="MRI preview" className="preview-img" />
            </div>
          )}

          {error && <div className="error-banner">{error}</div>}
        </section>

        {/* ── Loading state ── */}
        {loading && (
          <div className="loading-panel">
            <div className="pulse-ring" />
            <p className="loading-text">Running dual-brain pipeline…</p>
            <div className="loading-steps">
              {["Skull stripping", "SAM segmentation", "SLIC superpixels", "Graph build", "GNN inference", "Veto fusion"].map((s, i) => (
                <span key={i} className="loading-step" style={{ animationDelay: `${i * 0.4}s` }}>{s}</span>
              ))}
            </div>
          </div>
        )}

        {/* ── Results ── */}
        {result && (
          <div className="result-section">

            {/* Pipeline title */}
            <div className="section-header">
              <h2 className="section-title">SAM-GNN Framework Pipeline</h2>
              <span className="section-sub">7-stage multi-modal analysis</span>
            </div>

            {/* Stage cards */}
            <div className="pipeline-grid">
              {STAGES.map(({ key, step, label, sub, highlight }) => (
                result.stage_images[key] && (
                  <div
                    key={key}
                    className={`stage-card ${highlight ? "stage-highlight" : ""}`}
                  >
                    <div className="stage-header">
                      <span className={`step-badge ${highlight ? "badge-highlight" : ""}`}>
                        {step}
                      </span>
                      <span className="stage-title">{label}</span>
                    </div>
                    <div className="stage-img-wrap">
                      <img
                        src={result.stage_images[key]}
                        alt={label}
                        className="stage-img"
                      />
                    </div>
                    <p className="stage-sub">{sub}</p>
                  </div>
                )
              ))}
            </div>

            {/* ── Diagnosis Panel ── */}
            <div className="diagnosis-row">

              {/* Result card */}
              <div className="diagnosis-card" style={{ "--accent": predColor }}>
                <p className="diag-eyebrow">Final Diagnosis</p>
                <h2 className="diag-prediction" style={{ color: predColor }}>
                  {result.prediction}
                </h2>
                <div className="confidence-wrap">
                  <span className="conf-label">Confidence</span>
                  <span className="conf-value">{result.confidence.toFixed(2)}%</span>
                </div>
                <div className="conf-track">
                  <div
                    className="conf-fill"
                    style={{
                      width: `${result.confidence}%`,
                      background: predColor,
                    }}
                  />
                </div>
                <button className="pdf-btn" onClick={downloadReport}>
                  Download PDF Report
                </button>
              </div>

              {/* Probability bars */}
              <div className="prob-card">
                <p className="prob-title">Class probabilities</p>
                {Object.entries(result.probabilities)
                  .sort((a, b) => b[1] - a[1])
                  .map(([label, val]) => (
                    <div key={label} className="prob-row">
                      <span className="prob-label">{label}</span>
                      <div className="prob-track">
                        <div
                          className="prob-fill"
                          style={{
                            width: `${val}%`,
                            background: CLASS_COLORS[label] || "#6b7280",
                          }}
                        />
                      </div>
                      <span className="prob-val">{val.toFixed(1)}%</span>
                    </div>
                  ))}
              </div>

              {/* System info card */}
              <div className="info-card">
                <p className="info-title">Pipeline info</p>
                <div className="info-row">
                  <span className="info-key">CNN branch</span>
                  <span className="info-val">MobileNetV2</span>
                </div>
                <div className="info-row">
                  <span className="info-key">Segmentation</span>
                  <span className="info-val">SAM ViT-H</span>
                </div>
                <div className="info-row">
                  <span className="info-key">Graph model</span>
                  <span className="info-val">PyG GCNConv</span>
                </div>
                <div className="info-row">
                  <span className="info-key">Superpixels</span>
                  <span className="info-val">SLIC (n=100)</span>
                </div>
                <div className="info-row">
                  <span className="info-key">Node features</span>
                  <span className="info-val">16 regionprops</span>
                </div>
                <div className="info-row">
                  <span className="info-key">GNN weight</span>
                  <span className="info-val info-ok">30% (active)</span>
                </div>
                <div className="info-row">
                  <span className="info-key">Input size</span>
                  <span className="info-val">224 × 224</span>
                </div>
                <div className="info-row">
                  <span className="info-key">Veto system</span>
                  <span className="info-val info-ok">Active</span>
                </div>
              </div>

            </div>
          </div>
        )}
      </main>

      <footer className="footer">
        <span>Dual-Brain SAM-GNN · MobileNetV2 + Meta SAM + PyTorch Geometric</span>
      </footer>
    </div>
  );
}

export default App;