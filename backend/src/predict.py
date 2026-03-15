import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from matplotlib import colormaps

# Load model once at startup
model = load_model("models/brain_tumor_model.h5")

class_labels = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

def get_stages_images(img_path, output_dir):
    """Handles Stage 2 (ROI) and Stage 3 (Preprocessing)."""
    img = cv2.imread(img_path)
    filename = os.path.basename(img_path)

    # Stage 2: ROI Extraction (Cropping the skull)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    roi_img = img[y:y+h, x:x+w]
    roi_path = f"stage2_{filename}"
    cv2.imwrite(os.path.join(output_dir, roi_path), roi_img)

    # Stage 3: Pre-processing (Resize & Denoise)
    processed = cv2.resize(roi_img, (224, 224))
    processed = cv2.fastNlMeansDenoisingColored(processed, None, 10, 10, 7, 21)
    processed_path = f"stage3_{filename}"
    cv2.imwrite(os.path.join(output_dir, processed_path), processed)

    return roi_path, processed_path, processed

def make_gradcam_heatmap(img_array, model):
    """Handles Stage 7 & 8 (Activation Maps)."""
    layer_name = None
    for layer in reversed(model.layers):
        if len(layer.output.shape) == 4:
            layer_name = layer.name
            break

    grad_model = tf.keras.models.Model(model.inputs, [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        predictions = tf.convert_to_tensor(predictions)
        pred_index = tf.argmax(predictions[0])
        class_channel = tf.gather(predictions[0], pred_index)

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def predict_image(img_path):
    output_dir = os.path.dirname(img_path)
    filename = os.path.basename(img_path)

    # Stages 2 & 3
    roi_path, processed_path, processed_img = get_stages_images(img_path, output_dir)

    # Stage 6: Prediction
    img_array = (processed_img / 255.0).reshape(1, 224, 224, 3)
    preds = model.predict(img_array)[0]
    idx = np.argmax(preds)
    label = class_labels[idx]
    conf = float(preds[idx] * 100)

    # Stage 4/5: Masking
    heatmap = make_gradcam_heatmap(img_array, model)
    mask = (heatmap > 0.4).astype(np.uint8) * 255
    mask_path = f"stage4_{filename}"
    cv2.imwrite(os.path.join(output_dir, mask_path), cv2.resize(mask, (224, 224)))

    # Stage 8/9: Final Result
    jet = colormaps.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = cv2.resize(jet_colors[np.uint8(255 * heatmap)], (224, 224))
    overlay = cv2.addWeighted(processed_img, 0.6, (jet_heatmap * 255).astype(np.uint8), 0.4, 0)
    gradcam_path = f"stage9_{filename}"
    cv2.imwrite(os.path.join(output_dir, gradcam_path), overlay)

    return {
        "prediction": label,
        "confidence": conf,
        "probabilities": {class_labels[i]: float(preds[i] * 100) for i in range(4)},
        "stage_images": {
            "roi": roi_path,
            "processed": processed_path,
            "mask": mask_path,
            "gradcam": gradcam_path
        }
    }