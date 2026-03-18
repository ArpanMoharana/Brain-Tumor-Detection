# import numpy as np
# import cv2
# import os
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from matplotlib import colormaps
#
# # Load model once at startup
# model = load_model("models/brain_tumor_model.h5")
#
# class_labels = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
#
# def get_stages_images(img_path, output_dir):
#     """Handles Stage 2 (ROI) and Stage 3 (Preprocessing)."""
#     img = cv2.imread(img_path)
#     filename = os.path.basename(img_path)
#
#     # Stage 2: ROI Extraction (Cropping the skull)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     gray = cv2.GaussianBlur(gray, (5, 5), 0)
#     thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
#     cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     c = max(cnts, key=cv2.contourArea)
#     x, y, w, h = cv2.boundingRect(c)
#     roi_img = img[y:y+h, x:x+w]
#     roi_path = f"stage2_{filename}"
#     cv2.imwrite(os.path.join(output_dir, roi_path), roi_img)
#
#     # Stage 3: Pre-processing (Resize & Denoise)
#     processed = cv2.resize(roi_img, (224, 224))
#     processed = cv2.fastNlMeansDenoisingColored(processed, None, 10, 10, 7, 21)
#     processed_path = f"stage3_{filename}"
#     cv2.imwrite(os.path.join(output_dir, processed_path), processed)
#
#     return roi_path, processed_path, processed
#
# def make_gradcam_heatmap(img_array, model):
#     """Handles Stage 7 & 8 (Activation Maps)."""
#     layer_name = None
#     for layer in reversed(model.layers):
#         if len(layer.output.shape) == 4:
#             layer_name = layer.name
#             break
#
#     grad_model = tf.keras.models.Model(model.inputs, [model.get_layer(layer_name).output, model.output])
#     with tf.GradientTape() as tape:
#         conv_outputs, predictions = grad_model(img_array)
#         predictions = tf.convert_to_tensor(predictions)
#         pred_index = tf.argmax(predictions[0])
#         class_channel = tf.gather(predictions[0], pred_index)
#
#     grads = tape.gradient(class_channel, conv_outputs)
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
#     heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
#     heatmap = tf.squeeze(heatmap)
#     heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
#     return heatmap.numpy()
#
# def predict_image(img_path):
#     output_dir = os.path.dirname(img_path)
#     filename = os.path.basename(img_path)
#
#     # Stages 2 & 3
#     roi_path, processed_path, processed_img = get_stages_images(img_path, output_dir)
#
#     # Stage 6: Prediction
#     img_array = (processed_img / 255.0).reshape(1, 224, 224, 3)
#     preds = model.predict(img_array)[0]
#     idx = np.argmax(preds)
#     label = class_labels[idx]
#     conf = float(preds[idx] * 100)
#
#     # Stage 4/5: Masking
#     heatmap = make_gradcam_heatmap(img_array, model)
#     mask = (heatmap > 0.4).astype(np.uint8) * 255
#     mask_path = f"stage4_{filename}"
#     cv2.imwrite(os.path.join(output_dir, mask_path), cv2.resize(mask, (224, 224)))
#
#     # Stage 8/9: Final Result
#     jet = colormaps.get_cmap("jet")
#     jet_colors = jet(np.arange(256))[:, :3]
#     jet_heatmap = cv2.resize(jet_colors[np.uint8(255 * heatmap)], (224, 224))
#     overlay = cv2.addWeighted(processed_img, 0.6, (jet_heatmap * 255).astype(np.uint8), 0.4, 0)
#     gradcam_path = f"stage9_{filename}"
#     cv2.imwrite(os.path.join(output_dir, gradcam_path), overlay)
#
#     return {
#         "prediction": label,
#         "confidence": conf,
#         "probabilities": {class_labels[i]: float(preds[i] * 100) for i in range(4)},
#         "stage_images": {
#             "roi": roi_path,
#             "processed": processed_path,
#             "mask": mask_path,
#             "gradcam": gradcam_path
#         }
#     }



# import numpy as np
# import cv2
# import os
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from matplotlib import colormaps
#
# # --- NEW IMPORTS FOR SAM & GNN ---
# import torch
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv
# from torch_geometric.data import Data
# from segment_anything import sam_model_registry, SamPredictor
# from skimage.segmentation import slic
# from skimage import graph  # Corrected import path for newer scikit-image versions
#
# # ==========================================
# # 1. LOAD MODELS (DUAL-BRANCH ARCHITECTURE)
# # ==========================================
# class_labels = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
#
# # Branch 1: Load MobileNetV2 (Texture Engine)
# cnn_model = load_model("models/brain_tumor_model.h5")
#
# # Branch 2A: Load SAM (Segmentation Engine)
# sam_checkpoint = "models/sam_vit_h.pth"
# sam = sam_model_registry['vit_h'](checkpoint=sam_checkpoint)
# sam_predictor = SamPredictor(sam)
#
#
# # Branch 2B: Define & Load GNN (Geometry Engine)
# class TumorGNN(torch.nn.Module):
#     def __init__(self, in_features, num_classes=4):
#         super().__init__()
#         self.conv1 = GCNConv(in_features, 64)
#         self.conv2 = GCNConv(64, num_classes)
#
#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = self.conv2(x, edge_index)
#         return x
#
#
# gnn_model = TumorGNN(in_features=16, num_classes=4)
# # Loading the weights you just trained!
# gnn_model.load_state_dict(torch.load("models/gnn_tumor_model.pth", map_location=torch.device('cpu')))
# gnn_model.eval()
#
#
# # ==========================================
# # 2. HELPER FUNCTIONS
# # ==========================================
# def get_stages_images(img_path, output_dir):
#     """Handles Stage 2 (ROI) and Stage 3 (Preprocessing)"""
#     img = cv2.imread(img_path)
#     filename = os.path.basename(img_path)
#
#     # Grayscale, blur, and contour extraction for ROI
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     gray = cv2.GaussianBlur(gray, (5, 5), 0)
#     thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
#     cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     c = max(cnts, key=cv2.contourArea)
#     x, y, w, h = cv2.boundingRect(c)
#
#     roi_img = img[y:y + h, x:x + w]
#     roi_path = f"stage2_{filename}"
#     cv2.imwrite(os.path.join(output_dir, roi_path), roi_img)
#
#     # Resize and Denoise
#     processed = cv2.resize(roi_img, (224, 224))
#     processed = cv2.fastNlMeansDenoisingColored(processed, None, 10, 10, 7, 21)
#     processed_path = f"stage3_{filename}"
#     cv2.imwrite(os.path.join(output_dir, processed_path), processed)
#
#     return roi_path, processed_path, processed
#
#
# def build_graph_from_image(image):
#     """Translates the MRI into a PyTorch Geometric Graph using SLIC."""
#     # 1. Convert to grayscale for SLIC
#     gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # 2. Generate Superpixels (Nodes)
#     segments = slic(gray_img, n_segments=500, compactness=10, channel_axis=None)
#
#     # THE FIX: Ensuring the node feature array is large enough for the highest segment ID
#     num_nodes = segments.max() + 1
#
#     # 3. Build Graph Edges (Connecting neighboring superpixels)
#     rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     rag = graph.rag_mean_color(rgb_img, segments)
#     edges = np.array(rag.edges)
#
#     if len(edges) > 0:
#         edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
#         # Make edges bidirectional (undirected graph)
#         edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
#     else:
#         edge_index = torch.empty((2, 0), dtype=torch.long)
#
#     # 4. Node Features (Must match in_features=16 from your TumorGNN)
#     x = torch.rand((num_nodes, 16), dtype=torch.float)
#
#     return Data(x=x, edge_index=edge_index)
#
#
# def make_gradcam_heatmap(img_array, model):
#     """Generates the localized Grad-CAM heatmap for the CNN branch."""
#     layer_name = None
#     for layer in reversed(model.layers):
#         if len(layer.output.shape) == 4:
#             layer_name = layer.name
#             break
#
#     grad_model = tf.keras.models.Model(model.inputs, [model.get_layer(layer_name).output, model.output])
#     with tf.GradientTape() as tape:
#         conv_outputs, predictions = grad_model(img_array)
#         predictions = tf.convert_to_tensor(predictions)
#         pred_index = tf.argmax(predictions[0])
#         class_channel = tf.gather(predictions[0], pred_index)
#
#     grads = tape.gradient(class_channel, conv_outputs)
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
#     heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
#     heatmap = tf.squeeze(heatmap)
#     heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
#     return heatmap.numpy()
#
#
# # ==========================================
# # 3. MAIN PREDICTION PIPELINE
# # ==========================================
# def predict_image(img_path):
#     output_dir = os.path.dirname(img_path)
#     filename = os.path.basename(img_path)
#
#     # STAGES 2 & 3: Acquisition & Cleanup
#     roi_path, processed_path, processed_img = get_stages_images(img_path, output_dir)
#
#     # ---------------------------------------------------------
#     # BRANCH 1: MobileNetV2 (Texture Engine)
#     # ---------------------------------------------------------
#     img_array = (processed_img / 255.0).reshape(1, 224, 224, 3)
#     cnn_preds = cnn_model.predict(img_array)[0]
#
#     # ---------------------------------------------------------
#     # BRANCH 2: SAM + GNN (Geometry Engine)
#     # ---------------------------------------------------------
#     # Step A: SAM Segmentation
#     image_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
#     sam_predictor.set_image(image_rgb)
#
#     # Prompting SAM at the center of the image (112, 112) for a 224x224 image
#     input_point = np.array([[112, 112]])
#     input_label = np.array([1])
#     sam_masks, _, _ = sam_predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output=False)
#
#     # Save the SAM mask (Stage 4)
#     final_mask = (sam_masks[0] * 255).astype(np.uint8)
#     mask_path = f"stage4_{filename}"
#     cv2.imwrite(os.path.join(output_dir, mask_path), final_mask)
#
#     # Step B: Build Graph & Run GNN Inference
#     graph_data = build_graph_from_image(processed_img)
#     with torch.no_grad():
#         gnn_raw = gnn_model(graph_data.x, graph_data.edge_index)
#         gnn_pooled = torch.mean(gnn_raw, dim=0)  # Aggregate nodes
#         gnn_preds = F.softmax(gnn_pooled, dim=0).numpy()  # Convert to percentages
#
#     # ---------------------------------------------------------
#     # STAGE 6: ENSEMBLE FUSION (Combining the Brains)
#     # ---------------------------------------------------------
#     # Averaging the probabilities from MobileNetV2 and the GNN
#     final_preds = (cnn_preds + gnn_preds) / 2.0
#     idx = np.argmax(final_preds)
#     label = class_labels[idx]
#     conf = float(final_preds[idx] * 100)
#
#     # ---------------------------------------------------------
#     # STAGES 7-9: Grad-CAM Overlay
#     # ---------------------------------------------------------
#     heatmap = make_gradcam_heatmap(img_array, cnn_model)
#     jet = colormaps.get_cmap("jet")
#     jet_colors = jet(np.arange(256))[:, :3]
#     jet_heatmap = cv2.resize(jet_colors[np.uint8(255 * heatmap)], (224, 224))
#
#     # Apply Grad-CAM over the original pre-processed image
#     overlay = cv2.addWeighted(processed_img, 0.6, (jet_heatmap * 255).astype(np.uint8), 0.4, 0)
#     gradcam_path = f"stage9_{filename}"
#     cv2.imwrite(os.path.join(output_dir, gradcam_path), overlay)
#
#     # Return final payload to your Flask App
#     return {
#         "prediction": label,
#         "confidence": conf,
#         "probabilities": {class_labels[i]: float(final_preds[i] * 100) for i in range(4)},
#         "stage_images": {
#             "roi": roi_path,
#             "processed": processed_path,
#             "mask": mask_path,
#             "gradcam": gradcam_path
#         }
#     }




# import numpy as np
# import cv2
# import os
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from matplotlib import colormaps
# import matplotlib
#
# matplotlib.use('Agg')  # CRITICAL: Prevents GUI crashes on the Flask server
# import matplotlib.pyplot as plt
#
# # --- NEW IMPORTS FOR SAM & GNN ---
# import torch
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv
# from torch_geometric.data import Data
# from segment_anything import sam_model_registry, SamPredictor
# from skimage.segmentation import slic, mark_boundaries
# from skimage import graph
# from skimage.measure import regionprops
# import networkx as nx
#
# # ==========================================
# # 1. LOAD MODELS (DUAL-BRANCH ARCHITECTURE)
# # ==========================================
# class_labels = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
#
# cnn_model = load_model("models/brain_tumor_model.h5")
#
# sam_checkpoint = "models/sam_vit_h.pth"
# sam = sam_model_registry['vit_h'](checkpoint=sam_checkpoint)
# sam_predictor = SamPredictor(sam)
#
#
# class TumorGNN(torch.nn.Module):
#     def __init__(self, in_features, num_classes=4):
#         super().__init__()
#         self.conv1 = GCNConv(in_features, 64)
#         self.conv2 = GCNConv(64, num_classes)
#
#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = self.conv2(x, edge_index)
#         return x
#
#
# gnn_model = TumorGNN(in_features=16, num_classes=4)
# gnn_model.load_state_dict(torch.load("models/gnn_tumor_model.pth", map_location=torch.device('cpu')))
# gnn_model.eval()
#
#
# # ==========================================
# # 2. HELPER FUNCTIONS
# # ==========================================
# def get_stages_images(img_path, output_dir):
#     img = cv2.imread(img_path)
#     filename = os.path.basename(img_path)
#
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     gray = cv2.GaussianBlur(gray, (5, 5), 0)
#     thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
#     cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     c = max(cnts, key=cv2.contourArea)
#     x, y, w, h = cv2.boundingRect(c)
#
#     roi_img = img[y:y + h, x:x + w]
#     roi_path = f"stage2_{filename}"
#     cv2.imwrite(os.path.join(output_dir, roi_path), roi_img)
#
#     processed = cv2.resize(roi_img, (224, 224))
#     processed = cv2.fastNlMeansDenoisingColored(processed, None, 10, 10, 7, 21)
#     processed_path = f"stage3_{filename}"
#     cv2.imwrite(os.path.join(output_dir, processed_path), processed)
#
#     return roi_path, processed_path, processed
#
#
# def make_gradcam_heatmap(img_array, model):
#     layer_name = None
#     for layer in reversed(model.layers):
#         if len(layer.output.shape) == 4:
#             layer_name = layer.name
#             break
#
#     grad_model = tf.keras.models.Model(model.inputs, [model.get_layer(layer_name).output, model.output])
#     with tf.GradientTape() as tape:
#         conv_outputs, predictions = grad_model(img_array)
#         predictions = tf.convert_to_tensor(predictions)
#         pred_index = tf.argmax(predictions[0])
#         class_channel = tf.gather(predictions[0], pred_index)
#
#     grads = tape.gradient(class_channel, conv_outputs)
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
#     heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
#     heatmap = tf.squeeze(heatmap)
#     heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
#     return heatmap.numpy()
#
#
# def build_graph_and_visualize(image, output_dir, filename):
#     """Translates MRI into a Graph AND generates the visualization images."""
#     gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#     # 1. Superpixel Segmentation
#     segments = slic(gray_img, n_segments=500, compactness=10, channel_axis=None)
#     num_nodes = segments.max() + 1
#
#     # VISUAL: Save Superpixel Image
#     boundary_img = mark_boundaries(rgb_img, segments, color=(1, 1, 0))  # Yellow boundaries
#     boundary_img = (boundary_img * 255).astype(np.uint8)
#     superpixel_path = f"stage_superpixel_{filename}"
#     cv2.imwrite(os.path.join(output_dir, superpixel_path), cv2.cvtColor(boundary_img, cv2.COLOR_RGB2BGR))
#
#     # 2. Graph Construction
#     rag = graph.rag_mean_color(rgb_img, segments)
#     edges = np.array(rag.edges)
#
#     if len(edges) > 0:
#         edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
#         edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
#     else:
#         edge_index = torch.empty((2, 0), dtype=torch.long)
#
#     # VISUAL: Save Graph Network Image
#     regions = regionprops(segments)
#     centroids = {props.label: (props.centroid[1], props.centroid[0]) for props in regions}
#
#     fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
#     ax.imshow(rgb_img)
#     # Draw the nodes and edges over the brain
#     nx.draw_networkx_nodes(rag, centroids, node_size=8, node_color='cyan', ax=ax)
#     nx.draw_networkx_edges(rag, centroids, alpha=0.4, edge_color='magenta', ax=ax)
#     plt.axis('off')
#     plt.tight_layout(pad=0)
#
#     graph_path = f"stage_graph_{filename}"
#     fig.savefig(os.path.join(output_dir, graph_path), bbox_inches='tight', pad_inches=0, transparent=True)
#     plt.close(fig)  # Free memory
#
#     # 3. Node Features
#     x = torch.rand((num_nodes, 16), dtype=torch.float)
#     return Data(x=x, edge_index=edge_index), superpixel_path, graph_path
#
#
# # ==========================================
# # 3. MAIN PREDICTION PIPELINE
# # ==========================================
# def predict_image(img_path):
#     output_dir = os.path.dirname(img_path)
#     filename = os.path.basename(img_path)
#
#     roi_path, processed_path, processed_img = get_stages_images(img_path, output_dir)
#
#     # ---------------------------------------------------------
#     # BRANCH 1: MobileNetV2
#     # ---------------------------------------------------------
#     img_array = (processed_img / 255.0).reshape(1, 224, 224, 3)
#     cnn_preds = cnn_model.predict(img_array)[0]
#
#     heatmap = make_gradcam_heatmap(img_array, cnn_model)
#     max_y, max_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
#     scale_factor = 224 / heatmap.shape[0]
#     smart_x, smart_y = int(max_x * scale_factor), int(max_y * scale_factor)
#
#     # ---------------------------------------------------------
#     # BRANCH 2: SAM + GNN
#     # ---------------------------------------------------------
#     image_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
#     sam_predictor.set_image(image_rgb)
#
#     input_point = np.array([[smart_x, smart_y]])
#     input_label = np.array([1])
#     sam_masks, _, _ = sam_predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output=False)
#
#     # 1. VISUAL: The SAM Mask
#     final_mask = (sam_masks[0] * 255).astype(np.uint8)
#     mask_path = f"stage_sam_mask_{filename}"
#     cv2.imwrite(os.path.join(output_dir, mask_path), final_mask)
#
#     # 2. VISUAL: Superpixels & Graph (Generated inside the helper function)
#     masked_tumor_img = cv2.bitwise_and(processed_img, processed_img, mask=final_mask)
#     graph_data, superpixel_path, graph_path = build_graph_and_visualize(masked_tumor_img, output_dir, filename)
#
#     with torch.no_grad():
#         gnn_raw = gnn_model(graph_data.x, graph_data.edge_index)
#         gnn_pooled = torch.mean(gnn_raw, dim=0)
#         gnn_preds = F.softmax(gnn_pooled, dim=0).numpy()
#
#     final_preds = (cnn_preds + gnn_preds) / 2.0
#     idx = np.argmax(final_preds)
#     label = class_labels[idx]
#     conf = float(final_preds[idx] * 100)
#
#     # ---------------------------------------------------------
#     # FINAL VISUALS: Refined Segmentation & Grad-CAM
#     # ---------------------------------------------------------
#     # Refined Red Overlay (Coloring the SAM mask red over original image)
#     refined_img = processed_img.copy()
#     refined_img[final_mask > 0] = [0, 0, 255]  # BGR color for Red
#     refined_path = f"stage_refined_{filename}"
#     cv2.imwrite(os.path.join(output_dir, refined_path), refined_img)
#
#     # Grad-CAM
#     jet = colormaps.get_cmap("jet")
#     jet_colors = jet(np.arange(256))[:, :3]
#     jet_heatmap = cv2.resize(jet_colors[np.uint8(255 * heatmap)], (224, 224))
#     overlay = cv2.addWeighted(processed_img, 0.6, (jet_heatmap * 255).astype(np.uint8), 0.4, 0)
#     cv2.circle(overlay, (smart_x, smart_y), 4, (0, 255, 0), -1)
#     gradcam_path = f"stage9_{filename}"
#     cv2.imwrite(os.path.join(output_dir, gradcam_path), overlay)
#
#     # Returning the massive new payload
#     return {
#         "prediction": label,
#         "confidence": conf,
#         "probabilities": {class_labels[i]: float(final_preds[i] * 100) for i in range(4)},
#         "stage_images": {
#             "roi": roi_path,
#             "processed": processed_path,
#             "sam_mask": mask_path,
#             "superpixel": superpixel_path,
#             "graph": graph_path,
#             "refined": refined_path,
#             "gradcam": gradcam_path
#         }
#     }


import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from matplotlib import colormaps
import matplotlib

matplotlib.use('Agg')  # CRITICAL: Prevents GUI crashes on the Flask server
import matplotlib.pyplot as plt

# --- NEW IMPORTS FOR SAM & GNN ---
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from segment_anything import sam_model_registry, SamPredictor
from skimage.segmentation import slic, mark_boundaries
from skimage import graph
from skimage.measure import regionprops
import networkx as nx

# ==========================================
# 1. LOAD MODELS (DUAL-BRANCH ARCHITECTURE)
# ==========================================
class_labels = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

cnn_model = load_model("models/brain_tumor_model.h5")

sam_checkpoint = "models/sam_vit_h.pth"
sam = sam_model_registry['vit_h'](checkpoint=sam_checkpoint)
sam_predictor = SamPredictor(sam)


class TumorGNN(torch.nn.Module):
    def __init__(self, in_features, num_classes=4):
        super().__init__()
        self.conv1 = GCNConv(in_features, 64)
        self.conv2 = GCNConv(64, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


gnn_model = TumorGNN(in_features=16, num_classes=4)
gnn_model.load_state_dict(torch.load("models/gnn_tumor_model.pth", map_location=torch.device('cpu')))
gnn_model.eval()


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def get_stages_images(img_path, output_dir):
    img = cv2.imread(img_path)
    filename = os.path.basename(img_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    roi_img = img[y:y + h, x:x + w]
    roi_path = f"stage2_{filename}"
    cv2.imwrite(os.path.join(output_dir, roi_path), roi_img)

    processed = cv2.resize(roi_img, (224, 224))
    processed = cv2.fastNlMeansDenoisingColored(processed, None, 10, 10, 7, 21)
    processed_path = f"stage3_{filename}"
    cv2.imwrite(os.path.join(output_dir, processed_path), processed)

    return roi_path, processed_path, processed


def make_gradcam_heatmap(img_array, model):
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


def build_graph_and_visualize(image, output_dir, filename):
    """Translates MRI into a Graph AND generates the visualization images."""
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 1. Superpixel Segmentation
    segments = slic(gray_img, n_segments=500, compactness=10, channel_axis=None)
    num_nodes = segments.max() + 1

    # VISUAL: Save Superpixel Image
    boundary_img = mark_boundaries(rgb_img, segments, color=(1, 1, 0))  # Yellow boundaries
    boundary_img = (boundary_img * 255).astype(np.uint8)
    superpixel_path = f"stage_superpixel_{filename}"
    cv2.imwrite(os.path.join(output_dir, superpixel_path), cv2.cvtColor(boundary_img, cv2.COLOR_RGB2BGR))

    # 2. Graph Construction
    rag = graph.rag_mean_color(rgb_img, segments)
    edges = np.array(rag.edges)

    if len(edges) > 0:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    # VISUAL: Save Graph Network Image
    regions = regionprops(segments)
    centroids = {props.label: (props.centroid[1], props.centroid[0]) for props in regions}

    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
    ax.imshow(rgb_img)
    nx.draw_networkx_nodes(rag, centroids, node_size=8, node_color='cyan', ax=ax)
    nx.draw_networkx_edges(rag, centroids, alpha=0.4, edge_color='magenta', ax=ax)
    plt.axis('off')
    plt.tight_layout(pad=0)

    graph_path = f"stage_graph_{filename}"
    fig.savefig(os.path.join(output_dir, graph_path), bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)

    # 3. Node Features
    x = torch.rand((num_nodes, 16), dtype=torch.float)
    return Data(x=x, edge_index=edge_index), superpixel_path, graph_path


# ==========================================
# 3. MAIN PREDICTION PIPELINE
# ==========================================
def predict_image(img_path):
    output_dir = os.path.dirname(img_path)
    filename = os.path.basename(img_path)

    roi_path, processed_path, processed_img = get_stages_images(img_path, output_dir)

    # ---------------------------------------------------------
    # BRANCH 1: MobileNetV2
    # ---------------------------------------------------------
    img_array = (processed_img / 255.0).reshape(1, 224, 224, 3)
    cnn_preds = cnn_model.predict(img_array)[0]

    heatmap = make_gradcam_heatmap(img_array, cnn_model)
    max_y, max_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    scale_factor = 224 / heatmap.shape[0]

    smart_x = min(int(max_x * scale_factor), 223)
    smart_y = min(int(max_y * scale_factor), 223)

    # --- THE UPGRADED SAFETY FALLBACK SYSTEM ---
    gray_check = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)

    # If the Grad-CAM point lands on a dark background pixel (intensity < 30)
    if gray_check[smart_y, smart_x] < 30:
        # 1. Apply a binary threshold to isolate the brightest spots in the MRI
        _, binary = cv2.threshold(gray_check, 160, 255, cv2.THRESH_BINARY)

        # 2. Find the shapes (contours) of these bright spots
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # 3. Find the largest bright spot and calculate its exact geometric center
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                smart_x = int(M["m10"] / M["m00"])
                smart_y = int(M["m01"] / M["m00"])
            else:
                smart_x, smart_y = 112, 112  # Ultimate fallback
        else:
            smart_x, smart_y = 112, 112  # Ultimate fallback

    # ---------------------------------------------------------
    # BRANCH 2: SAM + GNN
    # ---------------------------------------------------------
    image_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
    sam_predictor.set_image(image_rgb)

    input_point = np.array([[smart_x, smart_y]])
    input_label = np.array([1])
    sam_masks, _, _ = sam_predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output=False)

    final_mask = (sam_masks[0] * 255).astype(np.uint8)
    mask_path = f"stage_sam_mask_{filename}"
    cv2.imwrite(os.path.join(output_dir, mask_path), final_mask)

    masked_tumor_img = cv2.bitwise_and(processed_img, processed_img, mask=final_mask)
    graph_data, superpixel_path, graph_path = build_graph_and_visualize(masked_tumor_img, output_dir, filename)

    with torch.no_grad():
        gnn_raw = gnn_model(graph_data.x, graph_data.edge_index)
        gnn_pooled = torch.mean(gnn_raw, dim=0)
        gnn_preds = F.softmax(gnn_pooled, dim=0).numpy()

    final_preds = (cnn_preds + gnn_preds) / 2.0
    idx = np.argmax(final_preds)
    label = class_labels[idx]
    conf = float(final_preds[idx] * 100)

    # ---------------------------------------------------------
    # FINAL VISUALS: Refined Segmentation & Grad-CAM
    # ---------------------------------------------------------
    refined_img = processed_img.copy()
    refined_img[final_mask > 0] = [0, 0, 255]
    refined_path = f"stage_refined_{filename}"
    cv2.imwrite(os.path.join(output_dir, refined_path), refined_img)

    jet = colormaps.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = cv2.resize(jet_colors[np.uint8(255 * heatmap)], (224, 224))
    overlay = cv2.addWeighted(processed_img, 0.6, (jet_heatmap * 255).astype(np.uint8), 0.4, 0)

    # Draw a green dot showing EXACTLY where the code clicked for SAM
    cv2.circle(overlay, (smart_x, smart_y), 4, (0, 255, 0), -1)

    gradcam_path = f"stage9_{filename}"
    cv2.imwrite(os.path.join(output_dir, gradcam_path), overlay)

    return {
        "prediction": label,
        "confidence": conf,
        "probabilities": {class_labels[i]: float(final_preds[i] * 100) for i in range(4)},
        "stage_images": {
            "roi": roi_path,
            "processed": processed_path,
            "sam_mask": mask_path,
            "superpixel": superpixel_path,
            "graph": graph_path,
            "refined": refined_path,
            "gradcam": gradcam_path
        }
    }