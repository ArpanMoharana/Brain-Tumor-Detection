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


# import numpy as np
# import cv2
# import os
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from matplotlib import colormaps
# import matplotlib
#
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
#
# import torch
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv
# from torch_geometric.data import Data
# from segment_anything import sam_model_registry, SamPredictor
# from skimage.segmentation import slic
# from skimage import graph
# from skimage.measure import regionprops
# import networkx as nx
#
# # ==========================================
# # 1. LOAD MODELS
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
# gnn_model.load_state_dict(
#     torch.load("models/gnn_tumor_model_smart.pth", map_location=torch.device('cpu'))
# )
# gnn_model.eval()
#
#
# # ==========================================
# # 2. HELPER FUNCTIONS
# # ==========================================
#
# def get_stages_images(img_path, output_dir):
#     """Stage 1: ROI crop + denoise. Returns roi, processed paths and the processed image array."""
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
#     roi_path = f"stage1_roi_{filename}"
#     cv2.imwrite(os.path.join(output_dir, roi_path), roi_img)
#
#     processed = cv2.resize(roi_img, (224, 224))
#     processed = cv2.fastNlMeansDenoisingColored(processed, None, 10, 10, 7, 21)
#     processed_path = f"stage1_processed_{filename}"
#     cv2.imwrite(os.path.join(output_dir, processed_path), processed)
#
#     return roi_path, processed_path, processed
#
#
# def make_gradcam_heatmap(img_array, model):
#     """Generates Grad-CAM heatmap for the CNN branch."""
#     layer_name = None
#     for layer in reversed(model.layers):
#         if len(layer.output.shape) == 4:
#             layer_name = layer.name
#             break
#
#     grad_model = tf.keras.models.Model(
#         model.inputs,
#         [model.get_layer(layer_name).output, model.output]
#     )
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
# # FIX 1: Morphological Skull Stripping
# # ==========================================
# def get_safe_sam_target(gray_img, cnn_guess):
#     """
#     Uses morphological erosion to strip the skull ring away, then finds the
#     brightest tumor mass STRICTLY inside the safe inner-brain zone.
#     This prevents SAM from clicking the skull or eye sockets.
#     Returns (sam_x, sam_y).
#     """
#     kernel = np.ones((7, 7), np.uint8)
#
#     # Build the skull-stripped safe zone via contour + erosion
#     _, skull_bin = cv2.threshold(gray_img, 15, 255, cv2.THRESH_BINARY)
#     skull_contours, _ = cv2.findContours(
#         skull_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
#     )
#
#     if not skull_contours:
#         return 112, 112  # fallback: dead center
#
#     skull_mask = np.zeros_like(gray_img)
#     cv2.drawContours(
#         skull_mask,
#         [max(skull_contours, key=cv2.contourArea)],
#         -1, 255, -1
#     )
#
#     # 5 iterations of erosion = "acid wash" the skull ring + outer brain tissue away
#     safe_brain_mask = cv2.erode(skull_mask, kernel, iterations=5)
#
#     if cnn_guess == 3:
#         # PITUITARY: restrict further to the strict lower-center zone
#         h, w = gray_img.shape
#         pituitary_mask = np.zeros_like(gray_img)
#         cx, cy = w // 2, h // 2
#         cv2.rectangle(
#             pituitary_mask,
#             (cx - int(w * 0.15), cy),
#             (cx + int(w * 0.15), cy + int(h * 0.30)),
#             255, -1
#         )
#         safe_brain_mask = cv2.bitwise_and(safe_brain_mask, pituitary_mask)
#
#     elif cnn_guess == 2:
#         # NO TUMOR (Hunter Mode): use the full safe zone, wide blur to find any hidden mass
#         pass  # safe_brain_mask already covers the whole inner brain
#
#     # else GLIOMA / MENINGIOMA: trust the safe zone as-is
#
#     # Find the brightest region inside the safe zone
#     safe_region = cv2.bitwise_and(gray_img, gray_img, mask=safe_brain_mask)
#
#     # Use a large blur to pool scattered brightness into a single hotspot
#     blur_ksize = 45 if cnn_guess == 2 else 21
#     blurred = cv2.GaussianBlur(safe_region, (blur_ksize, blur_ksize), 0)
#     _, _, _, max_loc = cv2.minMaxLoc(blurred)
#
#     # Verify the target point isn't on a dark background pixel
#     sam_x, sam_y = max_loc
#     if gray_img[sam_y, sam_x] < 20:
#         sam_x, sam_y = 112, 112  # fallback if nothing bright found
#
#     return sam_x, sam_y
#
#
# # ==========================================
# # FIX 2 + 3: Superpixel (colorful patches) + Graph (drawn on MRI)
# # ==========================================
# def build_graph_and_visualize(image, output_dir, filename):
#     """
#     Generates:
#     - Colorful SLIC superpixel image (label2rgb avg colors, not yellow grid lines)
#     - Graph overlaid ON TOP of the MRI brain image (cyan nodes, magenta edges)
#     Returns graph Data, superpixel_path, graph_path.
#     """
#     gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#     # Fewer segments = more readable graph (100 instead of 250)
#     segments = slic(gray_img, n_segments=100, compactness=0.5, channel_axis=None)
#     num_nodes = segments.max() + 1
#
#     # ---------------------------------------------------------
#     # FIX 2: Vivid random-color per region + MRI blend → "stained glass" look
#     # label2rgb produces near-gray output on grayscale MRIs, so use random colors
#     # ---------------------------------------------------------
#     np.random.seed(42)
#     rand_colors = np.random.randint(40, 255, size=(num_nodes, 3), dtype=np.uint8)
#     colored_map = rand_colors[segments]                    # (H, W, 3) in RGB
#     colored_bgr = cv2.cvtColor(colored_map, cv2.COLOR_RGB2BGR)
#     # Blend: 65% vivid color + 35% original MRI so brain structure shows through
#     colored_blend = cv2.addWeighted(colored_bgr, 0.65, image, 0.35, 0)
#
#     superpixel_path = f"stage_superpixel_{filename}"
#     cv2.imwrite(os.path.join(output_dir, superpixel_path), colored_blend)
#
#     # Build RAG edges for GNN
#     rag = graph.rag_mean_color(rgb_img, segments)
#     edges = np.array(rag.edges)
#
#     if len(edges) > 0:
#         edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
#         edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
#     else:
#         edge_index = torch.empty((2, 0), dtype=torch.long)
#
#     # Extract 16 regionprop features per superpixel node
#     regions = regionprops(segments, intensity_image=gray_img)
#     x_features = np.zeros((num_nodes, 16), dtype=np.float32)
#
#     for props in regions:
#         idx = props.label
#         if idx < num_nodes:
#             x_features[idx, 0]  = props.centroid[0] / 224.0
#             x_features[idx, 1]  = props.centroid[1] / 224.0
#             x_features[idx, 2]  = props.area / (224.0 * 224.0)
#             x_features[idx, 3]  = props.mean_intensity / 255.0
#             x_features[idx, 4]  = props.eccentricity
#             x_features[idx, 5]  = props.extent
#             x_features[idx, 6]  = props.solidity
#             x_features[idx, 7]  = props.perimeter / 224.0
#             x_features[idx, 8]  = props.major_axis_length / 224.0
#             x_features[idx, 9]  = props.minor_axis_length / 224.0
#             x_features[idx, 10] = props.orientation
#             minr, minc, maxr, maxc = props.bbox
#             x_features[idx, 11] = minr / 224.0
#             x_features[idx, 12] = minc / 224.0
#             x_features[idx, 13] = maxr / 224.0
#             x_features[idx, 14] = maxc / 224.0
#             x_features[idx, 15] = props.equivalent_diameter_area / 224.0
#
#     x = torch.tensor(x_features, dtype=torch.float)
#
#     # ---------------------------------------------------------
#     # FIX 3: Draw graph nodes + edges ON TOP of the MRI image
#     # Matches the reference: cyan nodes, magenta edges over brain
#     # ---------------------------------------------------------
#     fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
#     ax.imshow(rgb_img, cmap='gray')                    # MRI as background
#
#     centroids = {
#         props.label: (props.centroid[1], props.centroid[0])
#         for props in regions
#     }
#
#     nx.draw_networkx_nodes(
#         rag, centroids,
#         node_size=10,
#         node_color='cyan',
#         alpha=0.85,
#         ax=ax
#     )
#     nx.draw_networkx_edges(
#         rag, centroids,
#         alpha=0.35,
#         edge_color='magenta',
#         width=0.6,
#         ax=ax
#     )
#
#     ax.axis('off')
#     plt.tight_layout(pad=0)
#
#     graph_path = f"stage_graph_{filename}"
#     fig.savefig(
#         os.path.join(output_dir, graph_path),
#         bbox_inches='tight', pad_inches=0
#     )
#     plt.close(fig)
#
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
#     # Stage 1: ROI + Pre-processing
#     roi_path, processed_path, processed_img = get_stages_images(img_path, output_dir)
#
#     # ---------------------------------------------------------
#     # BRANCH 1: MobileNetV2 (Texture Engine)
#     # ---------------------------------------------------------
#     img_array = (processed_img / 255.0).reshape(1, 224, 224, 3)
#     cnn_preds = cnn_model.predict(img_array)[0]
#
#     heatmap = make_gradcam_heatmap(img_array, cnn_model)
#     max_y, max_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
#     scale_factor = 224 / heatmap.shape[0]
#     gradcam_x = min(int(max_x * scale_factor), 223)
#     gradcam_y = min(int(max_y * scale_factor), 223)
#
#     cnn_guess = np.argmax(cnn_preds)
#
#     # ---------------------------------------------------------
#     # FIX 1: Morphological Skull Stripping for SAM target
#     # ---------------------------------------------------------
#     gray_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
#
#     # For Glioma/Meningioma, blend Grad-CAM hotspot with safe-zone search.
#     # For Pituitary/No-Tumor, rely fully on the safe-zone morphological targeting.
#     if cnn_guess in [0, 1]:
#         # Grad-CAM is reliable for Glioma/Meningioma — use it but safety-check it
#         sam_x, sam_y = gradcam_x, gradcam_y
#         if gray_img[sam_y, sam_x] < 20:
#             # Hotspot landed on dark background — fall back to skull-stripped search
#             sam_x, sam_y = get_safe_sam_target(gray_img, cnn_guess)
#     else:
#         # Pituitary or No-Tumor: always use skull-stripped safe targeting
#         sam_x, sam_y = get_safe_sam_target(gray_img, cnn_guess)
#
#     print(f"SAM target point → x={sam_x}, y={sam_y}  (CNN guess: {class_labels[cnn_guess]})")
#
#     # ---------------------------------------------------------
#     # BRANCH 2: SAM Segmentation
#     # ---------------------------------------------------------
#     image_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
#     sam_predictor.set_image(image_rgb)
#
#     # Positive: tumor hotspot | Negative: 4 skull-edge anchors
#     input_point = np.array([
#         [sam_x, sam_y],
#         [112, 10], [112, 213], [10, 112], [213, 112],
#     ])
#     input_label = np.array([1, 0, 0, 0, 0])
#
#     sam_masks, scores, _ = sam_predictor.predict(
#         point_coords=input_point,
#         point_labels=input_label,
#         multimask_output=True
#     )
#
#     # Pick highest-scoring mask under 15% of image area to avoid whole-head floods
#     MAX_MASK_AREA = 224 * 224 * 0.15
#     valid = [(sam_masks[i], scores[i]) for i in range(len(sam_masks))
#              if np.sum(sam_masks[i]) < MAX_MASK_AREA]
#
#     if valid:
#         best_mask = max(valid, key=lambda x: x[1])[0]
#     else:
#         areas = [np.sum(sam_masks[i]) for i in range(len(sam_masks))]
#         best_mask = sam_masks[int(np.argmin(areas))]
#
#     final_mask = (best_mask * 255).astype(np.uint8)
#     mask_path = f"stage_sam_mask_{filename}"
#     cv2.imwrite(os.path.join(output_dir, mask_path), final_mask)
#
#     # ---------------------------------------------------------
#     # BRANCH 2B: GNN (Graph Neural Network — Shape Engine)
#     # ---------------------------------------------------------
#     graph_data, superpixel_path, graph_path = build_graph_and_visualize(
#         processed_img, output_dir, filename
#     )
#
#     with torch.no_grad():
#         gnn_raw = gnn_model(graph_data.x, graph_data.edge_index)
#         gnn_pooled = torch.mean(gnn_raw, dim=0)
#         gnn_preds = F.softmax(gnn_pooled, dim=0).numpy()
#
#     print("\n" + "=" * 50)
#     print("DUAL-BRAIN AI DIAGNOSTIC LOG")
#     print(f"Classes        : {class_labels}")
#     print(f"MobileNetV2    : {np.round(cnn_preds * 100, 2)}%")
#     print(f"GNN (Shape)    : {np.round(gnn_preds * 100, 2)}%")
#     print(f"SAM mask area  : {np.sum(final_mask > 0)} px")
#     print("=" * 50 + "\n")
#
#     # ---------------------------------------------------------
#     # STAGE: Clinical Geometry Veto System
#     # ---------------------------------------------------------
#     mask_area = np.sum(final_mask > 0)
#
#     # If SAM found a meaningful mass but CNN says "No Tumor" — override CNN
#     if mask_area > 300:
#         cnn_preds[2] = 0.001  # suppress "No Tumor"
#
#         # Spatial boost: lower-center region = Pituitary zone
#         if sam_y > 115 and 80 < sam_x < 144:
#             cnn_preds[3] += 0.60
#
#         cnn_preds = cnn_preds / np.sum(cnn_preds)
#
#     # NOTE: Keep GNN at 0.0 weight until you train with real BraTS data.
#     # Once trained, change to: (cnn_preds * 0.70) + (gnn_preds * 0.30)
#     final_preds = (cnn_preds * 0.70) + (gnn_preds * 0.30)
#
#     idx = np.argmax(final_preds)
#     label = class_labels[idx]
#     conf = float(final_preds[idx] * 100)
#
#     # ---------------------------------------------------------
#     # FINAL VISUALS
#     # ---------------------------------------------------------
#
#     # FIX 4 flows from Fix 1: tight mask → tight red overlay (no extra code needed)
#     refined_img = processed_img.copy()
#     refined_img[final_mask > 0] = [0, 0, 255]   # BGR red on tumor pixels only
#     refined_path = f"stage_refined_{filename}"
#     cv2.imwrite(os.path.join(output_dir, refined_path), refined_img)
#
#     # Grad-CAM overlay with green dot at SAM target
#     jet = colormaps.get_cmap("jet")
#     jet_colors = jet(np.arange(256))[:, :3]
#     jet_heatmap = cv2.resize(jet_colors[np.uint8(255 * heatmap)], (224, 224))
#     overlay = cv2.addWeighted(
#         processed_img, 0.6,
#         (jet_heatmap * 255).astype(np.uint8), 0.4,
#         0
#     )
#     cv2.circle(overlay, (sam_x, sam_y), 5, (0, 255, 0), -1)
#     gradcam_path = f"stage_gradcam_{filename}"
#     cv2.imwrite(os.path.join(output_dir, gradcam_path), overlay)
#
#     return {
#         "prediction": label,
#         "confidence": conf,
#         "probabilities": {
#             class_labels[i]: float(final_preds[i] * 100) for i in range(4)
#         },
#         "stage_images": {
#             "roi":        roi_path,
#             "processed":  processed_path,
#             "sam_mask":   mask_path,
#             "superpixel": superpixel_path,
#             "graph":      graph_path,
#             "refined":    refined_path,
#             "gradcam":    gradcam_path
#         }
#     }



import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from matplotlib import colormaps
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from segment_anything import sam_model_registry, SamPredictor
from skimage.segmentation import slic
from skimage import graph
from skimage.measure import regionprops
import networkx as nx

# ==========================================
# 1. LOAD MODELS
# ==========================================
class_labels = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

cnn_model = load_model("models/brain_tumor_model.h5")

sam_checkpoint = "models/sam_vit_h.pth"
sam = sam_model_registry['vit_h'](checkpoint=sam_checkpoint)
sam_predictor = SamPredictor(sam)


class TumorGNN(torch.nn.Module):
    def __init__(self, in_features=16, num_classes=4):
        super().__init__()
        self.c1      = GCNConv(in_features, 64)
        self.c2      = GCNConv(64, 128)
        self.c3      = GCNConv(128, 64)
        self.dropout = torch.nn.Dropout(0.3)
        self.fc      = torch.nn.Linear(64, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.c1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.c2(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.c3(x, edge_index))
        # Global mean pool → single graph-level prediction vector
        x = x.mean(dim=0, keepdim=True)
        return self.fc(x)


gnn_model = TumorGNN(in_features=16, num_classes=4)
gnn_model.load_state_dict(
    torch.load("models/gnn_tumor_model_smart.pth", map_location=torch.device('cpu'))
)
gnn_model.eval()


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def get_stages_images(img_path, output_dir):
    """Stage 1: ROI crop + denoise. Returns roi, processed paths and the processed image array."""
    img = cv2.imread(img_path)
    filename = os.path.basename(img_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    roi_img = img[y:y + h, x:x + w]
    roi_path = f"stage1_roi_{filename}"
    cv2.imwrite(os.path.join(output_dir, roi_path), roi_img)

    processed = cv2.resize(roi_img, (224, 224))
    processed = cv2.fastNlMeansDenoisingColored(processed, None, 10, 10, 7, 21)
    processed_path = f"stage1_processed_{filename}"
    cv2.imwrite(os.path.join(output_dir, processed_path), processed)

    return roi_path, processed_path, processed


def make_gradcam_heatmap(img_array, model):
    """Generates Grad-CAM heatmap for the CNN branch."""
    layer_name = None
    for layer in reversed(model.layers):
        if len(layer.output.shape) == 4:
            layer_name = layer.name
            break

    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(layer_name).output, model.output]
    )
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


# ==========================================
# FIX 1: Morphological Skull Stripping
# ==========================================
def get_safe_sam_target(gray_img, cnn_guess):
    """
    Uses morphological erosion to strip the skull ring away, then finds the
    brightest tumor mass STRICTLY inside the safe inner-brain zone.
    This prevents SAM from clicking the skull or eye sockets.
    Returns (sam_x, sam_y).
    """
    kernel = np.ones((7, 7), np.uint8)

    # Build the skull-stripped safe zone via contour + erosion
    _, skull_bin = cv2.threshold(gray_img, 15, 255, cv2.THRESH_BINARY)
    skull_contours, _ = cv2.findContours(
        skull_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not skull_contours:
        return 112, 112  # fallback: dead center

    skull_mask = np.zeros_like(gray_img)
    cv2.drawContours(
        skull_mask,
        [max(skull_contours, key=cv2.contourArea)],
        -1, 255, -1
    )

    # 5 iterations of erosion = "acid wash" the skull ring + outer brain tissue away
    safe_brain_mask = cv2.erode(skull_mask, kernel, iterations=5)

    if cnn_guess == 3:
        # PITUITARY: restrict further to the strict lower-center zone
        h, w = gray_img.shape
        pituitary_mask = np.zeros_like(gray_img)
        cx, cy = w // 2, h // 2
        cv2.rectangle(
            pituitary_mask,
            (cx - int(w * 0.15), cy),
            (cx + int(w * 0.15), cy + int(h * 0.30)),
            255, -1
        )
        safe_brain_mask = cv2.bitwise_and(safe_brain_mask, pituitary_mask)

    elif cnn_guess == 2:
        # NO TUMOR (Hunter Mode): use the full safe zone, wide blur to find any hidden mass
        pass  # safe_brain_mask already covers the whole inner brain

    # else GLIOMA / MENINGIOMA: trust the safe zone as-is

    # Find the brightest region inside the safe zone
    safe_region = cv2.bitwise_and(gray_img, gray_img, mask=safe_brain_mask)

    # Use a large blur to pool scattered brightness into a single hotspot
    blur_ksize = 45 if cnn_guess == 2 else 21
    blurred = cv2.GaussianBlur(safe_region, (blur_ksize, blur_ksize), 0)
    _, _, _, max_loc = cv2.minMaxLoc(blurred)

    # Verify the target point isn't on a dark background pixel
    sam_x, sam_y = max_loc
    if gray_img[sam_y, sam_x] < 20:
        sam_x, sam_y = 112, 112  # fallback if nothing bright found

    return sam_x, sam_y


# ==========================================
# FIX 2 + 3: Superpixel (colorful patches) + Graph (drawn on MRI)
# ==========================================
def build_graph_and_visualize(image, output_dir, filename):
    """
    Generates:
    - Colorful SLIC superpixel image (label2rgb avg colors, not yellow grid lines)
    - Graph overlaid ON TOP of the MRI brain image (cyan nodes, magenta edges)
    Returns graph Data, superpixel_path, graph_path.
    """
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Fewer segments = more readable graph (100 instead of 250)
    # segments = slic(gray_img, n_segments=100, compactness=0.5, channel_axis=None)
    segments = slic(gray_img, n_segments=200, compactness=0.5, channel_axis=None)
    num_nodes = segments.max() + 1

    # ---------------------------------------------------------
    # FIX 2: Vivid random-color per region + MRI blend → "stained glass" look
    # label2rgb produces near-gray output on grayscale MRIs, so use random colors
    # ---------------------------------------------------------
    np.random.seed(42)
    rand_colors = np.random.randint(40, 255, size=(num_nodes, 3), dtype=np.uint8)
    colored_map = rand_colors[segments]                    # (H, W, 3) in RGB
    colored_bgr = cv2.cvtColor(colored_map, cv2.COLOR_RGB2BGR)
    # Blend: 65% vivid color + 35% original MRI so brain structure shows through
    colored_blend = cv2.addWeighted(colored_bgr, 0.65, image, 0.35, 0)

    superpixel_path = f"stage_superpixel_{filename}"
    cv2.imwrite(os.path.join(output_dir, superpixel_path), colored_blend)

    # Build RAG edges for GNN
    rag = graph.rag_mean_color(rgb_img, segments)
    edges = np.array(rag.edges)

    if len(edges) > 0:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    # Extract 16 regionprop features per superpixel node
    regions = regionprops(segments, intensity_image=gray_img)
    x_features = np.zeros((num_nodes, 16), dtype=np.float32)

    for props in regions:
        idx = props.label
        if idx < num_nodes:
            x_features[idx, 0]  = props.centroid[0] / 224.0
            x_features[idx, 1]  = props.centroid[1] / 224.0
            x_features[idx, 2]  = props.area / (224.0 * 224.0)
            x_features[idx, 3]  = props.mean_intensity / 255.0
            x_features[idx, 4]  = props.eccentricity
            x_features[idx, 5]  = props.extent
            x_features[idx, 6]  = props.solidity
            x_features[idx, 7]  = props.perimeter / 224.0
            x_features[idx, 8]  = props.major_axis_length / 224.0
            x_features[idx, 9]  = props.minor_axis_length / 224.0
            x_features[idx, 10] = props.orientation
            minr, minc, maxr, maxc = props.bbox
            x_features[idx, 11] = minr / 224.0
            x_features[idx, 12] = minc / 224.0
            x_features[idx, 13] = maxr / 224.0
            x_features[idx, 14] = maxc / 224.0
            x_features[idx, 15] = props.equivalent_diameter_area / 224.0

    x = torch.tensor(x_features, dtype=torch.float)

    # ---------------------------------------------------------
    # FIX 3: Draw graph nodes + edges ON TOP of the MRI image
    # Matches the reference: cyan nodes, magenta edges over brain
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    ax.imshow(rgb_img, cmap='gray')                    # MRI as background

    centroids = {
        props.label: (props.centroid[1], props.centroid[0])
        for props in regions
    }

    nx.draw_networkx_nodes(
        rag, centroids,
        node_size=10,
        node_color='cyan',
        alpha=0.85,
        ax=ax
    )
    nx.draw_networkx_edges(
        rag, centroids,
        alpha=0.35,
        edge_color='magenta',
        width=0.6,
        ax=ax
    )

    ax.axis('off')
    plt.tight_layout(pad=0)

    graph_path = f"stage_graph_{filename}"
    fig.savefig(
        os.path.join(output_dir, graph_path),
        bbox_inches='tight', pad_inches=0
    )
    plt.close(fig)

    return Data(x=x, edge_index=edge_index), superpixel_path, graph_path


# ==========================================
# 3. MAIN PREDICTION PIPELINE
# ==========================================
def predict_image(img_path):
    output_dir = os.path.dirname(img_path)
    filename = os.path.basename(img_path)

    # Stage 1: ROI + Pre-processing
    roi_path, processed_path, processed_img = get_stages_images(img_path, output_dir)

    # ---------------------------------------------------------
    # BRANCH 1: MobileNetV2 (Texture Engine)
    # ---------------------------------------------------------
    img_array = (processed_img / 255.0).reshape(1, 224, 224, 3)
    cnn_preds = cnn_model.predict(img_array)[0]

    heatmap = make_gradcam_heatmap(img_array, cnn_model)
    max_y, max_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    scale_factor = 224 / heatmap.shape[0]
    gradcam_x = min(int(max_x * scale_factor), 223)
    gradcam_y = min(int(max_y * scale_factor), 223)

    cnn_guess = np.argmax(cnn_preds)

    # ---------------------------------------------------------
    # FIX 1: Morphological Skull Stripping for SAM target
    # ---------------------------------------------------------
    gray_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)

    # For Glioma/Meningioma, blend Grad-CAM hotspot with safe-zone search.
    # For Pituitary/No-Tumor, rely fully on the safe-zone morphological targeting.
    if cnn_guess in [0, 1]:
        # Grad-CAM is reliable for Glioma/Meningioma — use it but safety-check it
        sam_x, sam_y = gradcam_x, gradcam_y
        if gray_img[sam_y, sam_x] < 20:
            # Hotspot landed on dark background — fall back to skull-stripped search
            sam_x, sam_y = get_safe_sam_target(gray_img, cnn_guess)
    else:
        # Pituitary or No-Tumor: always use skull-stripped safe targeting
        sam_x, sam_y = get_safe_sam_target(gray_img, cnn_guess)

    print(f"SAM target point → x={sam_x}, y={sam_y}  (CNN guess: {class_labels[cnn_guess]})")

    # ---------------------------------------------------------
    # BRANCH 2: SAM Segmentation
    # ---------------------------------------------------------
    image_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
    sam_predictor.set_image(image_rgb)

    # Positive: tumor hotspot | Negative: 4 skull-edge anchors
    input_point = np.array([
        [sam_x, sam_y],
        [112, 10], [112, 213], [10, 112], [213, 112],
    ])
    input_label = np.array([1, 0, 0, 0, 0])

    sam_masks, scores, _ = sam_predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True
    )

    # Pick highest-scoring mask under 15% of image area to avoid whole-head floods
    MAX_MASK_AREA = 224 * 224 * 0.15
    valid = [(sam_masks[i], scores[i]) for i in range(len(sam_masks))
             if np.sum(sam_masks[i]) < MAX_MASK_AREA]

    if valid:
        best_mask = max(valid, key=lambda x: x[1])[0]
    else:
        areas = [np.sum(sam_masks[i]) for i in range(len(sam_masks))]
        best_mask = sam_masks[int(np.argmin(areas))]

    final_mask = (best_mask * 255).astype(np.uint8)
    mask_path = f"stage_sam_mask_{filename}"
    cv2.imwrite(os.path.join(output_dir, mask_path), final_mask)

    # ---------------------------------------------------------
    # BRANCH 2B: GNN (Graph Neural Network — Shape Engine)
    # ---------------------------------------------------------
    graph_data, superpixel_path, graph_path = build_graph_and_visualize(
        processed_img, output_dir, filename
    )

    with torch.no_grad():
        gnn_out   = gnn_model(graph_data.x, graph_data.edge_index)  # shape (1, 4)
        gnn_preds = F.softmax(gnn_out[0], dim=0).numpy()

    print("\n" + "=" * 50)
    print("DUAL-BRAIN AI DIAGNOSTIC LOG")
    print(f"Classes        : {class_labels}")
    print(f"MobileNetV2    : {np.round(cnn_preds * 100, 2)}%")
    print(f"GNN (Shape)    : {np.round(gnn_preds * 100, 2)}%")
    print(f"SAM mask area  : {np.sum(final_mask > 0)} px")
    print("=" * 50 + "\n")

    # ---------------------------------------------------------
    # STAGE: Clinical Geometry Veto System (upgraded)
    # ---------------------------------------------------------
    mask_area = np.sum(final_mask > 0)

    print(f"Veto check     : mask_area={mask_area}  sam_x={sam_x}  sam_y={sam_y}")
    print(f"CNN pre-veto   : {np.round(cnn_preds * 100, 2)}%")

    if mask_area > 50:
        # ── Rule 1: SAM found a real mass → CNN "No Tumor" is wrong, suppress it
        cnn_preds[2] = 0.001

        # ── Rule 2: Spatial zone boosts
        # Pituitary zone: lower-center of brain
        # if sam_y > 115 and 75 < sam_x < 149:
        #     cnn_preds[3] += 0.55
        #     print("Veto           : Pituitary spatial boost applied")

        if sam_y > 60 and 60 < sam_x < 164:
            # cnn_preds[3] += 0.55
            cnn_preds[3] += 1.20
            print("Veto           : Pituitary spatial boost applied")

        # Glioma zone: upper or lateral brain (outside center box)
        # Gliomas tend to be in cerebral hemispheres, not center structures
        elif not (75 < sam_x < 149 and 75 < sam_y < 149):
            cnn_preds[0] += 0.40
            print("Veto           : Glioma lateral-zone boost applied")

        # Meningioma zone: near brain surface / skull margin
        # If the SAM point is close to the brain edge (within outer 25% of image)
        edge_dist = min(sam_x, sam_y, 223 - sam_x, 223 - sam_y)
        if edge_dist < 56:   # within ~25% of any edge
            cnn_preds[1] += 0.35
            print("Veto           : Meningioma edge-zone boost applied")

        # ── Rule 3: Large mask area (> 5000px) strongly suggests Glioma
        # Meningiomas are typically smaller and more compact
        if mask_area > 5000:
            cnn_preds[0] += 0.30
            print(f"Veto           : Large mask ({mask_area}px) Glioma boost applied")

        cnn_preds = np.clip(cnn_preds, 0, None)
        cnn_preds = cnn_preds / np.sum(cnn_preds)

    print(f"CNN post-veto  : {np.round(cnn_preds * 100, 2)}%")

    # GNN trained on real data — activate at 30% weight
    # final_preds = (cnn_preds * 0.70) + (gnn_preds * 0.30)
    final_preds = (cnn_preds * 0.60) + (gnn_preds * 0.40)

    idx = np.argmax(final_preds)
    label = class_labels[idx]
    conf = float(final_preds[idx] * 100)

    # ---------------------------------------------------------
    # FINAL VISUALS
    # ---------------------------------------------------------

    # FIX 4 flows from Fix 1: tight mask → tight red overlay (no extra code needed)
    refined_img = processed_img.copy()
    refined_img[final_mask > 0] = [0, 0, 255]   # BGR red on tumor pixels only
    refined_path = f"stage_refined_{filename}"
    cv2.imwrite(os.path.join(output_dir, refined_path), refined_img)

    # Grad-CAM overlay with green dot at SAM target
    jet = colormaps.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = cv2.resize(jet_colors[np.uint8(255 * heatmap)], (224, 224))
    overlay = cv2.addWeighted(
        processed_img, 0.6,
        (jet_heatmap * 255).astype(np.uint8), 0.4,
        0
    )
    cv2.circle(overlay, (sam_x, sam_y), 5, (0, 255, 0), -1)
    gradcam_path = f"stage_gradcam_{filename}"
    cv2.imwrite(os.path.join(output_dir, gradcam_path), overlay)

    return {
        "prediction": label,
        "confidence": conf,
        "probabilities": {
            class_labels[i]: float(final_preds[i] * 100) for i in range(4)
        },
        "stage_images": {
            "roi":        roi_path,
            "processed":  processed_path,
            "sam_mask":   mask_path,
            "superpixel": superpixel_path,
            "graph":      graph_path,
            "refined":    refined_path,
            "gradcam":    gradcam_path
        }
    }