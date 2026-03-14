import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# -----------------------------
# DATASET PATHS
# -----------------------------
train_dir = "../dataset/train"
test_dir = "../dataset/test"

# -----------------------------
# DATA GENERATORS
# -----------------------------
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # IMPORTANT for confusion matrix
)

# Get class labels automatically
class_labels = list(train_generator.class_indices.keys())

# -----------------------------
# BUILD MODEL (MobileNetV2)
# -----------------------------
base_model = tf.keras.applications.MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(4, activation='softmax')(x)

model = tf.keras.Model(inputs=base_model.input, outputs=outputs)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# TRAIN MODEL
# -----------------------------
model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=5
)

# -----------------------------
# SAVE MODEL
# -----------------------------
os.makedirs("models", exist_ok=True)
model.save("models/brain_tumor_model.h5")

print("Model trained and saved successfully.")

# -----------------------------
# CONFUSION MATRIX
# -----------------------------
print("Generating confusion matrix...")

Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)

cm = confusion_matrix(test_generator.classes, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels,
            yticklabels=class_labels)

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

plt.savefig("confusion_matrix.png")
plt.close()

print("Confusion matrix saved successfully.")