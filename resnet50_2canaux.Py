import os
import time
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D

# === Modèle ResNet50 adapté 2 canaux ===
def build_resnet_model_2channels(input_shape, num_classes):
    inputs = Input(shape=input_shape)  # input_shape = (13, 16, 2)
    base_model = ResNet50(include_top=False, weights=None, input_tensor=inputs)
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# === Chargement fichier .mat ===
FILE_PATH = "nueva_carpeta/Rp_3_25_05_2023_12_00_57_C040_B80_Vacio_LRR.mat"
labels = {"vacio": 0, "1persona": 1, "2personas": 2, "3personas": 3}
X_all, y_all = [], []

print("Chargement du fichier...")
try:
    mat = loadmat(FILE_PATH)
    if "csi_lrr" not in mat:
        raise KeyError("Clé 'csi_lrr' manquante dans le fichier")

    csi_data = mat["csi_lrr"]  # forme attendue : (N, 208) complex
    fname = os.path.basename(FILE_PATH).lower()
    for key in labels:
        if key in fname:
            label = labels[key]
            break
    else:
        raise ValueError(f"Nom de fichier non reconnu : {FILE_PATH}")

    for i in range(csi_data.shape[0]):
        X_all.append(csi_data[i])
        y_all.append(label)

    print(f"{os.path.basename(FILE_PATH)} : {csi_data.shape[0]} échantillons, label {label}")
except Exception as e:
    print(f"Erreur : {e}")
    exit()

# === Préparation des données ===
X_complex = np.array(X_all)  # shape (N, 208), dtype complex
y = np.array(y_all, dtype=np.int64)

# Extraire amplitude + phase pour 2 canaux
amplitude = np.abs(X_complex)          # (N, 208), float
phase = np.angle(X_complex)            # (N, 208), float

# Empiler en 3D (N, 208, 2)
X_stacked = np.stack((amplitude, phase), axis=-1)  # (N, 208, 2)

print(f"Shape X stacked (amplitude + phase): {X_stacked.shape}, Shape y : {y.shape}")

# Normalisation par canal
X_scaled = np.zeros_like(X_stacked)
for c in range(2):
    scaler = StandardScaler()
    # On aplatie sur 2D pour scaler.fit_transform, puis reshape
    X_scaled[..., c] = scaler.fit_transform(X_stacked[..., c])

# Reshape en (N, 13, 16, 2) pour ResNet50 (car 13*16=208)
height, width = 13, 16
X_reshaped = X_scaled.reshape((-1, height, width, 2))
print(f"Input ResNet (avec 2 canaux) : {X_reshaped.shape}")

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, stratify=y, random_state=42)

# === Construction et entraînement du modèle ===
model = build_resnet_model_2channels((height, width, 2), len(np.unique(y)))
model.summary()

start_time = time.time()
history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test), verbose=1)
training_time = time.time() - start_time

# === Évaluation complète ===
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
y_pred = np.argmax(model.predict(X_test), axis=1)

report = classification_report(y_test, y_pred, output_dict=True)
precision = report['weighted avg']['precision']
recall = report['weighted avg']['recall']
f1_score = report['weighted avg']['f1-score']

# Temps d'inférence
start_inf = time.time()
_ = model.predict(np.expand_dims(X_test[0], axis=0))
inference_time_ms = (time.time() - start_inf) * 1000

# Sauvegarde du modèle
model.save("resnet50_model_2channels.keras")
model_size_mb = os.path.getsize("resnet50_model_2channels.keras") / (1024 * 1024)

# Overfitting
train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]
overfitting_gap = abs(train_acc - val_acc)

# === Résumé ===
print("\n=== Évaluation complète du modèle ResNet50 (2 canaux) ===")
print(f"Test Accuracy       : {test_accuracy:.4f}")
print(f"Test Loss           : {test_loss:.4f}")
print(f"Precision           : {precision:.4f}")
print(f"Recall              : {recall:.4f}")
print(f"F1-score            : {f1_score:.4f}")
print(f"Training Time (s)   : {training_time:.2f}")
print(f"Inference Time (ms) : {inference_time_ms:.2f}")
print(f"Model Size (MB)     : {model_size_mb:.2f}")
print(f"Overfitting Gap     : {overfitting_gap:.4f}")
