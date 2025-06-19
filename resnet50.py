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
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D, Conv2D

# === Modèle ResNet50 ===
def build_resnet_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv2D(3, (3, 3), padding='same', activation='relu')(inputs)  # Convertir 1 canal en 3 canaux
    base_model = ResNet50(include_top=False, weights=None, input_tensor=x)
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

    csi_data = mat["csi_lrr"]
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
X = np.abs(np.array(X_all)).astype(np.float32)
y = np.array(y_all, dtype=np.int64)
print(f"Shape X : {X.shape}, Shape y : {y.shape}")
print(f"Classes : {np.unique(y, return_counts=True)}")

# Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.reshape(len(X), -1))

# Reshape pour ResNet50
height, width = 13, 16  # car 13 × 16 = 208
X_reshaped = X_scaled.reshape((-1, height, width, 1))
print(f"Input ResNet : {X_reshaped.shape}")

# Split
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, stratify=y, random_state=42)

# === Modèle ===
model = build_resnet_model(input_shape=(height, width, 1), num_classes=4)
model.summary()

# === Entraînement avec chrono
start_time = time.time()
history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test), verbose=1)
training_time = time.time() - start_time

# === Évaluation complète
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
y_pred = np.argmax(model.predict(X_test), axis=1)

report = classification_report(y_test, y_pred, output_dict=True)
precision = report['weighted avg']['precision']
recall = report['weighted avg']['recall']
f1_score = report['weighted avg']['f1-score']

# Inference time
start_inf = time.time()
_ = model.predict(np.expand_dims(X_test[0], axis=0))
inference_time_ms = (time.time() - start_inf) * 1000

# Sauvegarde du modèle
model.save("resnet50_model_singlefile.keras")
model_size_mb = os.path.getsize("resnet50_model_singlefile.keras") / (1024 * 1024)

# Overfitting
train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]
overfitting_gap = abs(train_acc - val_acc)

# === Résumé
print("\n=== Évaluation complète du modèle ResNet50 ===")
print(f"Test Accuracy       : {test_accuracy:.4f}")
print(f"Test Loss           : {test_loss:.4f}")
print(f"Precision           : {precision:.4f}")
print(f"Recall              : {recall:.4f}")
print(f"F1-score            : {f1_score:.4f}")
print(f"Training Time (s)   : {training_time:.2f}")
print(f"Inference Time (ms) : {inference_time_ms:.2f}")
print(f"Model Size (MB)     : {model_size_mb:.2f}")
print(f"Overfitting Gap     : {overfitting_gap:.4f}")
