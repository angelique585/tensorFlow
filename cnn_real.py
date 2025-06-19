import os
import time
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dense, Dropout, Flatten

# === Modèle CNN ===
def build_cnn_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv1D(32, 5, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.2)(x)

    x = Conv1D(64, 5, activation='relu', padding='same')(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.2)(x)

    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.2)(x)

    x = Conv1D(64, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# === Chargement des données ===
FILE_PATH = "nueva_carpeta/Rp_3_25_05_2023_12_00_57_C040_B80_Vacio_LRR.mat"
labels = {"vacio": 0, "1persona": 1, "2personas": 2, "3personas": 3}
X_all, y_all = [], []

print("Chargement du fichier...")
try:
    mat = loadmat(FILE_PATH)
    if "csi_lrr" not in mat:
        raise KeyError("Clé 'csi_lrr' manquante dans le fichier")

    csi_data = mat["csi_lrr"]
    filename = os.path.basename(FILE_PATH).lower()
    for key in labels:
        if key in filename:
            label = labels[key]
            break
    else:
        raise ValueError(f"Nom de fichier non reconnu : {FILE_PATH}")

    for i in range(csi_data.shape[0]):
        X_all.append(csi_data[i])
        y_all.append(label)

    print(f"{os.path.basename(FILE_PATH)} : Label {label}, {csi_data.shape[0]} échantillons")
except Exception as e:
    print(f"Erreur : {e}")
    exit()

# === Préparation des données ===
X = np.array(X_all, dtype=np.float32)
y = np.array(y_all, dtype=np.int64)
print(f"Shape X : {X.shape}, Shape y : {y.shape}")
print(f"Classes dans y: {np.unique(y, return_counts=True)}")

samples, features = X.shape
scaler = StandardScaler()
X_scaled_2d = scaler.fit_transform(X)
X_scaled = X_scaled_2d.reshape((samples, features, 1))

# === Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# === Modèle CNN ===
model = build_cnn_model(input_shape=(features, 1), num_classes=4)
model.summary()

# === Entraînement avec mesure du temps
start_time = time.time()
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=64,
    validation_data=(X_test, y_test),
    verbose=1
)
training_time = time.time() - start_time

# === Évaluation
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
y_pred = np.argmax(model.predict(X_test), axis=1)

report = classification_report(y_test, y_pred, output_dict=True)
precision = report['weighted avg']['precision']
recall = report['weighted avg']['recall']
f1_score = report['weighted avg']['f1-score']

# === Inference time
start_inf = time.time()
_ = model.predict(np.expand_dims(X_test[0], axis=0))
inference_time_ms = (time.time() - start_inf) * 1000

# === Taille du modèle
model.save("cnn_model_singlefile.keras")
model_size_mb = os.path.getsize("cnn_model_singlefile.keras") / (1024 * 1024)

# === Overfitting gap
train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]
overfitting_gap = abs(train_acc - val_acc)

# === Résumé des performances
print("\n=== Évaluation complète du modèle CNN ===")
print(f"Test Accuracy       : {test_accuracy:.4f}")
print(f"Test Loss           : {test_loss:.4f}")
print(f"Precision           : {precision:.4f}")
print(f"Recall              : {recall:.4f}")
print(f"F1-score            : {f1_score:.4f}")
print(f"Training Time (s)   : {training_time:.2f}")
print(f"Inference Time (ms) : {inference_time_ms:.2f}")
print(f"Model Size (MB)     : {model_size_mb:.2f}")
print(f"Overfitting Gap     : {overfitting_gap:.4f}")
