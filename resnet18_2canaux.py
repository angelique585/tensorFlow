import os
import time
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Conv2D, GlobalAveragePooling2D,
    ZeroPadding2D, MaxPooling2D, Add, Activation, BatchNormalization
)

# === Construction du modèle ResNet18 pour 2 canaux ===
def conv_block(x, filters, kernel_size=3, strides=1):
    shortcut = x
    x = Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization()(x)

    if strides != 1 or shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, 1, strides=strides, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def build_resnet18_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv2D(3, (3, 3), padding='same', activation='relu')(inputs)

    x = ZeroPadding2D(padding=(3, 3))(x)
    x = Conv2D(64, 7, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(3, strides=2, padding='same')(x)

    x = conv_block(x, 64)
    x = conv_block(x, 64)

    x = conv_block(x, 128, strides=2)
    x = conv_block(x, 128)

    x = conv_block(x, 256, strides=2)
    x = conv_block(x, 256)

    x = conv_block(x, 512, strides=2)
    x = conv_block(x, 512)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
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
        sample = csi_data[i].reshape(13, 16)
        amp = np.abs(sample)
        phase = np.angle(sample)
        two_channel = np.stack([amp, phase], axis=-1)  # (13, 16, 2)
        X_all.append(two_channel)
        y_all.append(label)

    print(f"{os.path.basename(FILE_PATH)} : Label {label}, {csi_data.shape[0]} échantillons")

except Exception as e:
    print(f"Erreur : {e}")
    exit()

# === Préparation des données ===
X = np.array(X_all).astype(np.float32)
y = np.array(y_all, dtype=np.int64)
print(f"Shape X : {X.shape}, Shape y : {y.shape}")

# Normalisation indépendante par canal
scaler = StandardScaler()
for c in range(X.shape[-1]):
    X_channel = X[..., c].reshape(len(X), -1)
    X_scaled = scaler.fit_transform(X_channel)
    X[..., c] = X_scaled.reshape(len(X), 13, 16)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === Entraînement ===
model = build_resnet18_model((13, 16, 2), 4)
start_time = time.time()
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=64,
    validation_data=(X_test, y_test),
    verbose=1
)
training_time = time.time() - start_time

# === Évaluation ===
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
y_pred = np.argmax(model.predict(X_test), axis=1)

report = classification_report(y_test, y_pred, output_dict=True)
precision = report['weighted avg']['precision']
recall = report['weighted avg']['recall']
f1_score = report['weighted avg']['f1-score']

start_inf = time.time()
_ = model.predict(np.expand_dims(X_test[0], axis=0))
inference_time_ms = (time.time() - start_inf) * 1000

model.save("resnet18_model_2canaux.keras")
model_size_mb = os.path.getsize("resnet18_model_2canaux.keras") / (1024 * 1024)

train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]
overfitting_gap = abs(train_acc - val_acc)

# === Résumé des résultats ===
print("\n=== Évaluation complète du modèle ===")
print(f"Test Accuracy       : {test_accuracy:.4f}")
print(f"Test Loss           : {test_loss:.4f}")
print(f"Precision           : {precision:.4f}")
print(f"Recall              : {recall:.4f}")
print(f"F1-score            : {f1_score:.4f}")
print(f"Training Time (s)   : {training_time:.2f}")
print(f"Inference Time (ms) : {inference_time_ms:.2f}")
print(f"Model Size (MB)     : {model_size_mb:.2f}")
print(f"Overfitting Gap     : {overfitting_gap:.4f}")
