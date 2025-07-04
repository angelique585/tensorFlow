import numpy as np
from scipy.io import loadmat
import os
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import f1_score, confusion_matrix, cohen_kappa_score, classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dense, Dropout, Flatten
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

class CSIDataProcessor:
    """Classe pour traiter les données CSI sans PyTorch"""
    
    def __init__(self, data_path, class_names, n_symbols=None, n_subcarriers=None, 
                 norm_type='standard', script_name='csi_model', timestamp=None):
        self.data_path = data_path
        self.class_names = class_names
        self.n_symbols = n_symbols
        self.n_subcarriers = n_subcarriers
        self.norm_type = norm_type
        self.script_name = script_name
        self.timestamp = timestamp or time.strftime("%Y%m%d_%H%M%S")
        self.scaler = None
        
        print(f"RUTA UTILIZADA: {os.path.basename(os.path.dirname(os.path.dirname(data_path)))} --> "
              f"{os.path.basename(os.path.dirname(data_path))} --> {os.path.basename(data_path)}")
    
    def _create_scaler(self, data):
        """Crée et ajuste le scaler sur les données"""
        if self.norm_type == 'standard':
            self.scaler = StandardScaler()
        elif self.norm_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = None
            return data
        
        # Reshape pour le scaler si nécessaire
        original_shape = data.shape
        if len(original_shape) > 2:
            data_reshaped = data.reshape(original_shape[0], -1)
        else:
            data_reshaped = data
            
        data_scaled = self.scaler.fit_transform(data_reshaped)
        
        # Revenir à la forme originale
        if len(original_shape) > 2:
            data_scaled = data_scaled.reshape(original_shape)
        
        print(f"Transformación de data: {self.norm_type}")
        return data_scaled
    
    def _save_scaler(self):
        """Sauvegarde le scaler"""
        if self.scaler is not None:
            scaler_filename = f"scalers_{self.script_name}_{self.norm_type}_{self.timestamp}.pkl"
            joblib.dump(self.scaler, scaler_filename)
            print(f"Scaler sauvé : {scaler_filename}")
    
    def load_and_process_data(self):
        """Charge et traite tous les fichiers .mat"""
        print(f"\n=== CHARGEMENT DE TOUS LES FICHIERS .MAT ===")
        
        X_all, y_all = [], []
        file_stats = {}
        class_counts = {i: 0 for i in range(len(self.class_names))}
        
        # Définir les labels basés sur les noms de classes
        labels = {}
        for i, class_name in enumerate(self.class_names):
            labels[class_name.lower()] = i
        
        # Obtenir tous les fichiers .mat dans le dossier
        mat_files = [f for f in os.listdir(self.data_path) if f.endswith('.mat')]
        
        print("Fichiers trouvés:")
        for f in mat_files:
            print(f"  - {f}")
        
        subcarriers = None
        
        for filename in mat_files:
            file_path = os.path.join(self.data_path, filename)
            print(f"\n Traitement : {filename}")
            
            try:
                # Charger le fichier .mat
                mat = loadmat(file_path)
                # Prendre la clé qui contient les données (généralement la dernière clé non-système)
                data_keys = [k for k in mat.keys() if not k.startswith('__')]
                if len(data_keys) < 1:
                    print(f"     Aucune donnée trouvée - fichier ignoré")
                    continue
                    
                csi_data = mat[data_keys[-1]]  # Prendre la dernière clé
                filename_lower = filename.lower()
                
                # Déterminer le label
                label = None
                matched_key = None
                for key in labels:
                    if key in filename_lower:
                        label = labels[key]
                        matched_key = key
                        break
                
                if label is None:
                    print(f"     Aucun label reconnu dans le nom - fichier ignoré")
                    continue

                # Vérifier et ajuster la forme des données
                if csi_data.shape[1] > csi_data.shape[0]:
                    csi_data = csi_data.T
                
                print(f"    Forme des données: {csi_data.shape}")
                
                # Stocker le nombre de subcarriers
                if subcarriers is None:
                    subcarriers = csi_data.shape[1]
                    self.n_subcarriers = subcarriers
                    print(f"    Subcarriers détectés: {subcarriers}")
                
                # Si n_symbols n'est pas défini, prendre toutes les données
                if self.n_symbols is None:
                    # Ajouter toutes les données
                    for i in range(csi_data.shape[0]):
                        X_all.append(csi_data[i])
                        y_all.append(label)
                    num_samples = csi_data.shape[0]
                else:
                    # Diviser en chunks de n_symbols
                    n_images = csi_data.shape[0] // self.n_symbols
                    csi_data = csi_data[:n_images * self.n_symbols]
                    
                    for i in range(n_images):
                        start_idx = i * self.n_symbols
                        end_idx = start_idx + self.n_symbols
                        chunk = csi_data[start_idx:end_idx]
                        # Aplatir le chunk pour avoir un vecteur 1D
                        X_all.append(chunk.flatten())
                        y_all.append(label)
                    
                    num_samples = n_images
                
                # Statistiques
                class_counts[label] += num_samples
                file_stats[filename] = {
                    'label': label,
                    'matched_key': matched_key,
                    'samples': num_samples,
                    'shape': csi_data.shape
                }
                
                print(f"    Label: {label} ('{matched_key}') - {num_samples} échantillons")
                
            except Exception as e:
                print(f"    Erreur lors du chargement : {e}")
                continue

        # === VÉRIFICATION DES DONNÉES CHARGÉES ===
        print(f"\n=== RÉSUMÉ DU CHARGEMENT ===")
        print(f"Total d'échantillons chargés : {len(X_all)}")
        print(f"Répartition par classe :")
        
        for class_id, count in class_counts.items():
            if count > 0:
                percentage = (count / len(X_all)) * 100
                print(f"  Classe {class_id} ({self.class_names[class_id]}): {count} échantillons ({percentage:.1f}%)")

        # Vérifier qu'on a au moins 2 classes
        unique_labels = np.unique(y_all)
        if len(unique_labels) < 2:
            print(f"\n ERREUR : Seulement {len(unique_labels)} classe(s) trouvée(s). Besoin d'au moins 2 classes pour l'entraînement.")
            return None, None, None, None

        print(f"\n {len(unique_labels)} classes détectées : {unique_labels}")
        
        # Convertir en arrays numpy
        X = np.array(X_all, dtype=np.float32)
        y = np.array(y_all, dtype=np.int64)
        
        print(f"Shape finale X : {X.shape}, Shape finale y : {y.shape}")
        
        # Appliquer la normalisation
        if self.norm_type != 'none':
            X = self._create_scaler(X)
            self._save_scaler()
        
        return X, y, file_stats, class_counts

def build_cnn_model(input_shape, num_classes):
    """Construit le modèle CNN avec TensorFlow/Keras"""
    inputs = Input(shape=input_shape)
    
    # Première couche de convolution
    x = Conv1D(32, 5, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.2)(x)

    # Deuxième couche de convolution
    x = Conv1D(64, 5, activation='relu', padding='same')(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.2)(x)

    # Troisième couche de convolution
    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.2)(x)

    # Quatrième couche de convolution
    x = Conv1D(64, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)

    # Couches denses
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)

    # Couche de sortie
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def export_data_to_csv(X, y, class_names, file_stats, prefix="csi_data"):
    """Exporte les données vers des fichiers CSV"""
    print("\n=== EXPORT DES DONNÉES ===")
    
    # Créer un DataFrame avec les données
    if len(X.shape) > 2:
        # Si les données sont 3D, les aplatir pour le CSV
        X_flat = X.reshape(X.shape[0], -1)
    else:
        X_flat = X
    
    df = pd.DataFrame(X_flat)
    df['label'] = y
    df['class_name'] = df['label'].map({i: class_names[i] for i in range(len(class_names))})

    # Sauvegarder les données
    csv_filename = f"{prefix}_multiclass.csv"
    df.to_csv(csv_filename, index=False)
    print(f" Données sauvées : {csv_filename}")

    # Créer un rapport détaillé des fichiers
    if file_stats:
        file_report = []
        for filename, stats in file_stats.items():
            file_report.append({
                'filename': filename,
                'label': stats['label'],
                'class_name': class_names[stats['label']],
                'matched_key': stats['matched_key'],
                'samples': stats['samples'],
                'data_shape': str(stats['shape'])
            })

        df_files = pd.DataFrame(file_report)
        files_report_filename = "files_analysis_report.csv"
        df_files.to_csv(files_report_filename, index=False)
        print(f" Rapport des fichiers sauvé : {files_report_filename}")
        
        return csv_filename, files_report_filename
    
    return csv_filename, None

def evaluate_model_performance(model, X_test, y_test, class_names, training_time, history):
    """Évalue les performances du modèle de manière complète"""
    print(f"\n=== ÉVALUATION COMPLÈTE ===")
    
    # Évaluation de base
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    # Rapport de classification détaillé
    print("\n Rapport de classification :")
    unique_labels = np.unique(y_test)
    target_names = [class_names[i] for i in sorted(unique_labels)]
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n Matrice de confusion :")
    print(cm)

    kappa_score = cohen_kappa_score(y_test, y_pred)

    # Métriques globales
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1_score_weighted = report['weighted avg']['f1-score']

    # === MESURES DE PERFORMANCE ===
    # Temps d'inférence
    start_inf = time.time()
    _ = model.predict(np.expand_dims(X_test[0], axis=0), verbose=0)
    inference_time_ms = (time.time() - start_inf) * 1000

    # Taille du modèle
    model_filename = "cnn_model_multiclass.keras"
    model.save(model_filename)
    model_size_mb = os.path.getsize(model_filename) / (1024 * 1024)

    # Gap d'overfitting
    train_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    overfitting_gap = abs(train_acc - val_acc)

    # === RAPPORT DE PERFORMANCE FINAL ===
    print(f"\n=== PERFORMANCES FINALES ===")
    print(f"Test Accuracy       : {test_accuracy:.4f}")
    print(f"Test Loss           : {test_loss:.4f}")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted)   : {recall:.4f}")
    print(f"F1-score (weighted) : {f1_score_weighted:.4f}")
    print(f"Paramètres du modèle: {model.count_params():,}")
    print(f"Temps d'entraînement: {training_time:.2f}s")
    print(f"Temps d'inférence   : {inference_time_ms:.2f}ms")
    print(f"Taille du modèle    : {model_size_mb:.2f}MB")
    print(f"Gap d'overfitting   : {overfitting_gap:.4f}")
    # print(f"Classes traitées    : {len(unique_labels)} »")
    print(f"\n Cohen's Kappa : {kappa_score:.4f}")

    # Créer le rapport de performance
    performance_data = []
    performance_data.append(['Test_Accuracy', test_accuracy])
    performance_data.append(['Test_Loss', test_loss])
    performance_data.append(['Precision_Weighted', precision])
    performance_data.append(['Recall_Weighted', recall])
    performance_data.append(['F1_Score_Weighted', f1_score_weighted])
    performance_data.append(['Model_Parameters', model.count_params()])
    performance_data.append(['Training_Time_s', training_time])
    performance_data.append(['Inference_Time_ms', inference_time_ms])
    performance_data.append(['Model_Size_MB', model_size_mb])
    performance_data.append(['Overfitting_Gap', overfitting_gap])
    performance_data.append(['Num_Classes', len(unique_labels)])
    performance_data.append(['Cohen_Kappa', kappa_score])
    performance_data.append(['Model_Parameters', model.count_params()])

    # Ajouter les métriques par classe
    for class_id in sorted(unique_labels):
        class_name = class_names[class_id]
        if str(class_id) in report:
            class_metrics = report[str(class_id)]
            performance_data.append([f'Precision_{class_name}', class_metrics['precision']])
            performance_data.append([f'Recall_{class_name}', class_metrics['recall']])
            performance_data.append([f'F1_Score_{class_name}', class_metrics['f1-score']])
            performance_data.append([f'Support_{class_name}', class_metrics['support']])

    return performance_data, model_filename

def main():
    """Fonction principale pour exécuter l'entraînement complet"""
    
    # === CONFIGURATION ===
    # CHANGEZ CES PARAMÈTRES SELON VOS DONNÉES
    data_path = "/datos/TSFR"  # CHANGEZ CETTE LIGNE
    class_names = ['vacio', '1_persona', '2_personas', '3_personas']  # 4 classes
    n_symbols = None  # None pour prendre toutes les données, ou spécifiez un nombre
    norm_type = 'standard'  # 'standard', 'minmax', ou 'none'
    
    # Vérifiez que le chemin existe
    if not os.path.exists(data_path):
        print(f" ERREUR : Le chemin {data_path} n'existe pas.")
        print("Veuillez modifier la variable 'data_path' avec le bon chemin vers vos fichiers .mat")
        return
    
    # === INITIALISATION DU PROCESSEUR DE DONNÉES ===
    processor = CSIDataProcessor(
        data_path=data_path,
        class_names=class_names,
        n_symbols=n_symbols,
        norm_type=norm_type
    )
    
    # === CHARGEMENT ET TRAITEMENT DES DONNÉES ===
    X, y, file_stats, class_counts = processor.load_and_process_data()
    
    if X is None:
        return
    
    # === PRÉPARATION POUR CNN 1D ===
    print(f"\n=== PRÉPARATION POUR CNN ===")
    
    # Si les données sont 1D, les reformater pour Conv1D
    if len(X.shape) == 2:
        X = X.reshape(*X.shape, 1)  # Ajouter une dimension pour les canaux
    
    print(f"Shape finale pour CNN : {X.shape}")
    print(f'Input dimensions pour le modèle: {X.shape[1:]}')

    # === EXPORT DES DONNÉES ===
    csv_filename, files_report_filename = export_data_to_csv(
        X, y, class_names, file_stats, "csi_data_processed"
    )

    # === SPLIT DES DONNÉES ===
    print(f"\n=== DIVISION DES DONNÉES ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Entraînement : {X_train.shape[0]} échantillons")
    print(f"Test : {X_test.shape[0]} échantillons")

    # Vérifier la répartition dans les ensembles
    print("Répartition train/test par classe :")
    train_unique, train_counts = np.unique(y_train, return_counts=True)
    test_unique, test_counts = np.unique(y_test, return_counts=True)

    for i, class_id in enumerate(train_unique):
        train_count = train_counts[i]
        test_count = test_counts[i] if class_id in test_unique else 0
        print(f"  Classe {class_id} ({class_names[class_id]}): {train_count} train, {test_count} test")

    # === CONSTRUCTION ET ENTRAÎNEMENT DU MODÈLE ===
    print(f"\n=== CONSTRUCTION DU MODÈLE CNN ===")
    unique_labels = np.unique(y)
    num_classes = len(unique_labels)
    
    model = build_cnn_model(input_shape=X.shape[1:], num_classes=num_classes)
    model.summary()

    print(f"\n=== ENTRAÎNEMENT ===")
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=64,
        validation_data=(X_test, y_test),
        verbose=1
    )
    training_time = time.time() - start_time

    # === ÉVALUATION DÉTAILLÉE ===
    performance_data, model_filename = evaluate_model_performance(
        model, X_test, y_test, class_names, training_time, history
    )
    
    # === EXPORT DU RAPPORT DE PERFORMANCE ===
    performance_df = pd.DataFrame(performance_data, columns=['Metric', 'Value'])
    performance_filename = "model_performance_multiclass.csv"
    performance_df.to_csv(performance_filename, index=False)

    print(f"\n Fichiers générés :")
    print(f"  1. {csv_filename} - Données CSI traitées")
    if files_report_filename:
        print(f"  2. {files_report_filename} - Rapport d'analyse des fichiers")
    print(f"  3. {performance_filename} - Métriques de performance détaillées")
    print(f"  4. {model_filename} - Modèle CNN entraîné")
    
    print(f"\n Entraînement terminé avec succès !")

if __name__ == "__main__":
    main()