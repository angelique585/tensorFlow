import os
import time
import numpy as np
import pandas as pd
import joblib
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Dropout, Conv1D, Conv2D, 
                                   BatchNormalization, MaxPooling1D, MaxPooling2D,
                                   LSTM, GRU, TimeDistributed, Reshape, 
                                   Concatenate, GlobalAveragePooling1D,
                                   Flatten, Permute)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

class CSIDataProcessor:
    """Classe pour traiter les données CSI avec extraction d'amplitude et phase pour CNN-LSTM"""
    
    def __init__(self, data_path, class_names, n_symbols=None, n_subcarriers=None, 
                 norm_type='standard', script_name='csi_cnn_lstm_model', timestamp=None):
        self.data_path = data_path
        self.class_names = class_names
        self.n_symbols = n_symbols
        self.n_subcarriers = n_subcarriers
        self.norm_type = norm_type
        self.script_name = script_name
        self.timestamp = timestamp or time.strftime("%Y%m%d_%H%M%S")
        self.scaler_amplitude = None
        self.scaler_phase = None
        
        print(f"RUTA UTILIZADA: {os.path.basename(os.path.dirname(os.path.dirname(data_path)))} --> "
              f"{os.path.basename(os.path.dirname(data_path))} --> {os.path.basename(data_path)}")
    
    def _extract_amplitude_phase(self, csi_data):
        """Extrait l'amplitude et la phase des données CSI complexes"""
        if np.iscomplexobj(csi_data):
            amplitude = np.abs(csi_data)
            phase = np.angle(csi_data)
        else:
            amplitude = np.abs(csi_data)
            phase = np.zeros_like(amplitude)
            print("    Attention: Données non complexes détectées, phase mise à zéro")
        
        return amplitude, phase
    
    def _create_scaler(self, amplitude_data, phase_data):
        """Crée et ajuste les scalers sur les données d'amplitude et de phase"""
        if self.norm_type == 'standard':
            self.scaler_amplitude = StandardScaler()
            self.scaler_phase = StandardScaler()
        elif self.norm_type == 'minmax':
            self.scaler_amplitude = MinMaxScaler()
            self.scaler_phase = MinMaxScaler()
        else:
            self.scaler_amplitude = None
            self.scaler_phase = None
            return amplitude_data, phase_data
        
        # Reshape pour les scalers
        original_shape = amplitude_data.shape
        if len(original_shape) > 2:
            amplitude_reshaped = amplitude_data.reshape(original_shape[0], -1)
            phase_reshaped = phase_data.reshape(original_shape[0], -1)
        else:
            amplitude_reshaped = amplitude_data
            phase_reshaped = phase_data
            
        amplitude_scaled = self.scaler_amplitude.fit_transform(amplitude_reshaped)
        phase_scaled = self.scaler_phase.fit_transform(phase_reshaped)
        
        # Revenir à la forme originale
        if len(original_shape) > 2:
            amplitude_scaled = amplitude_scaled.reshape(original_shape)
            phase_scaled = phase_scaled.reshape(original_shape)
        
        print(f"Transformación de data: {self.norm_type}")
        return amplitude_scaled, phase_scaled
    
    def _save_scalers(self):
        """Sauvegarde les scalers d'amplitude et de phase"""
        if self.scaler_amplitude is not None:
            amplitude_scaler_filename = f"scaler_amplitude_{self.script_name}_{self.norm_type}_{self.timestamp}.pkl"
            joblib.dump(self.scaler_amplitude, amplitude_scaler_filename)
            print(f"Scaler amplitude sauvé : {amplitude_scaler_filename}")
            
        if self.scaler_phase is not None:
            phase_scaler_filename = f"scaler_phase_{self.script_name}_{self.norm_type}_{self.timestamp}.pkl"
            joblib.dump(self.scaler_phase, phase_scaler_filename)
            print(f"Scaler phase sauvé : {phase_scaler_filename}")
    
    def load_and_process_data(self):
        """Charge et traite tous les fichiers .mat pour CNN-LSTM avec amplitude et phase"""
        print(f"\n=== CHARGEMENT DE TOUS LES FICHIERS .MAT POUR CNN-LSTM ===")
        
        X_amplitude_all, X_phase_all, y_all = [], [], []
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
                data_keys = [k for k in mat.keys() if not k.startswith('__')]
                if len(data_keys) < 1:
                    print(f"     Aucune donnée trouvée - fichier ignoré")
                    continue
                    
                csi_data = mat[data_keys[-1]]
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
                
                # Extraire amplitude et phase
                amplitude, phase = self._extract_amplitude_phase(csi_data)
                
                # Stocker le nombre de subcarriers
                if subcarriers is None:
                    subcarriers = csi_data.shape[1]
                    self.n_subcarriers = subcarriers
                    print(f"    Subcarriers détectés: {subcarriers}")
                
                # Pour CNN-LSTM, nous créons des séquences temporelles
                if self.n_symbols is None:
                    # Définir une longueur de séquence temporelle
                    self.n_symbols = min(128, csi_data.shape[0])  # Longueur de séquence
                    print(f"    Longueur de séquence définie automatiquement: {self.n_symbols}")
                
                # Créer des séquences temporelles pour CNN-LSTM
                n_samples = amplitude.shape[0] // self.n_symbols
                if n_samples == 0:
                    print(f"     Pas assez de données pour créer des séquences - fichier ignoré")
                    continue
                
                amplitude = amplitude[:n_samples * self.n_symbols]
                phase = phase[:n_samples * self.n_symbols]
                
                for i in range(n_samples):
                    start_idx = i * self.n_symbols
                    end_idx = start_idx + self.n_symbols
                    
                    # Créer des séquences temporelles pour CNN-LSTM
                    amp_seq = amplitude[start_idx:end_idx]  # Shape: (n_symbols, n_subcarriers)
                    phase_seq = phase[start_idx:end_idx]   # Shape: (n_symbols, n_subcarriers)
                    
                    X_amplitude_all.append(amp_seq)
                    X_phase_all.append(phase_seq)
                    y_all.append(label)
                
                num_samples = n_samples
                
                # Statistiques
                class_counts[label] += num_samples
                file_stats[filename] = {
                    'label': label,
                    'matched_key': matched_key,
                    'samples': num_samples,
                    'shape': csi_data.shape,
                    'sequence_size': (self.n_symbols, subcarriers)
                }
                
                print(f"    Label: {label} ('{matched_key}') - {num_samples} séquences de taille {self.n_symbols}x{subcarriers}")
                
            except Exception as e:
                print(f"    Erreur lors du chargement : {e}")
                continue

        # === VÉRIFICATION DES DONNÉES CHARGÉES ===
        print(f"\n=== RÉSUMÉ DU CHARGEMENT ===")
        print(f"Total de séquences chargées : {len(X_amplitude_all)}")
        print(f"Répartition par classe :")
        
        for class_id, count in class_counts.items():
            if count > 0:
                percentage = (count / len(X_amplitude_all)) * 100
                print(f"  Classe {class_id} ({self.class_names[class_id]}): {count} séquences ({percentage:.1f}%)")

        # Vérifier qu'on a au moins 2 classes
        unique_labels = np.unique(y_all)
        if len(unique_labels) < 2:
            print(f"\n ERREUR : Seulement {len(unique_labels)} classe(s) trouvée(s). Besoin d'au moins 2 classes pour l'entraînement.")
            return None, None, None, None, None

        print(f"\n {len(unique_labels)} classes détectées : {unique_labels}")
        
        # Convertir en arrays numpy
        X_amplitude = np.array(X_amplitude_all, dtype=np.float32)
        X_phase = np.array(X_phase_all, dtype=np.float32)
        y = np.array(y_all, dtype=np.int64)
        
        print(f"Shape amplitude : {X_amplitude.shape}, Shape phase : {X_phase.shape}, Shape y : {y.shape}")
        print(f"Format CNN-LSTM: (batch_size, sequence_length, features)")
        
        # Appliquer la normalisation
        if self.norm_type != 'none':
            X_amplitude, X_phase = self._create_scaler(X_amplitude, X_phase)
            self._save_scalers()
        
        return X_amplitude, X_phase, y, file_stats, class_counts

def build_cnn_lstm_hybrid(sequence_length, n_subcarriers, num_classes):
    """
    Construit un modèle CNN-LSTM hybride pour traiter amplitude et phase séparément
    
    Architecture:
    1. Branches séparées pour amplitude et phase avec CNN 1D
    2. Fusion des features extraites
    3. LSTM pour capturer les dépendances temporelles
    4. Classification finale
    """
    
    # Entrées séparées pour amplitude et phase
    amplitude_input = Input(shape=(sequence_length, n_subcarriers), name='amplitude_input')
    phase_input = Input(shape=(sequence_length, n_subcarriers), name='phase_input')
    
    # ============= BRANCHE AMPLITUDE =============
    # CNN 1D pour extraire les features spatiales de l'amplitude
    amp_conv1 = Conv1D(64, kernel_size=3, activation='relu', padding='same', name='amp_conv1')(amplitude_input)
    amp_bn1 = BatchNormalization(name='amp_bn1')(amp_conv1)
    amp_pool1 = MaxPooling1D(pool_size=2, name='amp_pool1')(amp_bn1)
    amp_dropout1 = Dropout(0.3, name='amp_dropout1')(amp_pool1)
    
    amp_conv2 = Conv1D(128, kernel_size=3, activation='relu', padding='same', name='amp_conv2')(amp_dropout1)
    amp_bn2 = BatchNormalization(name='amp_bn2')(amp_conv2)
    amp_pool2 = MaxPooling1D(pool_size=2, name='amp_pool2')(amp_bn2)
    amp_dropout2 = Dropout(0.3, name='amp_dropout2')(amp_pool2)
    
    amp_conv3 = Conv1D(256, kernel_size=3, activation='relu', padding='same', name='amp_conv3')(amp_dropout2)
    amp_bn3 = BatchNormalization(name='amp_bn3')(amp_conv3)
    
    # ============= BRANCHE PHASE =============
    # CNN 1D pour extraire les features spatiales de la phase
    phase_conv1 = Conv1D(64, kernel_size=3, activation='relu', padding='same', name='phase_conv1')(phase_input)
    phase_bn1 = BatchNormalization(name='phase_bn1')(phase_conv1)
    phase_pool1 = MaxPooling1D(pool_size=2, name='phase_pool1')(phase_bn1)
    phase_dropout1 = Dropout(0.3, name='phase_dropout1')(phase_pool1)
    
    phase_conv2 = Conv1D(128, kernel_size=3, activation='relu', padding='same', name='phase_conv2')(phase_dropout1)
    phase_bn2 = BatchNormalization(name='phase_bn2')(phase_conv2)
    phase_pool2 = MaxPooling1D(pool_size=2, name='phase_pool2')(phase_bn2)
    phase_dropout2 = Dropout(0.3, name='phase_dropout2')(phase_pool2)
    
    phase_conv3 = Conv1D(256, kernel_size=3, activation='relu', padding='same', name='phase_conv3')(phase_dropout2)
    phase_bn3 = BatchNormalization(name='phase_bn3')(phase_conv3)
    
    # ============= FUSION DES BRANCHES =============
    # Concaténer les features d'amplitude et de phase
    fused_features = Concatenate(axis=-1, name='feature_fusion')([amp_bn3, phase_bn3])
    fused_dropout = Dropout(0.4, name='fused_dropout')(fused_features)
    
    # ============= PARTIE LSTM =============
    # LSTM bidirectionnel pour capturer les dépendances temporelles
    from tensorflow.keras.layers import Bidirectional
    lstm_out = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3), 
                            name='bidirectional_lstm')(fused_dropout)
    
    # LSTM final pour la classification
    lstm_final = LSTM(64, dropout=0.3, recurrent_dropout=0.3, name='final_lstm')(lstm_out)
    
    # ============= CLASSIFICATION =============
    dense1 = Dense(128, activation='relu', name='dense1')(lstm_final)
    dense1_dropout = Dropout(0.5, name='dense1_dropout')(dense1)
    
    dense2 = Dense(64, activation='relu', name='dense2')(dense1_dropout)
    dense2_dropout = Dropout(0.5, name='dense2_dropout')(dense2)
    
    outputs = Dense(num_classes, activation='softmax', name='predictions')(dense2_dropout)
    
    # ============= MODÈLE FINAL =============
    model = Model(inputs=[amplitude_input, phase_input], outputs=outputs, 
                 name='cnn_lstm_hybrid_csi')
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def export_data_to_csv(X_amplitude, X_phase, y, class_names, file_stats, prefix="csi_data_cnn_lstm"):
    """Exporte les données vers des fichiers CSV"""
    print("\n=== EXPORT DES DONNÉES CNN-LSTM ===")
    
    # Aplatir les données pour le CSV
    X_amp_flat = X_amplitude.reshape(X_amplitude.shape[0], -1)
    X_phase_flat = X_phase.reshape(X_phase.shape[0], -1)
    
    # Créer le DataFrame avec amplitude et phase
    df_amplitude = pd.DataFrame(X_amp_flat)
    df_amplitude.columns = [f'amp_t{i//X_amplitude.shape[2]}_sc{i%X_amplitude.shape[2]}' 
                           for i in range(X_amp_flat.shape[1])]
    
    df_phase = pd.DataFrame(X_phase_flat)
    df_phase.columns = [f'phase_t{i//X_phase.shape[2]}_sc{i%X_phase.shape[2]}' 
                       for i in range(X_phase_flat.shape[1])]
    
    # Combiner amplitude et phase
    df = pd.concat([df_amplitude, df_phase], axis=1)
    df['label'] = y
    df['class_name'] = df['label'].map({i: class_names[i] for i in range(len(class_names))})

    # Sauvegarder les données
    csv_filename = f"{prefix}_multiclass_dual_channel.csv"
    df.to_csv(csv_filename, index=False)
    print(f" Données CNN-LSTM sauvées : {csv_filename}")

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
                'data_shape': str(stats['shape']),
                'sequence_size': str(stats.get('sequence_size', 'N/A'))
            })

        df_files = pd.DataFrame(file_report)
        files_report_filename = "files_analysis_report_cnn_lstm.csv"
        df_files.to_csv(files_report_filename, index=False)
        print(f" Rapport des fichiers CNN-LSTM sauvé : {files_report_filename}")
        
        return csv_filename, files_report_filename
    
    return csv_filename, None

def evaluate_model_performance(model, X_amp_test, X_phase_test, y_test, class_names, training_time, history):
    """Évalue les performances du modèle CNN-LSTM de manière complète"""
    print(f"\n=== ÉVALUATION COMPLÈTE CNN-LSTM ===")
    
    # Évaluation de base
    test_loss, test_accuracy = model.evaluate([X_amp_test, X_phase_test], y_test, verbose=0)
    y_pred = np.argmax(model.predict([X_amp_test, X_phase_test], verbose=0), axis=1)

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
    _ = model.predict([np.expand_dims(X_amp_test[0], axis=0), 
                      np.expand_dims(X_phase_test[0], axis=0)], verbose=0)
    inference_time_ms = (time.time() - start_inf) * 1000

    # Taille du modèle
    model_filename = "cnn_lstm_hybrid_multiclass.keras"
    model.save(model_filename)
    model_size_mb = os.path.getsize(model_filename) / (1024 * 1024)

    # Gap d'overfitting
    train_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    overfitting_gap = abs(train_acc - val_acc)

    # === RAPPORT DE PERFORMANCE FINAL ===
    print(f"\n===  PERFORMANCES FINALES CNN-LSTM HYBRIDE ===")
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
    """Fonction principale pour exécuter l'entraînement CNN-LSTM hybride"""
    
    # === CONFIGURATION ===
    data_path = "/datos/TSFR"  # Chemin vers le dossier datos
    class_names = ['vacio', '1_persona', '2_personas', '3_personas']  # 4 classes
    n_symbols = 128  # Longueur de séquence pour CNN-LSTM (ajustez selon vos données)
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
        norm_type=norm_type,
        script_name='cnn_lstm_hybrid'
    )
    
    # === CHARGEMENT ET TRAITEMENT DES DONNÉES ===
    X_amplitude, X_phase, y, file_stats, class_counts = processor.load_and_process_data()
    
    if X_amplitude is None:
        return
    
    # === VÉRIFICATION DU FORMAT CNN-LSTM ===
    print(f"\n=== PRÉPARATION POUR CNN-LSTM HYBRIDE ===")
    print(f"Shape amplitude : {X_amplitude.shape}")
    print(f"Shape phase : {X_phase.shape}")
    print(f"Format attendu: (batch_size, sequence_length, n_subcarriers)")
    print(f"Longueur de séquence: {X_amplitude.shape[1]}")
    print(f"Nombre de subcarriers: {X_amplitude.shape[2]}")

    # === EXPORT DES DONNÉES ===
    csv_filename, files_report_filename = export_data_to_csv(
        X_amplitude, X_phase, y, class_names, file_stats, "csi_data_cnn_lstm_hybrid"
    )

    # === SPLIT DES DONNÉES ===
    print(f"\n=== DIVISION DES DONNÉES ===")
    X_amp_train, X_amp_test, X_phase_train, X_phase_test, y_train, y_test = train_test_split(
        X_amplitude, X_phase, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Entraînement : {X_amp_train.shape[0]} séquences")
    print(f"Test : {X_amp_test.shape[0]} séquences")

    # Vérifier la répartition dans les ensembles
    print("Répartition train/test par classe :")
    train_unique, train_counts = np.unique(y_train, return_counts=True)
    test_unique, test_counts = np.unique(y_test, return_counts=True)

    for i, class_id in enumerate(train_unique):
        train_count = train_counts[i]
        test_count = test_counts[i] if class_id in test_unique else 0
        print(f"  Classe {class_id} ({class_names[class_id]}): {train_count} train, {test_count} test")

    # === CONSTRUCTION ET ENTRAÎNEMENT DU MODÈLE CNN-LSTM ===
    print(f"\n=== CONSTRUCTION DU MODÈLE CNN-LSTM HYBRIDE ===")
    unique_labels = np.unique(y)
    num_classes = len(unique_labels)
    
    model = build_cnn_lstm_hybrid(
        sequence_length=X_amplitude.shape[1], 
        n_subcarriers=X_amplitude.shape[2], 
        num_classes=num_classes
    )
    model.summary()

    # Callbacks pour améliorer l'entraînement
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]

    print(f"\n=== ENTRAÎNEMENT CNN-LSTM HYBRIDE ===")
    start_time = time.time()
    history = model.fit(
        [X_amp_train, X_phase_train], y_train,
        epochs=10,  
        batch_size=64,  # Batch size pour CNN-LSTM
        validation_data=([X_amp_test, X_phase_test], y_test),
        callbacks=callbacks,
        verbose=1
    )
    training_time = time.time() - start_time

    # === ÉVALUATION DU MODÈLE ===
    performance_data, model_filename = evaluate_model_performance(
        model, X_amp_test, X_phase_test, y_test, class_names, training_time, history
    )

    # === SAUVEGARDE DU RAPPORT DE PERFORMANCE ===
    performance_df = pd.DataFrame(performance_data, columns=['Metric', 'Value'])
    performance_filename = f"cnn_lstm_hybrid_performance_report_{processor.timestamp}.csv"
    performance_df.to_csv(performance_filename, index=False)
    print(f"\n Rapport de performance sauvé : {performance_filename}")

    print(f"\n ENTRAÎNEMENT CNN-LSTM HYBRIDE TERMINÉ AVEC SUCCÈS !")
    print(f" Fichiers générés :")
    print(f"  - Modèle : {model_filename}")
    print(f"  - Données : {csv_filename}")
    if files_report_filename:
        print(f"  - Rapport fichiers : {files_report_filename}")
    print(f"  - Performance : {performance_filename}")

if __name__ == "__main__":
    main()