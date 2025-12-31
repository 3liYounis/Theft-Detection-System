import io
import os
import time
import glob
import sqlite3
import joblib
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

from feature_extraction import FeatureExtractor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

from tensorflow import keras
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


class LSTMTheftDetector:
    def __init__(self, sequence_length=30, feature_dim=None, save_dir="./training_results"):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.model = None
        self.scaler = None
        self.history = None
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def load_sequences(self, db_path):
        import time
        all_sequences = []
        all_labels = []

        if not os.path.exists(db_path):
            raise ValueError(f"Database not found at {db_path}")

        extractor = FeatureExtractor()
        feature_names = extractor.get_feature_names()
        feature_columns = ", ".join(feature_names)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        print("Fetching sequence metadata from database...")
        cursor.execute(
            "SELECT video_name, start_frame, length, label FROM sequences")
        sequences_meta = cursor.fetchall()

        if not sequences_meta:
            print(f"No sequences found in database {db_path}")
            conn.close()
            return np.empty((0, self.sequence_length, self.feature_dim)), np.array([])

        num_sequences = len(sequences_meta)
        print(f"Found {num_sequences} sequences. Loading features...")

        start_time = time.time()

        sequences_by_video = {}
        for meta in sequences_meta:
            vname, start, length, label = meta
            if vname not in sequences_by_video:
                sequences_by_video[vname] = []
            sequences_by_video[vname].append((start, length, label))

        print(f"Processing {len(sequences_by_video)} videos...")

        for i, (vname, seqs) in enumerate(sequences_by_video.items()):
            max_frame = max([s + l for s, l, _ in seqs])

            query = f"""
                SELECT frame_index, {feature_columns}
                FROM features
                WHERE video_name = ? AND frame_index <= ?
                ORDER BY frame_index
            """
            cursor.execute(query, (vname, max_frame))
            rows = cursor.fetchall()

            video_features_map = {row[0]: np.array(row[1:]) for row in rows}

            for start, length, label in seqs:
                if length != self.sequence_length:
                    continue

                seq_data = []
                valid_seq = True
                for j in range(length):
                    frame_idx = start + j
                    if frame_idx in video_features_map:
                        seq_data.append(video_features_map[frame_idx])
                    else:
                        valid_seq = False
                        break

                if valid_seq:
                    all_sequences.append(np.array(seq_data))
                    all_labels.append(label)

            if (i + 1) % 10 == 0:
                print(
                    f"Processed {i + 1}/{len(sequences_by_video)} videos ({time.time() - start_time:.1f}s)")

        conn.close()

        X = np.array(all_sequences)
        y = np.array(all_labels)

        print(f"Total loading time: {time.time() - start_time:.1f}s")
        print(f"Loaded {len(X)} valid sequences.")

        if self.feature_dim is None and len(X) > 0:
            self.feature_dim = X.shape[2]

        return X, y

    def preprocess_data(self, X_train, X_test):
        n_samples_train, seq_len, n_features = X_train.shape
        n_samples_test = X_test.shape[0]

        X_train_2d = X_train.reshape(-1, n_features)
        X_test_2d = X_test.reshape(-1, n_features)

        self.scaler = StandardScaler()
        X_train_scaled_2d = self.scaler.fit_transform(X_train_2d)
        X_test_scaled_2d = self.scaler.transform(X_test_2d)

        X_train_scaled = X_train_scaled_2d.reshape(
            n_samples_train, seq_len, n_features)
        X_test_scaled = X_test_scaled_2d.reshape(
            n_samples_test, seq_len, n_features)

        return X_train_scaled, X_test_scaled

    def build_model(self, learning_rate=0.001):
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(self.sequence_length, self.feature_dim),
                 kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.3),

            LSTM(64, return_sequences=True,
                 kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.3),

            LSTM(32, return_sequences=False,
                 kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.3),

            Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.2),

            Dense(1, activation='sigmoid')
        ])

        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy',
                     tf.keras.metrics.Precision(name='precision'),
                     tf.keras.metrics.Recall(name='recall'),
                     tf.keras.metrics.AUC(name='auc')]
        )
        self.model = model
        model.summary()
        return model

    def train(self, X_train, y_train, X_val, y_val, X_test=None, y_test=None, epochs=50, batch_size=32):
        # early_stop = EarlyStopping(
        #     monitor='val_loss',
        #     patience=10,
        #     restore_best_weights=True,
        #     verbose=1
        # )

        checkpoint = ModelCheckpoint(
            './model/best_theft_detector_lstm.keras',
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )

        class_weights = self._compute_class_weights(y_train)
        callbacks_list = [checkpoint, reduce_lr]

        def on_epoch_end(epoch, logs):
            results = self.model.evaluate(X_test, y_test, verbose=0)
            metric_names = ['loss', 'accuracy', 'precision', 'recall', 'auc']
            for name, value in zip(metric_names, results):
                logs[f'test_{name}'] = value

        callbacks_list.append(
            keras.callbacks.LambdaCallback(on_epoch_end=on_epoch_end))

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            class_weight=class_weights,
            verbose=1
        )

        return self.history

    def _compute_class_weights(self, y):
        from sklearn.utils.class_weight import compute_class_weight

        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weight_dict = dict(zip(classes, weights))

        print(f"Class weights: {class_weight_dict}")

        return class_weight_dict

    def evaluate(self, X_test, y_test):
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()

        print(classification_report(y_test, y_pred,
                                    target_names=['Normal', 'Shoplifting'],
                                    digits=4))

        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        print(f"True Negatives: {cm[0,0]}")
        print(f"False Positives: {cm[0,1]} (Normal classified as Shoplifting)")
        print(f"False Negatives: {cm[1,0]} (Shoplifting classified as Normal)")
        print(f"True Positives: {cm[1,1]}")

        auc = roc_auc_score(y_test, y_pred_proba)
        print(f"\nROC-AUC Score: {auc:.4f}")

        self._plot_training_history()
        self._plot_confusion_matrix(cm)
        self._plot_roc_curve(y_test, y_pred_proba)

        return {
            'accuracy': np.mean(y_pred == y_test),
            'precision': cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0,
            'recall': cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0,
            'auc': auc,
            'confusion_matrix': cm
        }

    def _plot_training_history(self):
        if self.history is None:
            return

        metrics = {
            "accuracy": {
                "train": "accuracy",
                "val": "val_accuracy",
                "test": "test_accuracy",
                "title": "Model Accuracy",
                "ylabel": "Accuracy",
                "filename": "accuracy.png"
            },
            "loss": {
                "train": "loss",
                "val": "val_loss",
                "title": "Model Loss",
                "ylabel": "Loss",
                "filename": "loss.png"
            },
            "precision": {
                "train": "precision",
                "val": "val_precision",
                "test": "test_precision",
                "title": "Model Precision",
                "ylabel": "Precision",
                "filename": "precision.png"
            },
            "recall": {
                "train": "recall",
                "val": "val_recall",
                "test": "test_recall",
                "title": "Model Recall",
                "ylabel": "Recall",
                "filename": "recall.png"
            },
            "auc": {
                "train": "auc",
                "val": "val_auc",
                "test": "test_auc",
                "title": "Model AUC",
                "ylabel": "AUC",
                "filename": "auc.png"
            }
        }

        for key, m in metrics.items():
            plt.figure(figsize=(10, 6))
            plt.plot(self.history.history[m["train"]],
                     label='Train', color='blue')
            plt.plot(self.history.history[m["val"]],
                     label='Validation', color='orange')

            test_key = f"test_{key}"
            if test_key in self.history.history:
                plt.plot(
                    self.history.history[test_key], label='Test', color='green', linestyle='--')

            plt.title(m["title"])
            plt.xlabel('Epoch')
            plt.ylabel(m["ylabel"])
            plt.legend()
            plt.grid(True)

            save_path = os.path.join(self.save_dir, m["filename"])
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

    def _plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                    xticklabels=['Normal', 'Shoplifting'],
                    yticklabels=['Normal', 'Shoplifting'])
        plt.title('Test Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(self.save_dir, 'confusion_matrix.png'),
                    dpi=300, bbox_inches='tight')
        print("Confusion matrix plot saved to confusion_matrix.png")
        plt.close()

    def _plot_roc_curve(self, y_test, y_pred_proba):
        """Plot ROC curve"""
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Test ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, 'roc_curve.png'),
                    dpi=300, bbox_inches='tight')
        print("ROC curve plot saved to roc_curve.png")
        plt.close()

    def save(self, model_path='./model/theft_detector_lstm.keras', scaler_path='./scaler/scaler_lstm.pkl'):
        self.model.save(model_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"\nModel saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")

    def load(self, model_path='./model/theft_detector_lstm.keras', scaler_path='./scaler/scaler_lstm.pkl'):
        self.model = keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        print(f"Model loaded from {model_path}")
        print(f"Scaler loaded from {scaler_path}")


def main():
    SEQUENCE_DB_PATH = "../../Data/sequences.db"
    SEQUENCE_LENGTH = 90
    EPOCHS = 80
    BATCH_SIZE = 10
    LEARNING_RATE = 0.0001

    detector = LSTMTheftDetector(sequence_length=SEQUENCE_LENGTH)

    try:
        X, y = detector.load_sequences(SEQUENCE_DB_PATH)
    except ValueError as e:
        print(f"\nERROR: {e}")
        print("\nPlease run data_collection_enhanced.py first to generate sequences!")
        print("Example: python data_collection_enhanced.py")
        return

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"\nData split:")
    print(f"Training: {len(X_train)} sequences")
    print(f"Validation: {len(X_val)} sequences")
    print(f"Test: {len(X_test)} sequences")

    X_train_scaled, X_val_scaled = detector.preprocess_data(X_train, X_val)
    X_test_scaled = detector.scaler.transform(
        X_test.reshape(-1, X_test.shape[2])).reshape(X_test.shape)

    detector.build_model(learning_rate=LEARNING_RATE)

    detector.train(X_train_scaled, y_train, X_val_scaled, y_val,
                   X_test=X_test_scaled, y_test=y_test,
                   epochs=EPOCHS, batch_size=BATCH_SIZE)
    results = detector.evaluate(X_test_scaled, y_test)

    detector.save()

    print("\n" + "=" * 50)
    print("Training complete!")
    print("=" * 50)
    print(f"Final Test Accuracy: {results['accuracy']:.4f}")
    print(f"Final Test Precision: {results['precision']:.4f}")
    print(f"Final Test Recall: {results['recall']:.4f}")
    print(f"Final Test AUC: {results['auc']:.4f}")


if __name__ == "__main__":
    main()
