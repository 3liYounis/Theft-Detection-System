import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2
    print(f"TensorFlow version: {tf.__version__}")
    TF_AVAILABLE = True
except ImportError:
    print("TensorFlow not available. Please install: pip install tensorflow")
    TF_AVAILABLE = False


class LSTMTheftDetector:
    """
    LSTM-based theft detection model with preprocessing and evaluation utilities.
    """

    def __init__(self, sequence_length=30, feature_dim=None):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.model = None
        self.scaler = None
        self.history = None

    def load_sequences(self, sequence_dir):
        """
        Load sequence data from directory of .npz files.
        """
        print("Loading sequence data...")

        all_sequences = []
        all_labels = []

        npz_files = glob.glob(os.path.join(sequence_dir, "*.npz"))

        if not npz_files:
            raise ValueError(f"No .npz files found in {sequence_dir}")

        for npz_file in npz_files:
            data = np.load(npz_file)
            sequences = data['sequences']
            labels = data['labels']

            all_sequences.append(sequences)
            all_labels.append(labels)

            print(
                f"Loaded {len(sequences)} sequences from {os.path.basename(npz_file)}")

        X = np.vstack(all_sequences)
        y = np.concatenate(all_labels)

        print(f"\nTotal sequences: {len(X)}")
        print(f"Sequence shape: {X.shape}")
        print(
            f"Label distribution: Normal={np.sum(y==0)}, Shoplifting={np.sum(y==1)}")

        if self.feature_dim is None:
            self.feature_dim = X.shape[2]

        return X, y

    def preprocess_data(self, X_train, X_test):
        """
        Normalize features using StandardScaler.
        Fit on training data only to prevent data leakage.
        """
        print("Preprocessing data...")

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

        print(f"Scaled training data: {X_train_scaled.shape}")
        print(f"Scaled test data: {X_test_scaled.shape}")

        return X_train_scaled, X_test_scaled

    def build_model(self, learning_rate=0.001):
        """
        Build LSTM model architecture.
        """
        print("Building LSTM model...")

        model = Sequential([
            # First LSTM layer - Bidirectional for better context
            Bidirectional(LSTM(128, return_sequences=True,
                               kernel_regularizer=l2(0.01)),
                          input_shape=(self.sequence_length, self.feature_dim)),
            BatchNormalization(),
            Dropout(0.3),

            # Second LSTM layer
            Bidirectional(LSTM(64, return_sequences=True,
                               kernel_regularizer=l2(0.01))),
            BatchNormalization(),
            Dropout(0.3),

            # Third LSTM layer (final)
            LSTM(32, return_sequences=False,
                 kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.3),

            # Dense layers
            Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.2),

            # Output layer
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

        print("\nModel Summary:")
        model.summary()

        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """
        Train the LSTM model with callbacks.
        """
        print("\nTraining model...")

        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )

        checkpoint = ModelCheckpoint(
            'best_theft_detector_lstm.keras',
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

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, checkpoint, reduce_lr],
            class_weight=class_weights,
            verbose=1
        )

        return self.history

    def _compute_class_weights(self, y):
        """Compute class weights for imbalanced data"""
        from sklearn.utils.class_weight import compute_class_weight

        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weight_dict = dict(zip(classes, weights))

        print(f"Class weights: {class_weight_dict}")

        return class_weight_dict

    def evaluate(self, X_test, y_test):
        """
        Comprehensive model evaluation.
        """
        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)

        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()

        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred,
                                    target_names=['Normal', 'Shoplifting'],
                                    digits=4))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        print(f"True Negatives: {cm[0,0]}")
        print(f"False Positives: {cm[0,1]} (Normal classified as Shoplifting)")
        print(f"False Negatives: {cm[1,0]} (Shoplifting classified as Normal)")
        print(f"True Positives: {cm[1,1]}")

        # ROC-AUC
        auc = roc_auc_score(y_test, y_pred_proba)
        print(f"\nROC-AUC Score: {auc:.4f}")

        # Create visualizations
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
        """Plot training history"""
        if self.history is None:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train')
        axes[0, 0].plot(self.history.history['val_accuracy'],
                        label='Validation')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Train')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Train')
        axes[1, 0].plot(self.history.history['val_precision'],
                        label='Validation')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Train')
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        print("\nTraining history plot saved to training_history.png")
        plt.close()

    def _plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Normal', 'Shoplifting'],
                    yticklabels=['Normal', 'Shoplifting'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
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
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
        print("ROC curve plot saved to roc_curve.png")
        plt.close()

    def save(self, model_path='theft_detector_lstm.keras', scaler_path='scaler_lstm.pkl'):
        """Save model and scaler"""
        self.model.save(model_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"\nModel saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")

    def load(self, model_path='theft_detector_lstm.keras', scaler_path='scaler_lstm.pkl'):
        """Load model and scaler"""
        self.model = keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        print(f"Model loaded from {model_path}")
        print(f"Scaler loaded from {scaler_path}")


def main():
    """Main training pipeline"""

    if not TF_AVAILABLE:
        print("ERROR: TensorFlow is required. Install with: pip install tensorflow")
        return

    SEQUENCE_DIR = "../../Data/Sequences"
    SEQUENCE_LENGTH = 30
    EPOCHS = 50
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001

    detector = LSTMTheftDetector(sequence_length=SEQUENCE_LENGTH)

    try:
        X, y = detector.load_sequences(SEQUENCE_DIR)
    except ValueError as e:
        print(f"\nERROR: {e}")
        print("\nPlease run data_collection_enhanced.py first to generate sequences!")
        print("Example: python data_collection_enhanced.py")
        return

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
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
