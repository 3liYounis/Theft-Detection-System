import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, precision_recall_curve,
                             average_precision_score)
import os
import glob

try:
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    print("TensorFlow not available")
    TF_AVAILABLE = False


class ModelEvaluator:
    """
    Comprehensive evaluation suite for theft detection models.
    Supports both LSTM and traditional ML models.
    """

    def __init__(self, model_path, scaler_path=None, model_type='lstm'):
        """
        Args:
            model_path: path to model file
            scaler_path: path to scaler (for LSTM)
            model_type: 'lstm' or 'random_forest'
        """
        self.model_type = model_type
        self.scaler = None

        if model_type == 'lstm':
            if not TF_AVAILABLE:
                raise ImportError("TensorFlow required for LSTM evaluation")
            self.model = keras.models.load_model(model_path)
            if scaler_path:
                self.scaler = joblib.load(scaler_path)
        else:
            self.model = joblib.load(model_path)

        print(f"Loaded {model_type} model from {model_path}")

    def load_test_data(self, data_path, is_sequence=False):
        """
        Load test data.

        Args:
            data_path: path to test data (csv for RF, directory for LSTM)
            is_sequence: whether data is sequences (for LSTM)
        """
        if is_sequence:
            X_list, y_list = [], []
            npz_files = glob.glob(os.path.join(data_path, "*.npz"))

            for npz_file in npz_files:
                data = np.load(npz_file)
                X_list.append(data['sequences'])
                y_list.append(data['labels'])

            X = np.vstack(X_list)
            y = np.concatenate(y_list)
        else:
            import pandas as pd
            data = pd.read_csv(data_path, header=None)
            X = data.iloc[:, :-1].values
            y = data.iloc[:, -1].values
        print(f"Loaded test data: {X.shape}")
        print(
            f"Class distribution: Normal={np.sum(y==0)}, Shoplifting={np.sum(y==1)}")

        return X, y

    def preprocess_data(self, X):
        """Preprocess data using scaler"""
        if self.scaler is None:
            return X

        if len(X.shape) == 3:
            n_samples, seq_len, n_features = X.shape
            X_2d = X.reshape(-1, n_features)
            X_scaled_2d = self.scaler.transform(X_2d)
            return X_scaled_2d.reshape(n_samples, seq_len, n_features)
        else:
            return self.scaler.transform(X)

    def predict(self, X):
        """Make predictions"""
        if self.model_type == 'lstm':
            y_pred_proba = self.model.predict(X, verbose=0)
            if len(y_pred_proba.shape) > 1:
                y_pred_proba = y_pred_proba.flatten()
        else:
            y_pred_proba = self.model.predict_proba(X)[:, 1]

        return y_pred_proba

    def evaluate_comprehensive(self, X, y, save_dir='evaluation_results'):
        """
        Run comprehensive evaluation.
        """
        os.makedirs(save_dir, exist_ok=True)

        print("\n" + "=" * 60)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("=" * 60)

        X_processed = self.preprocess_data(X)

        y_pred_proba = self.predict(X_processed)
        y_pred = (y_pred_proba > 0.5).astype(int)

        # 1. Classification Report
        print("\n1. CLASSIFICATION REPORT")
        print("-" * 60)
        report = classification_report(y, y_pred,
                                       target_names=['Normal', 'Shoplifting'],
                                       digits=4, output_dict=True)
        print(classification_report(y, y_pred, target_names=[
              'Normal', 'Shoplifting'], digits=4))

        # 2. Confusion Matrix
        print("\n2. CONFUSION MATRIX")
        print("-" * 60)
        cm = confusion_matrix(y, y_pred)
        print(cm)
        print(f"\nTrue Negatives (Correct Normal): {cm[0,0]}")
        print(f"False Positives (Normal → Shoplifting): {cm[0,1]}")
        print(f"False Negatives (Shoplifting → Normal): {cm[1,0]}")
        print(f"True Positives (Correct Shoplifting): {cm[1,1]}")

        # False Positive Rate and False Negative Rate
        fpr_value = cm[0, 1] / (cm[0, 0] + cm[0, 1]
                                ) if (cm[0, 0] + cm[0, 1]) > 0 else 0
        fnr_value = cm[1, 0] / (cm[1, 0] + cm[1, 1]
                                ) if (cm[1, 0] + cm[1, 1]) > 0 else 0
        print(f"\nFalse Positive Rate: {fpr_value:.4f} ({fpr_value*100:.2f}%)")
        print(f"False Negative Rate: {fnr_value:.4f} ({fnr_value*100:.2f}%)")

        # 3. ROC-AUC
        print("\n3. ROC-AUC ANALYSIS")
        print("-" * 60)
        auc_score = roc_auc_score(y, y_pred_proba)
        print(f"ROC-AUC Score: {auc_score:.4f}")

        # 4. Precision-Recall
        print("\n4. PRECISION-RECALL ANALYSIS")
        print("-" * 60)
        ap_score = average_precision_score(y, y_pred_proba)
        print(f"Average Precision Score: {ap_score:.4f}")

        # 5. Threshold Analysis
        print("\n5. THRESHOLD ANALYSIS")
        print("-" * 60)
        self._threshold_analysis(y, y_pred_proba)

        # 6. Feature Importance (if available)
        if self.model_type == 'random_forest':
            print("\n6. FEATURE IMPORTANCE")
            print("-" * 60)
            self._feature_importance_analysis(save_dir)

        # Generate plots
        self._plot_confusion_matrix(cm, save_dir)
        self._plot_roc_curve(y, y_pred_proba, save_dir)
        self._plot_precision_recall_curve(y, y_pred_proba, save_dir)
        self._plot_confidence_distribution(y, y_pred_proba, save_dir)
        self._plot_threshold_metrics(y, y_pred_proba, save_dir)

        # Summary statistics
        results = {
            'accuracy': float(report['accuracy']),
            'precision': float(report['Shoplifting']['precision']),
            'recall': float(report['Shoplifting']['recall']),
            'f1_score': float(report['Shoplifting']['f1-score']),
            'roc_auc': float(auc_score),
            'average_precision': float(ap_score),
            'fpr': float(fpr_value),
            'fnr': float(fnr_value),
            'confusion_matrix': cm.tolist()
        }

        import json
        with open(os.path.join(save_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n\nEvaluation complete! Results saved to {save_dir}/")

        return results

    def _threshold_analysis(self, y_true, y_pred_proba):
        """Analyze performance at different thresholds"""
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

        print(
            f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'FPR':<12}")
        print("-" * 60)

        for thresh in thresholds:
            y_pred = (y_pred_proba > thresh).astype(int)
            cm = confusion_matrix(y_true, y_pred)

            if cm.shape == (2, 2):
                precision = cm[1, 1] / (cm[1, 1] + cm[0, 1]
                                        ) if (cm[1, 1] + cm[0, 1]) > 0 else 0
                recall = cm[1, 1] / (cm[1, 1] + cm[1, 0]
                                     ) if (cm[1, 1] + cm[1, 0]) > 0 else 0
                f1 = 2 * precision * recall / \
                    (precision + recall) if (precision + recall) > 0 else 0
                fpr = cm[0, 1] / (cm[0, 0] + cm[0, 1]
                                  ) if (cm[0, 0] + cm[0, 1]) > 0 else 0

                print(
                    f"{thresh:<12.1f} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {fpr:<12.4f}")

    def _feature_importance_analysis(self, save_dir):
        """Analyze feature importance for tree-based models"""
        if not hasattr(self.model, 'feature_importances_'):
            print("Feature importance not available for this model")
            return

        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]

        print("\nTop 20 Most Important Features:")
        for i in range(min(20, len(importances))):
            idx = indices[i]
            print(f"  {i+1}. Feature {idx}: {importances[idx]:.4f}")

        plt.figure(figsize=(12, 6))
        top_n = 30
        plt.bar(range(top_n), importances[indices[:top_n]])
        plt.xlabel('Feature Index')
        plt.ylabel('Importance')
        plt.title(f'Top {top_n} Feature Importances')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'feature_importance.png'), dpi=300)
        plt.close()

    def _plot_confusion_matrix(self, cm, save_dir):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                    xticklabels=['Normal', 'Shoplifting'],
                    yticklabels=['Normal', 'Shoplifting'])
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_dir}/confusion_matrix.png")

    def _plot_roc_curve(self, y_true, y_pred_proba, save_dir):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})', linewidth=2, color='blue')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'roc_curve.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_dir}/roc_curve.png")

    def _plot_precision_recall_curve(self, y_true, y_pred_proba, save_dir):
        """Plot Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        ap = average_precision_score(y_true, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision,
                 label=f'PR Curve (AP = {ap:.4f})', linewidth=2, color='green')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        plt.legend(loc='lower left', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'precision_recall_curve.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_dir}/precision_recall_curve.png")

    def _plot_confidence_distribution(self, y_true, y_pred_proba, save_dir):
        """Plot confidence distribution for both classes"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        normal_confidences = y_pred_proba[y_true == 0]
        axes[0].hist(normal_confidences, bins=50, color='green',
                     alpha=0.7, edgecolor='black')
        axes[0].axvline(0.5, color='red', linestyle='--',
                        linewidth=2, label='Threshold')
        axes[0].set_xlabel('Confidence Score', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Normal Behavior Confidence Distribution',
                          fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        theft_confidences = y_pred_proba[y_true == 1]
        axes[1].hist(theft_confidences, bins=50, color='red',
                     alpha=0.7, edgecolor='black')
        axes[1].axvline(0.5, color='red', linestyle='--',
                        linewidth=2, label='Threshold')
        axes[1].set_xlabel('Confidence Score', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Shoplifting Confidence Distribution',
                          fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(
            save_dir, 'confidence_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_dir}/confidence_distribution.png")

    def _plot_threshold_metrics(self, y_true, y_pred_proba, save_dir):
        """Plot metrics vs threshold"""
        thresholds = np.linspace(0, 1, 100)
        precisions, recalls, f1s, fprs = [], [], [], []

        for thresh in thresholds:
            y_pred = (y_pred_proba > thresh).astype(int)
            cm = confusion_matrix(y_true, y_pred)

            if cm.shape == (2, 2):
                precision = cm[1, 1] / (cm[1, 1] + cm[0, 1]
                                        ) if (cm[1, 1] + cm[0, 1]) > 0 else 0
                recall = cm[1, 1] / (cm[1, 1] + cm[1, 0]
                                     ) if (cm[1, 1] + cm[1, 0]) > 0 else 0
                f1 = 2 * precision * recall / \
                    (precision + recall) if (precision + recall) > 0 else 0
                fpr = cm[0, 1] / (cm[0, 0] + cm[0, 1]
                                  ) if (cm[0, 0] + cm[0, 1]) > 0 else 0
            else:
                precision = recall = f1 = fpr = 0

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            fprs.append(fpr)

        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, precisions, label='Precision', linewidth=2)
        plt.plot(thresholds, recalls, label='Recall', linewidth=2)
        plt.plot(thresholds, f1s, label='F1 Score', linewidth=2)
        plt.plot(thresholds, fprs, label='False Positive Rate',
                 linewidth=2, linestyle='--')
        plt.axvline(0.5, color='black', linestyle=':',
                    linewidth=1, label='Default Threshold')
        plt.xlabel('Threshold', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Metrics vs Threshold', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'threshold_metrics.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_dir}/threshold_metrics.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Evaluate theft detection model')
    parser.add_argument('--model_path', type=str,
                        required=True, help='Path to model file')
    parser.add_argument('--data_path', type=str,
                        required=True, help='Path to test data')
    parser.add_argument('--scaler_path', type=str,
                        default=None, help='Path to scaler (for LSTM)')
    parser.add_argument('--model_type', type=str,
                        default='lstm', choices=['lstm', 'random_forest'])
    parser.add_argument('--save_dir', type=str,
                        default='evaluation_results', help='Output directory')

    args = parser.parse_args()

    evaluator = ModelEvaluator(
        model_path=args.model_path,
        scaler_path=args.scaler_path,
        model_type=args.model_type
    )

    is_sequence = args.model_type == 'lstm'
    X_test, y_test = evaluator.load_test_data(
        args.data_path, is_sequence=is_sequence)

    results = evaluator.evaluate_comprehensive(
        X_test, y_test, save_dir=args.save_dir)

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    for key, value in results.items():
        if key != 'confusion_matrix':
            print(f"{key}: {value}")
