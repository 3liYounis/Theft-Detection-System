from .feature_extraction import EnhancedFeatureExtractor
import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque
import sys
import os
sys.path.append(os.path.dirname(__file__))

try:
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    print("TensorFlow not available. Please install: pip install tensorflow")
    TF_AVAILABLE = False


class LSTMTheftDetectorInference:
    """
    Real-time theft detection using LSTM model.
    Handles sequence buffering and sliding window predictions.
    """

    def __init__(self, model_path='theft_detector_lstm.keras',
                 scaler_path='scaler_lstm.pkl',
                 sequence_length=30):
        """
        Initialize inference engine.

        Args:
            model_path: path to trained LSTM model
            scaler_path: path to fitted scaler
            sequence_length: number of frames in sequence (must match training)
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required for LSTM inference")

        self.sequence_length = sequence_length
        self.model = keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_extractor = EnhancedFeatureExtractor(temporal_window=10)
        self.sequence_buffer = deque(maxlen=sequence_length)

        print(f"Loaded LSTM model from {model_path}")
        print(f"Loaded scaler from {scaler_path}")
        print(f"Sequence length: {sequence_length}")

    def reset(self):
        """Reset internal buffers"""
        self.sequence_buffer.clear()
        self.feature_extractor.reset()

    def process_frame(self, pose_landmarks):
        """
        Process a single frame and update sequence buffer.

        Args:
            pose_landmarks: MediaPipe pose landmarks

        Returns:
            dict with prediction info, or None if not enough frames yet
        """
        if pose_landmarks is None:
            # No pose detected - add zero features
            if len(self.sequence_buffer) > 0:
                zero_features = np.zeros_like(self.sequence_buffer[-1])
                self.sequence_buffer.append(zero_features)
            return None

        try:
            features = self.feature_extractor.extract_features(pose_landmarks)
            self.sequence_buffer.append(features)
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None

        if len(self.sequence_buffer) < self.sequence_length:
            return {
                'ready': False,
                'buffer_size': len(self.sequence_buffer),
                'needed': self.sequence_length
            }

        sequence = np.array(list(self.sequence_buffer))
        sequence = sequence.reshape(1, self.sequence_length, -1)

        n_features = sequence.shape[2]
        sequence_2d = sequence.reshape(-1, n_features)
        sequence_scaled_2d = self.scaler.transform(sequence_2d)
        sequence_scaled = sequence_scaled_2d.reshape(
            1, self.sequence_length, n_features)

        pred_proba = self.model.predict(sequence_scaled, verbose=0)[0][0]
        pred_class = int(pred_proba > 0.5)

        return {
            'ready': True,
            'prediction': pred_class,
            'confidence': float(pred_proba),
            'label': 'Shoplifting' if pred_class == 1 else 'Normal',
            'buffer_size': len(self.sequence_buffer)
        }

    def detect_from_video(self, video_path,
                          threshold=0.5,
                          decision_window=90,
                          decision_ratio=0.6,
                          visualize=True,
                          save_path=None):
        """
        Run theft detection on a video file.

        Args:
            video_path: path to video file or 0 for webcam
            threshold: confidence threshold for classification
            decision_window: number of recent frames to aggregate for final decision
            decision_ratio: fraction of positive predictions needed to declare theft
            visualize: whether to show visualization
            save_path: path to save output video (optional)

        Returns:
            dict with detection results
        """
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            return {
                'theft': False,
                'reason': 'cannot_open_video',
                'video_path': video_path
            }

        mp_pose = mp.solutions.pose
        mp_draw = mp.solutions.drawing_utils
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.reset()

        total_frames = 0
        predictions_ready = 0
        prediction_history = deque(maxlen=decision_window)
        all_confidences = []

        writer = None

        print(f"\nProcessing video: {video_path}")
        print("=" * 50)

        while True:
            ret, frame = video.read()
            if not ret:
                break

            total_frames += 1

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)

            pred_info = self.process_frame(
                results.pose_landmarks if results.pose_landmarks else None)

            if pred_info and pred_info.get('ready'):
                predictions_ready += 1
                prediction_history.append(pred_info['prediction'])
                all_confidences.append(pred_info['confidence'])

                if visualize:
                    if results.pose_landmarks:
                        mp_draw.draw_landmarks(
                            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    if len(prediction_history) >= decision_window * 0.5:
                        window_positive = sum(prediction_history)
                        window_ratio = window_positive / \
                            len(prediction_history)
                        current_decision = "SHOPLIFTING" if window_ratio >= decision_ratio else "NORMAL"
                        decision_color = (
                            0, 0, 255) if current_decision == "SHOPLIFTING" else (0, 200, 0)
                    else:
                        current_decision = "WARMING UP"
                        window_ratio = 0
                        decision_color = (200, 200, 0)
                    conf = pred_info['confidence']
                    frame_label = pred_info['label']
                    frame_color = (
                        0, 0, 255) if frame_label == "Shoplifting" else (0, 255, 0)
                    cv2.putText(frame, f"Frame: {frame_label} ({conf:.2f})",
                                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, frame_color, 2)
                    cv2.putText(frame, f"Decision: {current_decision} (ratio: {window_ratio:.2f})",
                                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, decision_color, 2)
                    cv2.putText(frame, f"Window: {len(prediction_history)}/{decision_window} | Positive: {sum(prediction_history)}",
                                (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            elif visualize:
                if pred_info:
                    buffer_status = f"Buffer: {pred_info.get('buffer_size', 0)}/{self.sequence_length}"
                else:
                    buffer_status = "No pose detected"
                cv2.putText(frame, buffer_status, (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if visualize:
                cv2.imshow('LSTM Theft Detection', frame)
                if writer is None and save_path:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(save_path, fourcc, 30.0, (w, h))

                if writer:
                    writer.write(frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        video.release()
        pose.close()
        if writer:
            writer.release()
        if visualize:
            cv2.destroyAllWindows()
        if len(prediction_history) < decision_window * 0.3:
            theft_decision = False
            reason = "insufficient_data"
            final_ratio = 0
        else:
            window_positive = sum(prediction_history)
            final_ratio = window_positive / len(prediction_history)
            theft_decision = final_ratio >= decision_ratio
            reason = "detection_complete"

        avg_confidence = np.mean(all_confidences) if all_confidences else 0
        max_confidence = np.max(all_confidences) if all_confidences else 0

        results_dict = {
            'theft': theft_decision,
            'reason': reason,
            'video_path': video_path,
            'total_frames': total_frames,
            'predictions_made': predictions_ready,
            'window_size': len(prediction_history),
            'window_positive': int(sum(prediction_history)) if prediction_history else 0,
            'window_ratio': final_ratio,
            'avg_confidence': float(avg_confidence),
            'max_confidence': float(max_confidence),
            'threshold': threshold,
            'decision_ratio': decision_ratio
        }

        print("\n" + "=" * 50)
        print("DETECTION RESULTS")
        print("=" * 50)
        print(f"Theft Detected: {'YES' if theft_decision else 'NO'}")
        print(f"Confidence: {final_ratio:.2%} positive frames")
        print(f"Total Frames: {total_frames}")
        print(f"Predictions Made: {predictions_ready}")
        print(f"Average Confidence: {avg_confidence:.2%}")
        print("=" * 50)

        return results_dict


def detect_theft_lstm(video_path,
                      model_path='theft_detector_lstm.keras',
                      scaler_path='scaler_lstm.pkl',
                      sequence_length=30,
                      threshold=0.5,
                      window_size=90,
                      positive_ratio=0.6,
                      visualize=True,
                      save_path=None):
    """
    Convenience function for theft detection using LSTM.
    Compatible with existing interface.

    Args:
        video_path: path to video file
        model_path: path to LSTM model
        scaler_path: path to scaler
        sequence_length: sequence length (must match training)
        threshold: classification threshold
        window_size: sliding window for aggregation
        positive_ratio: ratio for final decision
        visualize: show visualization
        save_path: save output video

    Returns:
        dict with detection results
    """
    try:
        detector = LSTMTheftDetectorInference(
            model_path=model_path,
            scaler_path=scaler_path,
            sequence_length=sequence_length
        )

        results = detector.detect_from_video(
            video_path=video_path,
            threshold=threshold,
            decision_window=window_size,
            decision_ratio=positive_ratio,
            visualize=visualize,
            save_path=save_path
        )

        return results

    except Exception as e:
        print(f"Error during detection: {e}")
        import traceback
        traceback.print_exc()
        return {
            'theft': False,
            'reason': f'error: {str(e)}',
            'video_path': video_path
        }


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = "./Data/Stream/Shoplifting - Test/Shoplifting (92).mp4"
        # video_path = "./Data/Stream/Normal - Test/Normal (93).mp4"

    print(f"Testing LSTM detector on: {video_path}")

    results = detect_theft_lstm(
        video_path=video_path,
        model_path='theft_detector_lstm.keras',
        scaler_path='scaler_lstm.pkl',
        sequence_length=30,
        window_size=90,
        positive_ratio=0.6,
        visualize=True,
        save_path='output_lstm.mp4'
    )

    print("\nFinal Results:")
    for key, value in results.items():
        print(f"  {key}: {value}")
