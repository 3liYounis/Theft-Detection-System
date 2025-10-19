import cv2
import numpy as np
import mediapipe as mp
from Behaveioral.inference_lstm import LSTMTheftDetectorInference


class YOLOPoseTheftDetector:
    """
    YOLO-based pose estimation system for theft detection.
    Uses ultralytics YOLO for person detection, MediaPipe for pose estimation,
    and LSTM for behavior analysis to detect shoplifting activities.
    """

    def __init__(self, model_path='Behaveioral/theft_detector_lstm.keras',
                 scaler_path='Behaveioral/scaler_lstm.pkl',
                 sequence_length=30):
        """
        Initialize the YOLO pose detector.
        Args:
            model_path: Path to LSTM model
            scaler_path: Path to scaler
            sequence_length: Sequence length for LSTM
        """
        self.setup_yolo()
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            enable_segmentation=False
        )
        self.lstm_detector = LSTMTheftDetectorInference(
            model_path=model_path,
            scaler_path=scaler_path,
            sequence_length=sequence_length
        )
        print("YOLO Pose Detection system initialized")

    def setup_yolo(self, model_size='nano'):
        """Setup YOLO model for person detection."""
        from ultralytics import YOLO

        model_map = {
            'nano': 'yolov8n.pt',
            'small': 'yolov8s.pt',
            'medium': 'yolov8m.pt',
            'large': 'yolov8l.pt'
        }

        model_name = model_map.get(model_size, 'yolov8n.pt')
        self.yolo_model = YOLO(model_name)
        print(
            f"YOLOv8 {model_size} model loaded successfully for person detection")

    def detect_persons_yolo(self, frame):
        """Detect persons in frame using YOLOv8."""
        results = self.yolo_model(frame, classes=[0])
        person_boxes = []

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    if confidence > 0.3:
                        w, h = int(x2-x1), int(y2-y1)
                        if w > 30 and h > 60 and h/w > 1.2:
                            person_boxes.append({
                                'bbox': (int(x1), int(y1), w, h),
                                'confidence': float(confidence)
                            })
        return person_boxes

    def extract_pose_from_bbox(self, frame, bbox):
        x, y, w, h = bbox
        x = max(0, x)
        y = max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)

        if w <= 0 or h <= 0:
            return None
        person_roi = frame[y:y+h, x:x+w]

        if person_roi.size == 0:
            return None
        person_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
        results = self.pose.process(person_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks
            for landmark in landmarks.landmark:
                landmark.x = (landmark.x * w + x) / frame.shape[1]
                landmark.y = (landmark.y * h + y) / frame.shape[0]
            return landmarks
        return None

    def match_person_to_tracking(self, person_box, person_tracking):
        if not person_tracking:
            return None

        x, y, w, h = person_box['bbox']
        current_center = (x + w//2, y + h//2)

        best_match = None
        best_distance = float('inf')

        for person_id, tracking_data in person_tracking.items():
            if not tracking_data['bbox_history']:
                continue
            last_bbox = tracking_data['bbox_history'][-1]
            last_center = (last_bbox[0] + last_bbox[2] //
                           2, last_bbox[1] + last_bbox[3]//2)
            distance = ((current_center[0] - last_center[0])**2 +
                        (current_center[1] - last_center[1])**2)**0.5
            if distance < 100 and distance < best_distance:
                best_match = person_id
                best_distance = distance
        return best_match

    def detect_from_video(self, video_path, visualize=True, save_path=None,
                          skip_frames=2, batch_size=5):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'error': 'Cannot open video'}
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Processing video: {video_path}")
        print(f"Video FPS: {fps}, Total Frames: {total_video_frames}")
        print(f"Processing every {skip_frames} frames for speed optimization")
        print("=" * 50)

        writer = None
        total_frames = 0
        processed_frames = 0
        prediction_history = []
        all_confidences = []

        person_tracking = {}
        next_person_id = 1

        frame_buffer = []
        last_detections = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            total_frames += 1

            if total_frames % skip_frames != 0:
                if visualize and last_detections:
                    self.draw_multi_person_detection(
                        frame, last_detections, person_tracking, prediction_history)
                    cv2.imshow('YOLO Pose Theft Detection', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                continue

            processed_frames += 1

            person_boxes = self.detect_persons_yolo(frame)
            last_detections = person_boxes

            if processed_frames % 30 == 0:
                print(
                    f"Processed Frame {processed_frames}: Detected {len(person_boxes)} persons")

            current_person_ids = []

            for i, person_box in enumerate(person_boxes):
                x, y, w, h = person_box['bbox']

                person_id = self.match_person_to_tracking(
                    person_box, person_tracking)

                if person_id is None:
                    person_id = next_person_id
                    next_person_id += 1
                    person_tracking[person_id] = {
                        'predictions': [],
                        'confidences': [],
                        'bbox_history': [],
                        'last_seen': total_frames
                    }

                current_person_ids.append(person_id)
                person_tracking[person_id]['bbox_history'].append(
                    person_box['bbox'])
                person_tracking[person_id]['last_seen'] = total_frames

                pose_landmarks = self.extract_pose_from_bbox(
                    frame, person_box['bbox'])

                if pose_landmarks:

                    pred_info = self.lstm_detector.process_frame(
                        pose_landmarks)

                    if pred_info and pred_info.get('ready'):
                        person_tracking[person_id]['predictions'].append(
                            pred_info['prediction'])
                        person_tracking[person_id]['confidences'].append(
                            pred_info['confidence'])
                        prediction_history.append(pred_info['prediction'])
                        all_confidences.append(pred_info['confidence'])

            to_remove = []
            for person_id in person_tracking:
                if person_id not in current_person_ids and total_frames - person_tracking[person_id]['last_seen'] > 30:
                    to_remove.append(person_id)

            for person_id in to_remove:
                del person_tracking[person_id]

            if visualize:
                self.draw_multi_person_detection(
                    frame, person_boxes, person_tracking, prediction_history)
                cv2.imshow('YOLO Pose Theft Detection', frame)

                if writer is None and save_path:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(save_path, fourcc, 30.0, (w, h))

                if writer:
                    writer.write(frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        if writer:
            writer.release()
        if visualize:
            cv2.destroyAllWindows()

        if prediction_history:
            theft_ratio = sum(prediction_history) / len(prediction_history)
            avg_confidence = np.mean(all_confidences)
            max_confidence = np.max(all_confidences)
            theft_detected = (theft_ratio >= 0.5 or  # 50% threshold OR
                              max_confidence >= 0.8 or  # Very high single-frame confidence OR
                              avg_confidence >= 0.6)    # High average confidence
        else:
            theft_ratio = 0
            avg_confidence = 0
            max_confidence = 0
            theft_detected = False

        results = {
            'theft': theft_detected,
            'theft_ratio': theft_ratio,
            'avg_confidence': avg_confidence,
            'max_confidence': max_confidence,
            'total_frames': total_frames,
            'processed_frames': processed_frames,
            'predictions_made': len(prediction_history),
            'video_path': video_path,
            'performance': {
                'skip_frames': skip_frames,
                'speedup_factor': skip_frames,
                'processing_efficiency': f"{processed_frames}/{total_frames} frames"
            }
        }

        print("\n" + "=" * 50)
        print("YOLO POSE DETECTION RESULTS (OPTIMIZED)")
        print("=" * 50)
        print(f"Theft Detected: {'YES' if theft_detected else 'NO'}")
        print(f"Confidence: {theft_ratio:.2%} positive frames")
        print(f"Total Frames: {total_frames}")
        print(
            f"Processed Frames: {processed_frames} (every {skip_frames} frames)")
        print(f"Speed Improvement: {skip_frames}x faster")
        print(f"Predictions Made: {len(prediction_history)}")
        print(f"Average Confidence: {avg_confidence:.2%}")
        print("=" * 50)

        return results

    def draw_multi_person_detection(self, frame, person_boxes, person_tracking, prediction_history):
        for i, person_box in enumerate(person_boxes):
            x, y, w, h = person_box['bbox']
            confidence = person_box['confidence']

            person_id = self.match_person_to_tracking(
                person_box, person_tracking)

            if person_id and person_id in person_tracking:
                tracking_data = person_tracking[person_id]

                if tracking_data['predictions']:
                    recent_predictions = tracking_data['predictions'][-10:]
                    theft_probability = sum(
                        recent_predictions) / len(recent_predictions)
                    non_theft_probability = 1 - theft_probability

                    if theft_probability > 0.8:
                        color = (0, 0, 255)
                        status = "SHOPLIFTING"
                    elif theft_probability > 0.5:
                        color = (0, 165, 255)
                        status = "SUSPICIOUS"
                    else:
                        color = (0, 255, 0)
                        status = "NORMAL"

                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

                    label = f"Person_{person_id}: {status} ({theft_probability:.2%})"
                    cv2.putText(frame, label, (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    cv2.putText(frame, f"Det: {confidence:.2f}", (x, y+h+20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                else:
                    color = (128, 128, 128)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, f"Person_{person_id}: Analyzing...", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                color = (255, 255, 0)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f"New Person: {confidence:.2f}", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if prediction_history:
            recent_predictions = prediction_history[-30:]
            positive_count = sum(recent_predictions)
            current_ratio = positive_count / len(recent_predictions)

            overall_status = "SHOPLIFTING DETECTED" if current_ratio >= 0.5 else "NORMAL"
            status_color = (
                0, 0, 255) if overall_status == "SHOPLIFTING DETECTED" else (0, 255, 0)

            cv2.putText(frame, f"Overall Status: {overall_status} ({current_ratio:.2%})",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            cv2.putText(frame, f"People Tracked: {len(person_tracking)}",
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Frames Analyzed: {len(prediction_history)}",
                        (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "Analyzing...", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)


def detect_theft_yolo_pose(video_path, model_path='Behaveioral/theft_detector_lstm.keras',
                           scaler_path='Behaveioral/scaler_lstm.pkl',
                           sequence_length=30, visualize=True, save_path=None,
                           skip_frames=2, batch_size=5):
    detector = YOLOPoseTheftDetector(
        model_path=model_path,
        scaler_path=scaler_path,
        sequence_length=sequence_length
    )

    return detector.detect_from_video(
        video_path=video_path,
        visualize=visualize,
        save_path=save_path,
        skip_frames=skip_frames,
        batch_size=batch_size
    )
