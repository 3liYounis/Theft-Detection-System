import os
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from collections import deque
from datetime import datetime
from Behaveioral.inference_lstm import LSTMTheftDetectorInference
from Recognition.recognize_face import recognize_face, get_face_info
from Alert.alert import alert

current_dir = os.path.dirname(os.path.abspath(__file__))

THEFT_THRESHOLD = 0.9
SEQUENCE_LENGTH = 90
OUTPUT_DIR = os.path.join(current_dir, "./Behaveioral/suspect_clips")
YOLO_MODEL_PATH = os.path.join(current_dir, 'yolov8n.pt')


class PersonTheftAnalyzer:
    def __init__(self, track_id):
        self.track_id = track_id
        self.detector = LSTMTheftDetectorInference(
            model_path=os.path.join(
                current_dir, 'Behaveioral/model/theft_detector_lstm.keras'),
            scaler_path=os.path.join(
                current_dir, 'Behaveioral/scaler/scaler_lstm.pkl'),
            sequence_length=SEQUENCE_LENGTH
        )
        self.history = deque(maxlen=90)
        self.is_suspect = False
        self.frames = []

    def update(self, landmarks):
        result = self.detector.process_frame(landmarks)
        if result and result.get('ready'):
            return result['confidence']
        return 0.0


def run_theft_detection_pipeline(video_path=None):
    yolo_model = YOLO(YOLO_MODEL_PATH, verbose=False)

    mp_pose = mp.solutions.pose
    pose_estimator = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    if video_path:
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = os.path.join(current_dir, "output_main_pipeline.mp4")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    analyzers = {}
    suspect_ids = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model.track(
            frame, persist=True, classes=[0], verbose=False)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create blurred background once per frame (Optimization)
        frame_blurred = cv2.GaussianBlur(frame, (99, 99), 30)

        if results and results[0].boxes and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()

            for box, track_id in zip(boxes, track_ids):
                if track_id not in analyzers:
                    analyzers[track_id] = PersonTheftAnalyzer(track_id)

                analyzer = analyzers[track_id]
                analyzer = analyzers[track_id]

                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                # Create isolated frame: blurred background + clear suspect
                isolated_frame = frame_blurred.copy()
                if x2 > x1 and y2 > y1:
                    isolated_frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2]

                analyzer.frames.append(isolated_frame)

                if x2 - x1 < 10 or y2 - y1 < 10:
                    continue

                # Use full-frame isolated input for consistency with training data
                isolated_rgb = cv2.cvtColor(isolated_frame, cv2.COLOR_BGR2RGB)
                pose_results = pose_estimator.process(isolated_rgb)

                prob = 0.0

                label_text = "Normal"
                color = (0, 255, 0)

                if pose_results.pose_landmarks:
                    prob = analyzer.update(pose_results.pose_landmarks)
                    analyzer.history.append(prob)

                    if analyzer.history:
                        avg_prob = sum(analyzer.history) / \
                            len(analyzer.history)
                    else:
                        avg_prob = 0.0

                    prob_percent = avg_prob * 100

                    if prob_percent <= 60:
                        label_text = f"Normal {prob_percent:.1f}%"
                        color = (0, 255, 0)
                    elif prob_percent < 90:
                        label_text = f"Suspicious {prob_percent:.1f}%"
                        color = (0, 180, 255)
                    else:
                        label_text = f"Shoplifting {prob_percent:.1f}%"
                        color = (0, 0, 255)

                        analyzer.is_suspect = True
                        suspect_ids.add(track_id)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                (text_w, text_h), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_ITALIC, 1.0, 2)
                cv2.rectangle(frame, (x1, y1 - text_h - 10),
                              (x1 + text_w, y1), color, -1)
                cv2.putText(frame, label_text, (x1, y1 - 10),
                            cv2.FONT_ITALIC, 1, (255, 255, 255), 2, cv2.LINE_AA)

        out.write(frame)
        cv2.imshow("Theft Detection Realtime", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    pose_estimator.close()
    cv2.destroyAllWindows()

    if not suspect_ids:
        return

    for track_id in suspect_ids:
        analyzer = analyzers[track_id]
        if not analyzer.frames:
            continue

        clip_name = f"suspect_{track_id}.mp4"
        clip_path = os.path.abspath(os.path.join(OUTPUT_DIR, clip_name))

        h_c, w_c = analyzer.frames[0].shape[:2]
        c_out = cv2.VideoWriter(
            clip_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (w_c, h_c))
        for f in analyzer.frames:
            c_out.write(f)
        c_out.release()

        face_result = recognize_face(clip_path)

        matched_name = face_result.get('match')
        face_info = None
        if matched_name and matched_name != "Unknown":
            face_info = get_face_info(matched_name)

        if not face_info:
            face_info = {
                "id": "Unknown",
                "name": matched_name if matched_name else "Unknown",
                "phone": "N/A",
                "age": "N/A",
                "image": "N/A"
            }

        alert(
            theft_link=clip_path,
            face=face_info,
            time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )


if __name__ == "__main__":
    video_path = "../Data/Stream/TRY/5.mp4"
    run_theft_detection_pipeline()
