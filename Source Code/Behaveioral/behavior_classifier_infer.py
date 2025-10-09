import cv2
import mediapipe as mp
import joblib
from collections import deque


def extract_pose_features(pose_landmarks):
    features = []
    for lm in pose_landmarks.landmark:
        features.extend([lm.x, lm.y, lm.z])
    return features

# Return dict with theft decision based on sliding window of classifier outputs.

# - window_size: number of recent frames to aggregate (e.g., 60 ≈ 2 seconds @30FPS)
# - positive_ratio: fraction of positives in window to declare theft


def detect_theft_ml(video_path, model_path="pose_behavior_classifier.joblib", window_size=60, positive_ratio=0.5, visualize=False, save_path=None):
    clf = joblib.load(model_path)
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        return {"theft": False, "reason": "cannot_open_video", "video_path": video_path}

    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    window = deque(maxlen=window_size)
    total_frames = 0
    positive_frames = 0
    writer = None

    with mp_pose.Pose(min_detection_confidence=0.5) as pose:
        while True:
            ret, frame = video.read()
            if not ret:
                break
            total_frames += 1

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)

            if results.pose_landmarks:
                feats = extract_pose_features(results.pose_landmarks)
                # 1 = suspicious/shoplifting
                pred = int(clf.predict([feats])[0])
                if visualize:
                    mp_draw.draw_landmarks(
                        frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            else:
                pred = 0

            window.append(pred)
            if pred == 1:
                positive_frames += 1

            if visualize:
                ratio = (sum(window) / len(window)) if len(window) > 0 else 0.0
                label = "Shoplifting" if ratio >= positive_ratio else "Normal"
                color = (0, 0, 255) if label == "Shoplifting" else (0, 200, 0)
                cv2.putText(frame, f"{label}  ratio:{ratio:.2f}  pos:{int(sum(window))}/{len(window)}",
                            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                if writer is None and save_path:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(save_path, fourcc, 30.0, (w, h))
                if writer is not None:
                    writer.write(frame)
                cv2.imshow("Behavior ML Visualization", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    video.release()
    if writer is not None:
        writer.release()
    if visualize:
        cv2.destroyAllWindows()

    ratio = (sum(window) / len(window)) if len(window) > 0 else 0.0
    theft_flag = ratio >= positive_ratio and sum(
        window) >= int(window_size * positive_ratio)

    return {
        "theft": theft_flag,
        "frames_processed": total_frames,
        "window_size": window_size,
        "window_positive": int(sum(window)),
        "window_ratio": ratio,
        "positive_frames_total": positive_frames,
        "video_path": video_path,
    }
