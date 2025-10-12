import cv2
import mediapipe as mp
import numpy as np
import csv
import os
try:
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    MP_TASKS_AVAILABLE = True
except Exception:
    MP_TASKS_AVAILABLE = False
MODEL_TASK_PATH = "./Resources/Models/pose_landmarker_full.task"


def extract_pose_features(pose_landmarks):
    features = []
    for lm in pose_landmarks.landmark:
        features.extend([lm.x, lm.y, lm.z])
    return features


def extract_pose_features_from_list(landmarks_list):
    features = []
    for lm in landmarks_list:
        features.extend([lm.x, lm.y, getattr(lm, 'z', 0.0)])
    return features


def collect_data(video_path, label, output_csv):
    video = cv2.VideoCapture(video_path)
    detector = None
    single_pose = None
    if MP_TASKS_AVAILABLE:
        try:
            base_options = mp_python.BaseOptions(
                model_asset_path=MODEL_TASK_PATH)
            options = mp_vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.VIDEO,
                num_poses=5,
                min_pose_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            detector = mp_vision.PoseLandmarker.create_from_options(options)
        except Exception:
            detector = None
    if detector is None:
        single_pose = mp.solutions.pose.Pose(min_detection_confidence=0.5)

    with open(output_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        frame_idx = 0
        while True:
            ret, frame = video.read()
            if not ret:
                break
            frame_idx += 1

            if detector is not None:
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if hasattr(mp_vision, 'Image') and hasattr(mp_vision, 'ImageFormat'):
                        mp_img = mp_vision.Image(
                            image_format=mp_vision.ImageFormat.SRGB,
                            data=rgb
                        )
                        result = detector.detect_for_video(
                            mp_img, frame_idx * 33)
                    else:
                        raise AttributeError('mp_vision.Image not available')

                    if result and result.pose_landmarks:
                        for lm_list in result.pose_landmarks:
                            feats = extract_pose_features_from_list(lm_list)
                            writer.writerow(feats + [label])
                except Exception:
                    print("Fallback to single-pose")
                    detector = None
                    if 'single_pose' not in locals() or single_pose is None:
                        single_pose = mp.solutions.pose.Pose(
                            min_detection_confidence=0.5)
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = single_pose.process(img_rgb)
                    if results.pose_landmarks:
                        feats = extract_pose_features(results.pose_landmarks)
                        writer.writerow(feats + [label])
            else:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = single_pose.process(img_rgb)
                if results.pose_landmarks:
                    feats = extract_pose_features(results.pose_landmarks)
                    writer.writerow(feats + [label])

            cv2.imshow('Collecting Data', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    normal_dir = "./Data/Stream/Normal - Test/"
    shoplifting_dir = "./Data/Stream/Shoplifting - Test/"
    output_csv = "pose_data.csv"

    # Process all videos in normal_dir with label=0
    for filename in os.listdir(normal_dir):
        if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_path = os.path.join(normal_dir, filename)
            print(f"Processing normal video: {video_path}")
            collect_data(video_path, label=0, output_csv=output_csv)

    # Process all videos in suspicious_dir with label=1
    for filename in os.listdir(shoplifting_dir):
        if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_path = os.path.join(shoplifting_dir, filename)
            print(f"Processing suspicious video: {video_path}")
            collect_data(video_path, label=1, output_csv=output_csv)
