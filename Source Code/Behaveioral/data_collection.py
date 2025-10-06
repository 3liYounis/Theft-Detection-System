import cv2
import mediapipe as mp
import numpy as np
import csv
import os


def extract_pose_features(pose_landmarks):
    # Extract normalized landmark coordinates as flat array (x1,y1,x2,y2,...)
    features = []
    for lm in pose_landmarks.landmark:
        features.extend([lm.x, lm.y, lm.z])
    return features


def collect_data(video_path, label, output_csv):
    video = cv2.VideoCapture(video_path)
    mp_pose = mp.solutions.pose

    with mp_pose.Pose(min_detection_confidence=0.5) as pose, \
            open(output_csv, 'a', newline='') as f:

        writer = csv.writer(f)
        while True:
            ret, frame = video.read()
            if not ret:
                break

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)

            if results.pose_landmarks:
                features = extract_pose_features(results.pose_landmarks)
                writer.writerow(features + [label])

            cv2.imshow('Collecting Data', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    normal_dir = "./Data/Stream/Normal"
    shoplifting_dir = "./Data/Stream/Shoplifting"
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
