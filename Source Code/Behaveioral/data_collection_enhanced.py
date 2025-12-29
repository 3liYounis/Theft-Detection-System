from feature_extraction import EnhancedFeatureExtractor
import cv2
import mediapipe as mp
import numpy as np
import os


def collect_sequences(video_path, label, output_dir, sequence_length=30, stride=10, skip_existing=True):
    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.basename(video_path).replace('.mp4', '')
    output_file = os.path.join(output_dir, f"{video_name}_label{label}.npz")

    if skip_existing and os.path.exists(output_file):
        try:
            data = np.load(output_file)
            return -len(data['sequences'])
        except:
            print(f"Could not read existing file!")
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        return 0
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    feature_extractor = EnhancedFeatureExtractor(temporal_window=10)
    all_features = []
    frame_count = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame_count += 1
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)
        if results.pose_landmarks:
            try:
                features = feature_extractor.extract_features(
                    results.pose_landmarks)
                all_features.append(features)
            except Exception as e:
                print(f"Error: {e}")
                continue

    video.release()
    pose.close()
    if len(all_features) < sequence_length:
        return 0
    try:
        all_features_array = np.array(all_features)
    except ValueError as e:
        return 0
    sequences = []
    for i in range(0, len(all_features_array) - sequence_length + 1, stride):
        sequence = all_features_array[i:i + sequence_length]
        sequences.append(sequence)

    sequences_array = np.array(sequences)

    np.savez_compressed(output_file,
                        sequences=sequences_array,
                        labels=np.array([label] * len(sequences)))
    return len(sequences)


if __name__ == "__main__":
    normal_dir = "../../Data/Stream/Normal/"
    shoplifting_dir = "../../Data/Stream/Shoplifting/"

    sequence_output_dir = "../../Data/Sequences"
    sequence_length = 90
    stride = 10

    total_sequences = 0
    videos_processed = 0
    videos_skipped = 0

    if os.path.exists(normal_dir):
        normal_videos = sorted([f for f in os.listdir(normal_dir)
                               if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))])
        print(f"\nProcessing {len(normal_videos)} Normal Videos. . .")
        for filename in normal_videos:
            video_path = os.path.join(normal_dir, filename)
            print(
                f"\n[{videos_processed + videos_skipped + 1}/{len(normal_videos)}] {filename}")
            count = collect_sequences(video_path, label=0, output_dir=sequence_output_dir,
                                      sequence_length=sequence_length, stride=stride,
                                      skip_existing=True)
            if count:
                if count < 0:
                    videos_skipped += 1
                    total_sequences += abs(count)
                else:
                    videos_processed += 1
                    total_sequences += count

    if os.path.exists(shoplifting_dir):
        shoplifting_videos = sorted([f for f in os.listdir(shoplifting_dir)
                                    if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))])
        print(f"\nProcessing {len(shoplifting_videos)} shoplifting videos...")
        for filename in shoplifting_videos:
            video_path = os.path.join(shoplifting_dir, filename)
            print(f"\n[{videos_processed + videos_skipped + 1}] {filename}")
            count = collect_sequences(video_path, label=1, output_dir=sequence_output_dir,
                                      sequence_length=sequence_length, stride=stride,
                                      skip_existing=True)
            if count:
                if count < 0:
                    videos_skipped += 1
                    total_sequences += abs(count)
                else:
                    videos_processed += 1
                    total_sequences += count
