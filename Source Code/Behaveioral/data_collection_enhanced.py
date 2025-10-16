from feature_extraction import EnhancedFeatureExtractor, augment_pose_data
import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import sys
sys.path.append(os.path.dirname(__file__))

try:
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    MP_TASKS_AVAILABLE = True
except Exception:
    MP_TASKS_AVAILABLE = False

MODEL_TASK_PATH = "./Resources/Models/pose_landmarker_full.task"


def collect_data_enhanced(video_path, label, output_csv, augment=True):
    """
    Collect enhanced features from video with optional data augmentation.

    Args:
        video_path: path to video file
        label: 0 for normal, 1 for shoplifting
        output_csv: path to output CSV file
        augment: whether to apply data augmentation
    """
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Cannot open video: {video_path}")
        return
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    feature_extractor = EnhancedFeatureExtractor(temporal_window=10)
    frame_count = 0
    features_collected = 0
    with open(output_csv, 'a', newline='') as f:
        writer = csv.writer(f)
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
                    if len(feature_extractor.pose_history) >= feature_extractor.temporal_window:
                        writer.writerow(list(features) + [label])
                        features_collected += 1
                        # Data augmentation - create additional training samples
                        if augment and features_collected % 3 == 0:  # Augment every 3rd frame
                            lms = results.pose_landmarks.landmark
                            landmarks_array = np.array(
                                [[lm.x, lm.y, lm.z] for lm in lms])
                            # Augmentation 1: Horizontal flip
                            augmented_flip = augment_pose_data(
                                landmarks_array, flip=True, noise_level=0)
                            # Create fake pose_landmarks object
                            aug_features = feature_extractor.extract_spatial_features(
                                type('PoseLandmarks', (), {'landmark': [
                                    type('Landmark', (), {
                                         'x': pt[0], 'y': pt[1], 'z': pt[2]})()
                                    for pt in augmented_flip
                                ]})()
                            )
                            temporal_feats = features[len(aug_features):]
                            combined = np.concatenate(
                                [aug_features, temporal_feats])
                            writer.writerow(list(combined) + [label])
                            features_collected += 1
                            # Augmentation 2: Add noise
                            augmented_noise = augment_pose_data(
                                landmarks_array, flip=False, noise_level=0.01)
                            aug_features = feature_extractor.extract_spatial_features(
                                type('PoseLandmarks', (), {'landmark': [
                                    type('Landmark', (), {
                                         'x': pt[0], 'y': pt[1], 'z': pt[2]})()
                                    for pt in augmented_noise
                                ]})()
                            )
                            combined = np.concatenate(
                                [aug_features, temporal_feats])
                            writer.writerow(list(combined) + [label])
                            features_collected += 1
                except Exception as e:
                    print(f"Error extracting features: {e}")
                    continue
            if frame_count % 30 == 0:
                cv2.putText(frame, f"Frames: {frame_count} | Features: {features_collected}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('Collecting Enhanced Data', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    video.release()
    cv2.destroyAllWindows()
    pose.close()

    print(
        f"Processed {video_path}: {frame_count} frames, {features_collected} feature sets collected")


def collect_sequences(video_path, label, output_dir, sequence_length=30, stride=10, skip_existing=True):
    """
    Collect sequences for LSTM training.
    Instead of individual frames, collect overlapping sequences.

    Args:
        video_path: path to video file
        label: 0 for normal, 1 for shoplifting
        output_dir: directory to save sequence files
        sequence_length: number of frames per sequence
        stride: step size for sliding window
        skip_existing: if True, skip videos that have already been processed
    """
    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.basename(video_path).replace('.mp4', '')
    output_file = os.path.join(output_dir, f"{video_name}_label{label}.npz")

    if skip_existing and os.path.exists(output_file):
        print(f"✓ Skipping {video_name} (already processed)")
        try:
            data = np.load(output_file)
            return -len(data['sequences'])
        except:
            print(f"  Warning: Could not read existing file, will reprocess")

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"✗ Cannot open video: {video_path}")
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
    print(
        f"Extracted {len(all_features)} feature vectors from {frame_count} frames")
    if len(all_features) < sequence_length:
        print(
            f"Not enough frames ({len(all_features)}) for sequence length {sequence_length}")
        return 0
    try:
        all_features_array = np.array(all_features)
        print(f"Feature shape: {all_features_array.shape}")
    except ValueError as e:
        print(f"Error converting features to array: {e}")
        print(f"Feature lengths: {[len(f) for f in all_features[:5]]}")
        return 0

    sequences = []
    for i in range(0, len(all_features_array) - sequence_length + 1, stride):
        sequence = all_features_array[i:i + sequence_length]
        sequences.append(sequence)
    sequences_array = np.array(sequences)
    print(
        f"Created {len(sequences)} sequences with shape: {sequences_array.shape}")
    np.savez_compressed(output_file,
                        sequences=sequences_array,
                        labels=np.array([label] * len(sequences)))

    print(f"✓ Saved {len(sequences)} sequences to {output_file}")
    return len(sequences)


if __name__ == "__main__":
    normal_dir = "../../Data/Stream/Normal - Train/"
    shoplifting_dir = "../../Data/Stream/Shoplifting - Train/"

    # Option 1: Collect frame-by-frame features (for baseline)
    print("=" * 50)
    print("Collecting enhanced frame-level features...")
    print("=" * 50)

    output_csv = "pose_data_enhanced.csv"

    # if os.path.exists(output_csv):
    #     os.remove(output_csv)

    # if os.path.exists(normal_dir):
    #     for filename in sorted(os.listdir(normal_dir)):
    #         if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
    #             video_path = os.path.join(normal_dir, filename)
    #             print(f"Processing normal video: {filename}")
    #             collect_data_enhanced(
    #                 video_path, label=0, output_csv=output_csv, augment=True)

    # if os.path.exists(shoplifting_dir):
    #     for filename in sorted(os.listdir(shoplifting_dir)):
    #         if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
    #             video_path = os.path.join(shoplifting_dir, filename)
    #             print(f"Processing shoplifting video: {filename}")
    #             collect_data_enhanced(
    #                 video_path, label=1, output_csv=output_csv, augment=True)

    # print(f"\nFrame-level features saved to {output_csv}")

    # Option 2: Collect sequences for LSTM
    print("\n" + "=" * 50)
    print("Collecting sequences for LSTM training...")
    print("=" * 50)

    sequence_output_dir = "../../Data/Sequences"
    sequence_length = 30
    stride = 10

    total_sequences = 0
    videos_processed = 0
    videos_skipped = 0

    if os.path.exists(normal_dir):
        normal_videos = sorted([f for f in os.listdir(normal_dir)
                               if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))])
        print(f"\nProcessing {len(normal_videos)} normal videos...")
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

    print("\n" + "=" * 50)
    print("DATA COLLECTION COMPLETE!")
    print("=" * 50)
    print(f"Total sequences: {total_sequences}")
    print(f"Videos newly processed: {videos_processed}")
    print(f"Videos skipped (already done): {videos_skipped}")
    print(f"Sequences saved to: {sequence_output_dir}/")
    print("\nNext step: Run 'python train_lstm.py' to train the model")
    print("=" * 50)
