import io
import os
import cv2
import sqlite3
import numpy as np
import mediapipe as mp
from feature_extraction import FeatureExtractor


def init_db(db_path):
    extractor = FeatureExtractor()
    feature_names = extractor.get_feature_names()
    feature_columns = ", ".join([f"{name} REAL" for name in feature_names])

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON")

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sequences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_name TEXT,
            label INTEGER,
            start_frame INTEGER,
            length INTEGER
        )
    ''')

    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_name TEXT,
            frame_index INTEGER,
            {feature_columns},
            UNIQUE(video_name, frame_index) ON CONFLICT IGNORE
        )
    ''')

    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_features_video_frame ON features (video_name, frame_index)")

    conn.commit()
    conn.close()
    return feature_names


def save_data_to_db(db_path, video_name, label, all_features_array, sequences_metadata, feature_names):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    placeholders = ", ".join(["?"] * len(feature_names))
    feature_insert_sql = f"""
        INSERT OR IGNORE INTO features (video_name, frame_index, {", ".join(feature_names)})
        VALUES (?, ?, {placeholders})
    """

    rows_to_insert = []
    for i, frame_features in enumerate(all_features_array):
        row = [video_name, i] + frame_features.tolist()
        rows_to_insert.append(row)

    cursor.executemany(feature_insert_sql, rows_to_insert)

    seq_insert_sql = "INSERT INTO sequences (video_name, label, start_frame, length) VALUES (?, ?, ?, ?)"
    seq_rows = [(video_name, label, start, length)
                for start, length in sequences_metadata]

    cursor.executemany(seq_insert_sql, seq_rows)

    count = len(seq_rows)
    conn.commit()
    conn.close()
    return count


def check_existing_sequences(db_path, video_name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT COUNT(*) FROM sequences WHERE video_name = ?", (video_name,))
    count = cursor.fetchone()[0]
    conn.close()
    return count


def collect_sequences(video_path, label, db_path, sequence_length=30, stride=10, skip_existing=True):
    video_name = os.path.basename(video_path)
    if skip_existing:
        existing_count = check_existing_sequences(db_path, video_name)
        if existing_count > 0:
            return -existing_count

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
    feature_extractor = FeatureExtractor(temporal_window=10)
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

    try:
        all_features_array = np.array(all_features)
    except ValueError as e:
        return 0

    sequences_metadata = []
    for i in range(0, len(all_features_array) - sequence_length + 1, stride):
        sequences_metadata.append((i, sequence_length))

    extractor = FeatureExtractor()
    feature_names = extractor.get_feature_names()

    saved_count = save_data_to_db(
        db_path, video_name, label, all_features_array, sequences_metadata, feature_names)
    return saved_count


if __name__ == "__main__":
    normal_dir = "../../Data/Stream/Normal/"
    shoplifting_dir = "../../Data/Stream/Shoplifting/"

    sequence_db_path = "../../Data/sequences.db"
    init_db(sequence_db_path)

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
            count = collect_sequences(video_path, label=0, db_path=sequence_db_path,
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
            count = collect_sequences(video_path, label=1, db_path=sequence_db_path,
                                      sequence_length=sequence_length, stride=stride,
                                      skip_existing=True)
            if count:
                if count < 0:
                    videos_skipped += 1
                    total_sequences += abs(count)
                else:
                    videos_processed += 1
                    total_sequences += count
