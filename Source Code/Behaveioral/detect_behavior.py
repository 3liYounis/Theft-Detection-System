# import cv2
# import mediapipe as mp
# import numpy as np
# import joblib


# def extract_pose_features(pose_landmarks):
#     features = []
#     for lm in pose_landmarks.landmark:
#         features.extend([lm.x, lm.y, lm.z])
#     return features


# def detect_behavior(video_path, model_path):
#     clf = joblib.load(model_path)
#     video = cv2.VideoCapture(video_path)
#     mp_pose = mp.solutions.pose
#     mp_drawing = mp.solutions.drawing_utils

#     suspicious_count = 0
#     frame_count = 0

#     with mp_pose.Pose(min_detection_confidence=0.5) as pose:
#         while video.isOpened():
#             ret, frame = video.read()
#             if not ret:
#                 break

#             frame_count += 1
#             img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = pose.process(img_rgb)

#             if results.pose_landmarks:
#                 mp_drawing.draw_landmarks(
#                     frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

#                 features = extract_pose_features(results.pose_landmarks)
#                 pred = clf.predict([features])[0]

#                 label = "Suspicious" if pred == 1 else "Normal"
#                 if pred == 1:
#                     suspicious_count += 1

#                 cv2.putText(frame, label, (30, 30),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1,
#                             (0, 0, 255) if pred == 1 else (0, 255, 0), 2)

#             cv2.imshow('Behavior Detection with ML', frame)

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#     video.release()
#     cv2.destroyAllWindows()

#     print(
#         f"Total frames: {frame_count}, Suspicious frames: {suspicious_count}")


# if __name__ == "__main__":
#     normal_path = "./Data/Stream/Normal/Normal (85).mp4"
#     theft_path = "./Data/Stream/Shoplifting/Shoplifting (85).mp4"
#     info = detect_behavior(theft_path,
#                            "pose_behavior_classifier.joblib")
#     print(info)
