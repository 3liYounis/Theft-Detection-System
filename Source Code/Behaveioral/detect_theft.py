import cv2
import mediapipe as mp
import math


def distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)


def is_hand_near_hip(pose_landmarks, threshold=0.1):
    right_wrist = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
    right_hip = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_HIP]
    dist = distance(right_wrist, right_hip)
    return dist < threshold


def get_bounding_rect(pose_landmarks, image_width, image_height):
    x_coords = [landmark.x for landmark in pose_landmarks.landmark]
    y_coords = [landmark.y for landmark in pose_landmarks.landmark]

    x_min = int(min(x_coords) * image_width)
    x_max = int(max(x_coords) * image_width)
    y_min = int(min(y_coords) * image_height)
    y_max = int(max(y_coords) * image_height)

    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(image_width - 1, x_max)
    y_max = min(image_height - 1, y_max)

    return x_min, y_min, x_max, y_max


def detect_theft(stream_path: str = None):
    video = cv2.VideoCapture(stream_path if stream_path else 0)
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    image_frame = 0
    suspicious_count = 0
    cumulative_score = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            image_frame += 1
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)

            height, width, _ = frame.shape

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                x_min, y_min, x_max, y_max = get_bounding_rect(
                    results.pose_landmarks, width, height)
                cv2.rectangle(frame, (x_min, y_min),
                              (x_max, y_max), (0, 255, 0), 2)

                if is_hand_near_hip(results.pose_landmarks):
                    suspicious_count += 1
                    cumulative_score += 1
                    cv2.putText(frame, 'Suspicious Hand Movement Detected!',
                                (x_min, y_max + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                stealing_percentage = 0.0
                if image_frame > 0:
                    stealing_percentage = (
                        suspicious_count / image_frame) * 100

                # Display the scores right above the rectangle (adjust y_min to avoid going off top)
                text_y = y_min - 10 if y_min - 10 > 20 else y_min + 20

                cv2.putText(frame,
                            f"Stealing %: {stealing_percentage:.2f}%",
                            (x_min, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 0),
                            2)

                cv2.putText(frame,
                            f"Cumulative Score: {cumulative_score}",
                            (x_min, text_y - 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),
                            2)

            cv2.imshow('Theft & Behavioral Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    video.release()
    cv2.destroyAllWindows()

    return {
        "frames_processed": image_frame,
        "suspicious_movements_detected": suspicious_count,
        "cumulative_score": cumulative_score,
        "stealing_percentage": stealing_percentage
    }
