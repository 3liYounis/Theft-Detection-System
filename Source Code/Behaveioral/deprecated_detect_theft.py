# import cv2
# import mediapipe as mp
# import math
# from collections import deque
# from Detection.detect_item import detect_products


# class TheftState:
#     IDLE = 0
#     REACH = 1
#     TRANSPORT = 2
#     CONCEAL = 3


# def distance(p1, p2):
#     return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)


# def iou(boxA, boxB):
#     xA = max(boxA[0], boxB[0])
#     yA = max(boxA[1], boxB[1])
#     xB = min(boxA[2], boxB[2])
#     yB = min(boxA[3], boxB[3])
#     interW = max(0, xB - xA)
#     interH = max(0, yB - yA)
#     inter = interW * interH
#     areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
#     areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
#     union = areaA + areaB - inter if (areaA + areaB - inter) > 0 else 1
#     return inter / union


# def get_landmark(pose_landmarks, name):
#     return pose_landmarks.landmark[name]


# def is_hand_near_hip(pose_landmarks, side='R', threshold=0.1):
#     wrist_lm = mp.solutions.pose.PoseLandmark.RIGHT_WRIST if side == 'R' else mp.solutions.pose.PoseLandmark.LEFT_WRIST
#     hip_lm = mp.solutions.pose.PoseLandmark.RIGHT_HIP if side == 'R' else mp.solutions.pose.PoseLandmark.LEFT_HIP
#     wrist = get_landmark(pose_landmarks, wrist_lm)
#     hip = get_landmark(pose_landmarks, hip_lm)
#     return distance(wrist, hip) < threshold


# def is_arm_extended_forward(pose_landmarks, side='R', min_extension=0.25):
#     """Arm considered extended if wrist is far from shoulder (normalized)."""
#     wrist_lm = mp.solutions.pose.PoseLandmark.RIGHT_WRIST if side == 'R' else mp.solutions.pose.PoseLandmark.LEFT_WRIST
#     shoulder_lm = mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER if side == 'R' else mp.solutions.pose.PoseLandmark.LEFT_SHOULDER
#     wrist = get_landmark(pose_landmarks, wrist_lm)
#     shoulder = get_landmark(pose_landmarks, shoulder_lm)
#     return distance(wrist, shoulder) > min_extension


# def wrist_velocity_towards_hip(prev_wrist, curr_wrist, hip, min_delta=0.02):
#     """Positive if wrist moved closer to hip by at least min_delta (normalized)."""
#     prev_d = math.sqrt((prev_wrist.x - hip.x)**2 + (prev_wrist.y - hip.y)**2)
#     curr_d = math.sqrt((curr_wrist.x - hip.x)**2 + (curr_wrist.y - hip.y)**2)
#     return (prev_d - curr_d) > min_delta


# def get_bounding_rect(pose_landmarks, image_width, image_height):
#     x_coords = [landmark.x for landmark in pose_landmarks.landmark]
#     y_coords = [landmark.y for landmark in pose_landmarks.landmark]

#     x_min = int(min(x_coords) * image_width)
#     x_max = int(max(x_coords) * image_width)
#     y_min = int(min(y_coords) * image_height)
#     y_max = int(max(y_coords) * image_height)

#     x_min = max(0, x_min)
#     y_min = max(0, y_min)
#     x_max = min(image_width - 1, x_max)
#     y_max = min(image_height - 1, y_max)

#     return x_min, y_min, x_max, y_max


# def wrist_bbox(wrist, image_width, image_height, size_ratio=0.12):
#     """Axis-aligned square around wrist; size relative to min(image dims)."""
#     cx = int(wrist.x * image_width)
#     cy = int(wrist.y * image_height)
#     s = int(min(image_width, image_height) * size_ratio)
#     x1 = max(0, cx - s // 2)
#     y1 = max(0, cy - s // 2)
#     x2 = min(image_width - 1, cx + s // 2)
#     y2 = min(image_height - 1, cy + s // 2)
#     return (x1, y1, x2, y2)


# def detect_theft(stream_path: str = None):
#     video = cv2.VideoCapture(stream_path if stream_path else 0)
#     mp_pose = mp.solutions.pose
#     mp_drawing = mp.solutions.drawing_utils

#     image_frame = 0
#     suspicious_count = 0
#     cumulative_score = 0
#     stealing_percentage = 0.0
#     recent_flags = deque(maxlen=120)  # ~4 seconds at 30 FPS
#     consecutive_suspicious = 0
#     max_consecutive_suspicious = 0

#     # FSM variables per hand
#     fsm = {
#         'R': {
#             'state': TheftState.IDLE,
#             'frames': 0,
#             'last_wrist': None,
#             'reach_frame': None,
#             'transport_frame': None,
#             'conceal_frame': None,
#         },
#         'L': {
#             'state': TheftState.IDLE,
#             'frames': 0,
#             'last_wrist': None,
#             'reach_frame': None,
#             'transport_frame': None,
#             'conceal_frame': None,
#         }
#     }

#     with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
#         while video.isOpened():
#             ret, frame = video.read()
#             if not ret:
#                 break

#             image_frame += 1
#             img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = pose.process(img_rgb)

#             height, width, _ = frame.shape

#             if results.pose_landmarks:
#                 mp_drawing.draw_landmarks(
#                     frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

#                 x_min, y_min, x_max, y_max = get_bounding_rect(
#                     results.pose_landmarks, width, height)
#                 cv2.rectangle(frame, (x_min, y_min),
#                               (x_max, y_max), (0, 255, 0), 2)

#                 # Heuristic suspiciousness for telemetry (either hand near hip)
#                 is_suspicious = (
#                     is_hand_near_hip(results.pose_landmarks, 'R') or
#                     is_hand_near_hip(results.pose_landmarks, 'L')
#                 )
#                 recent_flags.append(1 if is_suspicious else 0)
#                 if is_suspicious:
#                     suspicious_count += 1
#                     cumulative_score += 1
#                     consecutive_suspicious += 1
#                     max_consecutive_suspicious = max(
#                         max_consecutive_suspicious, consecutive_suspicious)
#                     cv2.putText(frame, 'Suspicious Hand Movement Detected!',
#                                 (x_min, y_max + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
#                 else:
#                     consecutive_suspicious = 0

#                 # Detect products once per frame
#                 dets = detect_products(frame)

#                 # ---------------- Symmetric FSM per hand ----------------
#                 for side in ('R', 'L'):
#                     wrist_lm = mp.solutions.pose.PoseLandmark.RIGHT_WRIST if side == 'R' else mp.solutions.pose.PoseLandmark.LEFT_WRIST
#                     hip_lm = mp.solutions.pose.PoseLandmark.RIGHT_HIP if side == 'R' else mp.solutions.pose.PoseLandmark.LEFT_HIP

#                     wrist = get_landmark(results.pose_landmarks, wrist_lm)
#                     hip = get_landmark(results.pose_landmarks, hip_lm)

#                     arm_extended = is_arm_extended_forward(
#                         results.pose_landmarks, side=side)
#                     near_hip = is_hand_near_hip(
#                         results.pose_landmarks, side=side, threshold=0.1)

#                     # Require an item near the wrist for key transitions
#                     wbox = wrist_bbox(wrist, width, height)
#                     near_item = any(
#                         iou(wbox, det['bbox']) > 0.1 for det in dets) if dets else False

#                     moved_towards_hip = False
#                     if fsm[side]['last_wrist'] is not None:
#                         moved_towards_hip = wrist_velocity_towards_hip(
#                             fsm[side]['last_wrist'], wrist, hip)

#                     if fsm[side]['state'] == TheftState.IDLE:
#                         if arm_extended and near_item:
#                             fsm[side]['frames'] += 1
#                             if fsm[side]['frames'] >= 5:
#                                 fsm[side]['state'] = TheftState.REACH
#                                 fsm[side]['reach_frame'] = image_frame
#                                 fsm[side]['frames'] = 0
#                         else:
#                             fsm[side]['frames'] = 0
#                     elif fsm[side]['state'] == TheftState.REACH:
#                         fsm[side]['frames'] += 1
#                         if moved_towards_hip and near_item:
#                             fsm[side]['state'] = TheftState.TRANSPORT
#                             fsm[side]['transport_frame'] = image_frame
#                             fsm[side]['frames'] = 0
#                         elif fsm[side]['frames'] > 45:
#                             fsm[side]['state'] = TheftState.IDLE
#                             fsm[side]['frames'] = 0
#                     elif fsm[side]['state'] == TheftState.TRANSPORT:
#                         fsm[side]['frames'] += 1
#                         if near_hip and near_item:
#                             if fsm[side]['frames'] >= 15:
#                                 fsm[side]['state'] = TheftState.CONCEAL
#                                 fsm[side]['conceal_frame'] = image_frame
#                                 fsm[side]['frames'] = 0
#                         elif fsm[side]['frames'] > 90:
#                             fsm[side]['state'] = TheftState.IDLE
#                             fsm[side]['frames'] = 0
#                     elif fsm[side]['state'] == TheftState.CONCEAL:
#                         fsm[side]['frames'] += 1
#                         if not near_hip and fsm[side]['frames'] > 60:
#                             fsm[side]['state'] = TheftState.IDLE
#                             fsm[side]['frames'] = 0

#                     fsm[side]['last_wrist'] = wrist

#                 if image_frame > 0:
#                     stealing_percentage = (
#                         suspicious_count / image_frame) * 100

#                 text_y = y_min - 10 if y_min - 10 > 20 else y_min + 20

#                 cv2.putText(frame,
#                             f"Stealing %: {stealing_percentage:.2f}%",
#                             (x_min, text_y),
#                             cv2.FONT_HERSHEY_SIMPLEX,
#                             0.7,
#                             (255, 255, 0),
#                             2)

#                 cv2.putText(frame,
#                             f"Cumulative Score: {cumulative_score}",
#                             (x_min, text_y - 25),
#                             cv2.FONT_HERSHEY_SIMPLEX,
#                             0.7,
#                             (0, 255, 255),
#                             2)

#             cv2.imshow('Theft & Behavioral Detection', frame)

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#     video.release()
#     cv2.destroyAllWindows()

#     if image_frame > 0:
#         stealing_percentage = (suspicious_count / image_frame) * 100
#     window_sum = sum(recent_flags) if recent_flags else 0
#     window_ratio = (window_sum / len(recent_flags)) if recent_flags else 0.0

#     conceal_detected = (fsm['R']['conceal_frame'] is not None) or (
#         fsm['L']['conceal_frame'] is not None)
#     theft_flag = conceal_detected or (window_ratio > 0.6 and window_sum >= 60) or (
#         max_consecutive_suspicious >= 45)
#     return {
#         "frames_processed": image_frame,
#         "suspicious_movements_detected": suspicious_count,
#         "cumulative_score": cumulative_score,
#         "stealing_percentage": stealing_percentage,
#         "window_ratio": window_ratio,
#         "max_consecutive_suspicious": max_consecutive_suspicious,
#         "fsm_right": fsm['R'],
#         "fsm_left": fsm['L'],
#         "theft": theft_flag,
#         "video_path": stream_path if stream_path else "camera"
#     }
