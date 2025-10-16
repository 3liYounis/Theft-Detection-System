import numpy as np
from collections import deque


class EnhancedFeatureExtractor:
    """
    Enhanced feature extraction for pose-based behavior recognition.
    Includes spatial features (angles, distances), temporal features (velocity, acceleration),
    and body-relative normalization for camera-invariant detection.
    """

    def __init__(self, temporal_window=10):
        self.temporal_window = temporal_window
        self.pose_history = deque(maxlen=temporal_window)

    def reset(self):
        """Reset temporal history"""
        self.pose_history.clear()

    def extract_spatial_features(self, pose_landmarks):
        """
        Extract spatial features from a single frame.
        Returns camera-invariant features based on body geometry.
        """
        features = []
        lms = pose_landmarks.landmark if hasattr(
            pose_landmarks, 'landmark') else pose_landmarks
        landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in lms])

        # 1. NORMALIZATION - Get body center and scale
        # Hip center as body reference point
        hip_left = landmarks_array[23]   # Left hip
        hip_right = landmarks_array[24]  # Right hip
        hip_center = (hip_left + hip_right) / 2

        # Body scale (torso length for normalization)
        shoulder_center = (landmarks_array[11] + landmarks_array[12]) / 2
        torso_length = np.linalg.norm(shoulder_center - hip_center)
        torso_length = max(torso_length, 0.001)  # Avoid division by zero

        # 2. BODY-RELATIVE COORDINATES (normalized by torso length)
        # Key points: hands, elbows, shoulders, head, knees
        # Nose, shoulders, elbows, wrists, hips, knees, ankles
        key_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
        for idx in key_indices:
            relative_pos = (landmarks_array[idx] - hip_center) / torso_length
            features.extend(relative_pos)  # x, y, z

        # 3. JOINT ANGLES (critical for pose understanding)
        # Right elbow angle
        angle_r_elbow = self._calculate_angle(
            # shoulder-elbow-wrist
            landmarks_array[12], landmarks_array[14], landmarks_array[16]
        )
        features.append(angle_r_elbow)

        # Left elbow angle
        angle_l_elbow = self._calculate_angle(
            landmarks_array[11], landmarks_array[13], landmarks_array[15]
        )
        features.append(angle_l_elbow)

        # Right knee angle
        angle_r_knee = self._calculate_angle(
            # hip-knee-ankle
            landmarks_array[24], landmarks_array[26], landmarks_array[28]
        )
        features.append(angle_r_knee)

        # Left knee angle
        angle_l_knee = self._calculate_angle(
            landmarks_array[23], landmarks_array[25], landmarks_array[27]
        )
        features.append(angle_l_knee)

        # Torso bend (hip-shoulder-nose angle)
        angle_torso = self._calculate_angle(
            hip_center, shoulder_center, landmarks_array[0]
        )
        features.append(angle_torso)

        # 4. RELATIVE DISTANCES (suspicious behavior indicators)
        # Hand to hip distance (normalized)
        dist_r_hand_hip = np.linalg.norm(
            landmarks_array[16] - hip_right) / torso_length
        dist_l_hand_hip = np.linalg.norm(
            landmarks_array[15] - hip_left) / torso_length
        features.extend([dist_r_hand_hip, dist_l_hand_hip])

        # Hand height relative to hip (hiding items below waist)
        hand_r_height = (landmarks_array[16][1] - hip_center[1]) / torso_length
        hand_l_height = (landmarks_array[15][1] - hip_center[1]) / torso_length
        features.extend([hand_r_height, hand_l_height])

        # Hands to torso center distance (reaching inward)
        dist_r_hand_torso = np.linalg.norm(
            landmarks_array[16][:2] - hip_center[:2]) / torso_length
        dist_l_hand_torso = np.linalg.norm(
            landmarks_array[15][:2] - hip_center[:2]) / torso_length
        features.extend([dist_r_hand_torso, dist_l_hand_torso])

        # 5. BODY CONFIGURATION
        # Shoulder width (normalized)
        shoulder_width = np.linalg.norm(
            landmarks_array[11] - landmarks_array[12]) / torso_length
        features.append(shoulder_width)

        # Hip width (normalized)
        hip_width = np.linalg.norm(hip_left - hip_right) / torso_length
        features.append(hip_width)

        # Body symmetry (difference in left/right arm positions)
        arm_symmetry = np.linalg.norm(
            (landmarks_array[16] - hip_right) -
            (landmarks_array[15] - hip_left)
        ) / torso_length
        features.append(arm_symmetry)

        # 6. HEAD ORIENTATION
        # Head tilt (nose relative to shoulders)
        head_tilt_x = (landmarks_array[0][0] -
                       shoulder_center[0]) / torso_length
        head_tilt_y = (landmarks_array[0][1] -
                       shoulder_center[1]) / torso_length
        features.extend([head_tilt_x, head_tilt_y])

        # 7. CONFIDENCE/METADATA
        features.append(torso_length)  # Body scale (for debugging)

        return np.array(features)

    def extract_temporal_features(self):
        """
        Extract temporal features from pose history.
        SIMPLIFIED VERSION - Always returns exactly 30 features.
        """
        features = np.zeros(30)

        if len(self.pose_history) < 2:
            return features

        try:
            history_array = np.array(list(self.pose_history))

            if len(history_array) >= 2:
                velocity = history_array[-1] - history_array[-2]
                features[0:6] = velocity[9:15]

            if len(history_array) >= 3:
                velocity_prev = history_array[-2] - history_array[-3]
                acceleration = velocity - velocity_prev
                features[6:12] = acceleration[9:15]

            if len(history_array) >= 2:
                features[12] = np.linalg.norm(velocity)

            if len(history_array) >= 2:
                features[13] = np.linalg.norm(velocity[9:12])
                features[14] = np.linalg.norm(velocity[12:15])

            if len(history_array) >= 2:
                avg_position = np.mean(history_array, axis=0)
                features[15:21] = avg_position[9:15]

                position_var = np.var(history_array, axis=0)
                features[21:24] = position_var[9:12]

                features[24] = np.std(velocity)

                features[25] = np.mean(np.abs(velocity))
                features[26] = np.max(np.abs(velocity))
                features[27] = np.min(np.abs(velocity))
                features[28] = np.sum(np.abs(velocity))
                features[29] = len(history_array)

        except Exception as e:
            print(f"Warning: Temporal feature extraction failed: {e}")
            features = np.zeros(30)

        return features

    def extract_features(self, pose_landmarks):
        """
        Extract complete feature set (spatial + temporal).
        Always returns exactly the same number of features for consistency.
        """
        spatial_features = self.extract_spatial_features(pose_landmarks)

        self.pose_history.append(spatial_features)

        temporal_features = self.extract_temporal_features()

        combined_features = np.concatenate(
            [spatial_features, temporal_features])

        return combined_features

    def _calculate_angle(self, a, b, c):
        """
        Calculate angle at point b formed by points a-b-c.
        Returns angle in degrees.
        """
        # Vectors
        ba = a - b
        bc = c - b

        # Cosine of angle
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba)
                                   * np.linalg.norm(bc) + 1e-6)
        cosine = np.clip(cosine, -1.0, 1.0)

        # Angle in degrees
        angle = np.arccos(cosine) * 180.0 / np.pi

        return angle

    def get_feature_names(self):
        """
        Return feature names for interpretability.
        """
        names = []

        # Body-relative coordinates (13 keypoints * 3 coords)
        keypoint_names = ['nose', 'l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow',
                          'l_wrist', 'r_wrist', 'l_hip', 'r_hip', 'l_knee', 'r_knee',
                          'l_ankle', 'r_ankle']
        for kp in keypoint_names:
            names.extend([f'{kp}_x', f'{kp}_y', f'{kp}_z'])

        # Joint angles
        names.extend(['angle_r_elbow', 'angle_l_elbow',
                     'angle_r_knee', 'angle_l_knee', 'angle_torso'])

        # Relative distances
        names.extend(['dist_r_hand_hip', 'dist_l_hand_hip', 'hand_r_height', 'hand_l_height',
                     'dist_r_hand_torso', 'dist_l_hand_torso'])

        # Body configuration
        names.extend(['shoulder_width', 'hip_width', 'arm_symmetry',
                     'head_tilt_x', 'head_tilt_y', 'torso_length'])

        # Temporal features
        names.extend(['vel_r_hand_x', 'vel_r_hand_y', 'vel_r_hand_z',
                     'vel_l_hand_x', 'vel_l_hand_y', 'vel_l_hand_z'])
        names.extend(['acc_r_hand_x', 'acc_r_hand_y', 'acc_r_hand_z',
                     'acc_l_hand_x', 'acc_l_hand_y', 'acc_l_hand_z'])
        names.extend(['motion_magnitude', 'hand_motion_r', 'hand_motion_l', 'jerk_magnitude',
                     'motion_std', 'direction_consistency'])
        names.extend(['avg_r_hand_x', 'avg_r_hand_y', 'avg_r_hand_z',
                     'avg_l_hand_x', 'avg_l_hand_y', 'avg_l_hand_z'])
        names.extend(['var_r_hand_x', 'var_r_hand_y', 'var_r_hand_z'])

        return names


def augment_pose_data(landmarks_array, flip=False, noise_level=0.01, scale_factor=1.0):
    """
    Augment pose data for better generalization.

    Args:
        landmarks_array: numpy array of shape (33, 3) with x,y,z coordinates
        flip: whether to flip horizontally (mirror)
        noise_level: standard deviation of Gaussian noise to add
        scale_factor: scale the pose (simulate different distances)

    Returns:
        Augmented landmarks array
    """
    augmented = landmarks_array.copy()

    # Horizontal flip
    if flip:
        augmented[:, 0] = 1.0 - augmented[:, 0]
        # Swap left/right landmarks
        swap_pairs = [
            (1, 2), (3, 4), (5, 6), (7, 8),  # Face
            (9, 10),  # Mouth
            (11, 12), (13, 14), (15, 16),  # Arms
            (17, 18), (19, 20), (21, 22),  # Hands
            (23, 24), (25, 26), (27, 28),  # Legs
            (29, 30), (31, 32)  # Feet
        ]
        for left, right in swap_pairs:
            augmented[[left, right]] = augmented[[right, left]]

    # Add Gaussian noise
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, augmented.shape)
        augmented += noise
        # Keep in valid range [0, 1] for x and y
        augmented[:, :2] = np.clip(augmented[:, :2], 0, 1)

    # Scale (simulate different distances from camera)
    if scale_factor != 1.0:
        center = augmented.mean(axis=0)
        augmented = center + (augmented - center) * scale_factor

    return augmented
