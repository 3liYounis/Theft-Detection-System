from Detection.yolo_pose_detector import detect_theft_yolo_pose, detect_theft_yolo_pose_realtime
from Recognition.recognize_face import get_face_info, recognize_face
from Alert.alert import alert
from datetime import datetime


def run_yolo_pose_pipeline(video_path=None):
    """
    Run YOLO Pose-based theft detection pipeline.

    Modes:
    - If video_path is None → realtime detection
    - If video_path is given → analyze video file
    """
    theft_video_path = video_path

    print("\n" + "=" * 70)
    print("OFFLINE YOLO POSE THEFT DETECTION PIPELINE")
    print("=" * 70)
    print(f"Video: {theft_video_path}")
    print(f"Model: YOLO Pose Estimation + LSTM")
    print("=" * 70 + "\n")

    print("\n[1/3] Running YOLO Pose-based behavior detection...\n")
    if theft_video_path:
        ml_info = detect_theft_yolo_pose(
            video_path=theft_video_path,
            model_path='Behaveioral/theft_detector_lstm.keras',
            scaler_path='Behaveioral/scaler_lstm.pkl',
            sequence_length=30,
            visualize=True,
            save_path='output_yolo_pose.mp4',
            skip_frames=2,
        )
    else:
        ml_info = detect_theft_yolo_pose_realtime(
            model_path='Behaveioral/theft_detector_lstm.keras',
            scaler_path='Behaveioral/scaler_lstm.pkl',
            sequence_length=30,
            visualize=True,
            save_path='output_yolo_pose.mp4',
            skip_frames=2,
        )
    theft_decision = ml_info.get("theft")
    print("\n" + "=" * 70)
    print("BEHAVIOR DETECTION RESULTS")
    print("=" * 70)

    for key, value in ml_info.items():
        print(f"{key}: {value}")
    print("=" * 70)

    if not theft_decision:
        print("\n✓ No theft detected.")
        return

    print("\n⚠ THEFT DETECTED! Proceeding with identification...\n")

    print("[2/3] Running face recognition...")
    face_info = recognize_face(ml_info['video_path'])
    face_info = get_face_info(face_info['match'])
    print(f"Face info: {face_info}")

    print("\n[3/3] Sending alert...")
    time_str = datetime.now().strftime("%A, %Y-%m-%d %H:%M:%S")

    alert(
        theft_link=theft_video_path,
        face=face_info,
        time=time_str,
    )

    print("\n" + "=" * 70)
    print("OFFLINE YOLO POSE PIPELINE COMPLETE")
    print("=" * 70)
    print("✓ Theft detected and reported")
    print(f"✓ Video: {theft_video_path}")
    print(f"✓ Person: {face_info}")
    print(f"✓ Time: {time_str}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    theft_video_path = "../Data/Stream/Shoplifting - Test/Shoplifting (93).mp4"
    # theft_video_path = "../Data/Stream/Shoplifting/Shoplifting (85).mp4"
    # theft_video_path = "../Data/Stream/Normal - Test/Normal (91).mp4"
    # theft_video_path = "../Data/Stream/Random/ali_hair.mp4"
    # theft_video_path = "../Data/Stream/Random/yazan.mp4"
    # theft_video_path = "../Data/Stream/Random/hamza.mp4"
    run_yolo_pose_pipeline(theft_video_path)
