from Input.input_stream import input_stream, get_stream
from Behaveioral.inference_lstm import detect_theft_lstm
from Recognition.recognize_face import get_face_info, recognize_face
from Alert.alert import alert
from datetime import datetime


def run_pipeline(video_path=None):
    """
    Run the complete theft detection pipeline.

    Args:
        video_path: Path to video file. If None, uses default test video.
    """
    # Ensure we have a fresh recording
    # input_stream()  # optional: uncomment to record new video
    # stream = get_stream()

    # Default test videos
    if video_path is None:
        theft_video_path = "../Data/Stream/Shoplifting - Test/Shoplifting (91).mp4"
        # theft_video_path = "../Data/Stream/Normal - Test/Normal (93).mp4"
        # theft_video_path = "../Data/Stream/Random/ali_hair.mp4"
        # theft_video_path = "../Data/Stream/Random/yazan.mp4"
        # theft_video_path = "../Data/Stream/Random/hamza.mp4"
        # theft_video_path = "../Data/Stream/Normal/Normal (66).mp4"
        # theft_video_path = "../Data/Stream/Normal/Normal (27).mp4"
        # theft_video_path = "../Data/Stream/Normal/Normal (87).mp4"
        # theft_video_path = "../Data/Stream/Shoplifting/Shoplifting (17).mp4"
    else:
        theft_video_path = video_path

    print("\n" + "=" * 70)
    print("THEFT DETECTION PIPELINE")
    print("=" * 70)
    print(f"Video: {theft_video_path}")
    print(
        f"Model: 'LSTM (Enhanced Features)'")
    print("=" * 70 + "\n")

    # 1) Behavior Detection
    lstm_model_path = 'Behaveioral/theft_detector_lstm.keras'
    lstm_scaler_path = 'Behaveioral/scaler_lstm.pkl'
    print("\n[1/3] Running LSTM-based behavior detection...")
    ml_info = detect_theft_lstm(
        video_path=theft_video_path,
        model_path=lstm_model_path,
        scaler_path=lstm_scaler_path,
        sequence_length=30,
        window_size=90,
        positive_ratio=0.6,
        visualize=True,
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

    print("\n⚠ THEFT DETECTED! Proceeding with identification...")

    # 2) Face Recognition
    print("\n[2/3] Running face recognition...")
    face_info = recognize_face(theft_video_path)
    face_info = get_face_info(face_info['match'])
    print(f"Face info: {face_info}")

    # 3) Send Alert
    print("\n[3/3] Sending alert >>>>>")
    time_str = datetime.now().strftime("%A, %Y-%m-%d %H:%M:%S")
    alert(
        theft_link=theft_video_path,
        face=face_info,
        time=time_str,
    )
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print("✓ Theft detected and reported")
    print(f"✓ Video: {theft_video_path}")
    print(f"✓ Person: {face_info}")
    print(f"✓ Time: {time_str}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_pipeline()
