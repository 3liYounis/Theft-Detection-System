from Detection.yolo_pose_detector import detect_theft_yolo_pose_realtime
from Recognition.recognize_face import get_face_info, recognize_face
from Alert.alert import alert
from datetime import datetime
import threading
import queue

theft_queue = queue.Queue()


def handle_theft(ml_info):
    theft_video_path = ml_info['video_path']
    theft_decision = ml_info.get("theft")

    if not theft_decision:
        return

    print("\n⚠ THEFT DETECTED! Proceeding with identification...\n")

    face_info = recognize_face(theft_video_path)
    face_info = get_face_info(face_info['match'])

    time_str = datetime.now().strftime("%A, %Y-%m-%d %H:%M:%S")
    alert(theft_link=theft_video_path, face=face_info, time=time_str)


def run_realtime_detection():
    while True:
        ml_info = detect_theft_yolo_pose_realtime(
            model_path='Behaveioral/theft_detector_lstm.keras',
            scaler_path='Behaveioral/scaler_lstm.pkl',
            sequence_length=30,
            visualize=True,
            # save_path='output_yolo_pose.mp4',
            skip_frames=2,
        )
        if ml_info.get("theft"):
            threading.Thread(target=handle_theft, args=(
                ml_info,), daemon=True).start()


if __name__ == "__main__":
    run_realtime_detection()
