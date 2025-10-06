from Input.input_stream import input_stream, get_stream
from Behaveioral.detect_theft import detect_theft
from Behaveioral.behavior_classifier_infer import detect_theft_ml
from Recognition.recognize_face import recognize_face
from Detection.detect_item import detect_items_from_video
from Alert.alert import alert
from datetime import datetime


def run_pipeline():
    # Ensure we have a fresh recording
    # input_stream()  # optional: uncomment to record new video
    stream = get_stream()

    # We'll just use the file path used by get_stream
    theft_video_path = "./Data/Stream/Shoplifting/Shoplifting (1).mp4"
    # theft_video_path = "./Data/Stream/Normal/Normal (70).mp4"

    # 1) ML-based decision (pose classifier)
    ml_info = detect_theft_ml(
        theft_video_path, "pose_behavior_classifier.joblib", window_size=90, positive_ratio=0.6)
    # 2) FSM-based decision (pose + product overlap)
    # fsm_info = detect_theft(theft_video_path)

    theft_decision = ml_info.get("theft")
    if not theft_decision:
        # print("No theft detected. ML:", ml_info, "FSM:", fsm_info)
        print("No theft detected. ML:", ml_info)
        return

    items = detect_items_from_video(theft_video_path)
    face_info = recognize_face(theft_video_path)

    time_str = datetime.now().strftime("%A, %Y-%m-%d %H:%M:%S")
    alert(
        theft_link=theft_video_path,
        item=items,
        face=face_info,
        time=time_str,
    )


if __name__ == "__main__":
    run_pipeline()
