from Input.input_stream import input_stream, get_stream
from Behaveioral.detect_theft import detect_theft
from Recognition.recognize_face import recognize_face
from Detection.detect_item import detect_item
from Alert.alert import alert
# <------------------ In paralell Processing Is Required ------------------> #
# input_stream()
# stream = get_stream()
# <------------------ Current Implementation Is Sequential ------------------> #
# while stream.isOpened():
#     detection_result, theft_recording = detect_theft(stream)
#     if detection_result:
#         item = detect_item(theft_recording)
#         face = recognize_face(theft_recording)
#         time = "THU 1.5.2025 17:35"
#         alert(theft_recording, item, face, time)