from recognize_face import *

recording_path = "./Data/Stream/Random/ali_hair.mp4"
info = get_face_info(recognize_face(recording_path)['match'])
print(info)