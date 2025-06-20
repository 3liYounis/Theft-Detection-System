from detect_theft import detect_theft
normal_path = "./Data/Stream/Normal/Normal (85).mp4"
theft_path = "./Data/Stream/Shoplifting/Shoplifting (85).mp4"
info = detect_theft(normal_path)
print(info)
