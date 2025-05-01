import cv2 as cv
def input_stream():
    cam = cv.VideoCapture(0)
    frame_width = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter('./Data/Stream/stream.mp4', fourcc, 30.0, (frame_width, frame_height))

    while True:
        ret, frame = cam.read()
        out.write(frame)
        cv.imshow('Camera', frame)
        if cv.waitKey(1) == ord('q'):
            break
    cam.release()
    out.release()
    cv.destroyAllWindows()

def get_stream():
    file_path = "./Data/Stream/stream.mp4"
    cap = cv.VideoCapture(file_path)
    # while(cap.isOpened()):
    #     ret, frame = cap.read()
    #     cv.imshow('Frame', frame)
    #     if cv.waitKey(1) == ord('q'):
    #         break
    # cap.release()
    # cv.destroyAllWindows()
    return cap