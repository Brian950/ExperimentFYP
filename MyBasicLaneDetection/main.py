import cv2
from MyBasicLaneDetection import process_image

cap = cv2.VideoCapture("C:\\Users\\Brian\\Desktop\\test_videos\\highway1.mp4")
if not cap.isOpened:
    print("Error opening video stream")
    exit(1)

while True:

    ret, frame = cap.read()
    if ret:
        processed_view = process_image.process(frame)
        cv2.imshow('ImageStream', processed_view)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    else:
        cap = cv2.VideoCapture("C:\\Users\\Brian\\Desktop\\test_videos\\highway1.mp4")
