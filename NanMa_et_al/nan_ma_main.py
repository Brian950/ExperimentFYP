import cv2
import time
from NanMa_et_al import process_image

CALIBRATE = True
fps_count = 0

calibration_dir = "C:\\Users\\Brian\\Desktop\\test_videos\\calibration\\"

cap = cv2.VideoCapture("C:\\Users\\Brian\\Desktop\\test_videos\\BDD\\highway.mp4")
if not cap.isOpened:
    print("Error opening video stream")
    exit(1)

mtx, dist = process_image.calibrate_camera(calibration_dir, 9, 6, (720, 1280))
while True:
    start_time = time.time()

    fps_count += 1

    ret, frame = cap.read()
    if ret:
        if CALIBRATE:
            processed_view = process_image.process(frame, mtx, dist)
        else:
            processed_view = process_image.process(frame)
        cv2.imshow('ImageStream', processed_view)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    else:
        cap = cv2.VideoCapture("C:\\Users\\Brian\\Desktop\\test_videos\\BDD\\highway.mp4")

    #if fps_count == 300:
    #    cv2.imwrite("C:\\Users\\Brian\\Desktop\\test_videos\\project_images\\hist_eq_original.jpg", processed_view)
    print("FPS: ", 1.0 / (time.time() - start_time))