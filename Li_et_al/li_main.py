import cv2
import time
from Li_et_al import process_image
import tkinter as tk
from threading import Thread


VIDEO_OPTIONS = ["highway.mp4", "highway_sunlight.mp4",
                 "highway_night.mp4", "night_traffic.mp4",
                 "night_with_bend.mp4", "shadows_and_road_markings.mp4",
                 "shadows_and_traffic.mp4"]
cap = None
fps_count = 0


def main():
    setup_gui()


def setup_gui():
    global video_string
    gui = tk.Tk()
    gui.title("Lane Detection Menu")

    # List of test video options
    video_string = tk.StringVar(gui)
    video_string.set(VIDEO_OPTIONS[0])
    options = tk.OptionMenu(gui, video_string, *VIDEO_OPTIONS)
    options.pack()

    button = tk.Button(gui, text="Select", command=setup_video_thread)
    button.pack()

    tk.mainloop()


def setup_video_thread():
    global cap
    try:
        if cap is None:
            video_thread = Thread(target=setup_video)
            video_thread.start()
        else:
            selection = video_string.get()
            cap = cv2.VideoCapture("C:\\Users\\Brian\\Desktop\\test_videos\\BDD\\"+selection)
    except Exception as e:
        print(e)


def setup_video():
    global fps_count
    global cap

    selection = video_string.get()
    cap = cv2.VideoCapture("C:\\Users\\Brian\\Desktop\\test_videos\\BDD\\"+selection)

    if not cap.isOpened:
        print("Error opening video stream")
        exit(1)

    while True:
        start_time = time.time()
        selection = video_string.get()
        fps_count += 1

        ret, frame = cap.read()
        if ret:
            processed_view = process_image.process(frame, selection=selection)
            cv2.imshow('ImageStream', processed_view)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        else:
            cap = cv2.VideoCapture("C:\\Users\\Brian\\Desktop\\test_videos\\BDD\\"+selection)

        # if fps_count == 300:
        #    cv2.imwrite("C:\\Users\\Brian\\Desktop\\test_videos\\project_images\\hist_eq_original.jpg", processed_view)
        print("FPS: ", 1.0 / (time.time() - start_time))


if __name__ == "__main__": main()
