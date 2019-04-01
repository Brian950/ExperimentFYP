import cv2
import numpy as np
import math
from Common_Tools.draw_lane import DrawLane
from Common_Tools.image_manipulation import Manipulation
from Common_Tools.line import Line
from skimage.filters import threshold_sauvola

right_line = Line()
left_line = Line()
draw_lanes = DrawLane(right_line, left_line)

sharp_kernel = np.array([[-1, -1, -1],
                         [-1, 9, -1],
                         [-1, -1, -1]])

GRAY_MEAN_THRESH = 100
FRAME_BUFFER_LIMIT = 3
FRAME_BUFFER_COUNT = 0
BUFFERED_FRAMES = []


def process(frame, selection):
    image = crop(frame, 0, 360, 0, 854)

    man = Manipulation(selection)
    src, dst = man.get_perspective_matrix()
    transform = man.perspective_transform(image, src, dst, (854, 360))

    # Separate the V channel from the HSV image
    v = hsv(transform)
    hsv_white = set_gray_binary(v)
    hsv_white = cv2.GaussianBlur(hsv_white, (3, 3), 0)
    hsv_white = cv2.morphologyEx(hsv_white, cv2.MORPH_CLOSE, (5, 5))

    # Grayscale image
    gray = cv2.cvtColor(transform, cv2.COLOR_RGB2GRAY)
    gray_mean_val = np.mean(gray)

    # Pixel gray value
    if gray_mean_val > GRAY_MEAN_THRESH:
        # High light intensity = gray equalisation
        gray = gray_equalisation(gray)

    gray = sharpen(gray)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # Use adaptive threshold
    adap_thresh = cv2.adaptiveThreshold(gray, 225, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 61, -21)

    # First edge detection on whole frame
    full_canny = canny(transform, 200, 200)

    # Global gradient
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # Gray binary thresh
    gray_binary = set_gray_binary(gray)
    white_binary = cv2.bitwise_or(hsv_white, gray_binary)

    opening = cv2.morphologyEx(laplacian, cv2.MORPH_OPEN, (5, 5), iterations=1)
    laplacian_smooth = cv2.GaussianBlur(opening, (9, 9), 0)
    laplacian_smooth = laplacian_smooth.astype('uint8')

    if gray_mean_val > 50:
        combo = cv2.bitwise_and(white_binary, laplacian_smooth)
        combo = cv2.bitwise_or(combo, full_canny)
        combo = cv2.bitwise_or(combo, adap_thresh)
    else:
        combo = cv2.bitwise_or(white_binary, laplacian_smooth)
        combo = cv2.bitwise_or(combo, full_canny)
        combo = cv2.bitwise_and(combo, adap_thresh)

    combo = get_frame_blend(combo)
    combo = set_roi(combo)

    # Image halves
    left_side = crop(combo, 0, 360, 100, 500)
    right_side = crop(combo, 0, 360, 400, 800)

    # Uses the left & right sides to draw on adap_thresh
    get_hough_lines(left_side, combo, left=True)
    get_hough_lines(right_side, combo, right=True)

    output_img = draw_lanes.find_lanes(combo)
    trans_filler = np.zeros_like(frame)
    res = draw_lanes.draw_lane(image, trans_filler, output_img, src, dst)

    result = assemble_img(transform, output_img, res, combo,
                          cv2.cvtColor(adap_thresh, cv2.COLOR_GRAY2RGB),
                          cv2.cvtColor(laplacian_smooth, cv2.COLOR_GRAY2RGB))

    cv2.imwrite("C:\\Users\\Brian\\Desktop\\test_videos\\image.jpg", res)

    return result


# Combine previous frames to handle broken line markers
def get_frame_blend(combo):
    global FRAME_BUFFER_COUNT, FRAME_BUFFER_LIMIT

    if len(BUFFERED_FRAMES) < FRAME_BUFFER_LIMIT:
        BUFFERED_FRAMES.append(combo)
    else:
        BUFFERED_FRAMES[FRAME_BUFFER_COUNT] = combo
        FRAME_BUFFER_COUNT += 1
        if FRAME_BUFFER_COUNT == len(BUFFERED_FRAMES):
            FRAME_BUFFER_COUNT = 0

    result = None
    if len(BUFFERED_FRAMES) > 1:
        count = 0
        for frame in BUFFERED_FRAMES:
            if count == 0:
                result = cv2.addWeighted(combo, 1, frame, 1, 0)
                count = 1
            else:
                result = cv2.addWeighted(result, 1, frame, 1, 0)
    else:
        return combo

    return result


def get_hough_lines(split_image, whole_image, left=False, right=False):

    if not left and not right:
        return

    lines = cv2.HoughLinesP(split_image, rho=1, theta=np.pi / 120,
                            threshold=50, minLineLength=100, maxLineGap=200)
    # Two lines stored to try and accommodate for neighbouring lane issues
    # Line lengths
    p_len_1 = 0
    p_len_2 = 0
    n_len_1 = 0
    n_len_2 = 0
    # Lines
    p_line_1 = None
    p_line_2 = None
    n_line_1 = None
    n_line_2 = None
    try:
        # TO-DO tidy up conditional logic here. Possibly move to class.
        for line in lines:
            for x1, y1, x2, y2 in line:
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                # Validate both lines for the left side of the screen
                if left:
                    if -45 > angle > -91 or 91 > angle > 70:
                        length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                        if length > n_len_1:
                            if n_line_2 is None:
                                n_len_1 = length
                                n_line_1 = line
                            else:
                                # Check if line is to close to existing line
                                lx1, ly1, lx2, ly2 = n_line_2[0]
                                if x1 > lx1 + 75 or x1 < lx1 - 75:
                                    n_len_2 = length
                                    n_line_2 = line
                        elif length > n_len_2:
                            # Check if line is to close to existing line
                            lx1, ly1, lx2, ly2 = n_line_1[0]
                            if x1 > lx1+75 or x1 < lx1-75:
                                n_len_2 = length
                                n_line_2 = line

                elif right:
                    if 45 < angle < 91 or -70 > angle > -91:
                        length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                        if length > p_len_1:
                            if p_line_2 is None:
                                p_len_1 = length
                                p_line_1 = line
                            else:
                                # Check if line is to close to existing line
                                lx1, ly1, lx2, ly2 = p_line_2[0]
                                if x1 > lx1 + 75 or x1 < lx1 - 75:
                                    p_len_1 = length
                                    p_line_1 = line
                        elif length > p_len_2:
                            # Check if line is to close to existing line
                            lx1, ly1, lx2, ly2 = p_line_1[0]
                            if x1 > lx1 + 75 or x1 < lx1 - 75:
                                p_len_2 = length
                                p_line_2 = line
                else:
                    return

        if p_len_1 != 0:
            # Positive Line
            xp1, yp1, xp2, yp2 = p_line_1[0]
            cv2.line(whole_image, (xp1+400, yp1), (xp2+400, yp2), (255, 255, 255), 10)
        if p_len_2 != 0:
            # Positive Line
            xp1, yp1, xp2, yp2 = p_line_2[0]
            cv2.line(whole_image, (xp1 + 400, yp1), (xp2 + 400, yp2), (255, 255, 255), 10)
        if n_len_1 != 0:
            # Negative Line
            xn1, yn1, xn2, yn2 = n_line_1[0]
            cv2.line(whole_image, (xn1+100, yn1), (xn2+100, yn2), (255, 255, 255), 10)
        if n_len_2 != 0:
            # Negative Line
            xn1, yn1, xn2, yn2 = n_line_2[0]
            cv2.line(whole_image, (xn1+100, yn1), (xn2+100, yn2), (255, 255, 255), 10)

    except Exception as e:
        print(e)


def crop(image, ht, hb, wt, wb):
    image = image[ht:hb, wt:wb]
    return image


def set_roi(img):
    imshape = img.shape
    lower_left = [0, imshape[0]]
    lower_right = [imshape[1], imshape[0]]
    top_left = [imshape[1] / 2 - imshape[1] / 3, 0]
    top_right = [imshape[1] / 2 + imshape[1] / 3, 0]
    vertices = [np.array([lower_left, top_left, top_right, lower_right], dtype=np.int32)]
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)

    return masked_image


def sharpen(image):
    image = cv2.filter2D(image, -1, sharp_kernel)
    image = cv2.bilateralFilter(image, 5, 50, 50)
    return image


def set_gray_binary(image):
    ret, thresh = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
    return thresh


def hsv(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    result = hsv_image[:, :, 2]
    return result


def canny(image, min_val, max_val):
    result = cv2.Canny(image, min_val, max_val)
    return result


def gray_equalisation(image):
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(6, 6))
    image = clahe.apply(image)
    return image


def assemble_img(warped, polynomial_img, lane_img, combo_image, adap, lapla):
    # Output image
    img_out = np.zeros((740, 1290, 3), dtype=np.uint8)
    if lane_img is not None:
        img_out[0:360, 0:854, :] = lane_img

    combo_image = cv2.cvtColor(combo_image, cv2.COLOR_GRAY2RGB)
    font_size = 1
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Perspective transform
    img_out[0:240, 865:1285, :] = cv2.resize(warped, (420, 240))
    boxsize, _ = cv2.getTextSize("Transformed", font, font_size, thickness)
    cv2.putText(img_out, "Transformed", (int(1090 - boxsize[0] / 2), 40), font, font_size, (50, 200, 255),
                thickness, lineType=cv2.LINE_AA)

    # Threshold
    img_out[250:490, 865:1285, :] = cv2.resize(combo_image, (420, 240))
    boxsize, _ = cv2.getTextSize("Threshold", font, font_size, thickness)
    cv2.putText(img_out, "Threshold", (int(1090 - boxsize[0] / 2), 280), font, font_size, (200, 50, 255),
                thickness, lineType=cv2.LINE_AA)

    # Polynomial lines
    img_out[500: 740, 865:1285, :] = cv2.resize(polynomial_img * 255, (420, 240))
    boxsize, _ = cv2.getTextSize("Detected Lane", font, font_size, thickness)
    cv2.putText(img_out, "Detected Lane", (int(1090 - boxsize[0] / 2), 520), font, font_size, (255, 255, 255),
                thickness, lineType=cv2.LINE_AA)

    # Gray EQ
    img_out[500: 740, 440:860, :] = cv2.resize(adap, (420, 240))
    boxsize, _ = cv2.getTextSize("Adaptive Thresh", font, font_size, thickness)
    cv2.putText(img_out, "Adaptive Thresh", (int(650 - boxsize[0] / 2), 530), font, font_size, (200, 255, 50),
                thickness, lineType=cv2.LINE_AA)

    # Canny Edge
    img_out[500: 740, 10:430, :] = cv2.resize(lapla, (420, 240))
    boxsize, _ = cv2.getTextSize("Laplacian Thresh", font, font_size, thickness)
    cv2.putText(img_out, "Laplacian Thresh", (int(230 - boxsize[0] / 2), 530), font, font_size, (50, 255, 200),
                thickness, lineType=cv2.LINE_AA)

    return img_out
