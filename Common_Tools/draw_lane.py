import cv2
import numpy as np


class DrawLane:

    def __init__(self, right_line, left_line):
        self.right_line = right_line
        self.left_line = left_line
        self.old_result = None

    def find_lanes(self, img):
        if self.left_line.detected and self.right_line.detected:
            left_lane_points, right_lane_points, out_img = self.margin_search(img)
            self.validate_lane(img, left_lane_points, right_lane_points)
        else:
            left_lane_points, right_lane_points, out_img = self.window_search(img)
            self.validate_lane(img, left_lane_points, right_lane_points)
        return out_img

    def window_search(self, binary_warped):

        histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

        # Start points for the left and right lanes
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Sliding windows
        nwindows = 8

        window_height = np.int(binary_warped.shape[0]/nwindows)  # 360/8 = 45

        # Non zero pixel locations
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base

        # Search margin for each window
        margin = 50
        # Nonzero pixels to
        minpix = 50

        left_lane_points = []
        right_lane_points = []

        for window in range(nwindows):
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # nonzero pix
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                        nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                        nonzerox < win_xright_high)).nonzero()[0]

            left_lane_points.append(good_left_inds)
            right_lane_points.append(good_right_inds)

            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Group sub arrays
        left_lane_points = np.concatenate(left_lane_points)
        right_lane_points = np.concatenate(right_lane_points)

        # Left and right lane pixels
        leftx = nonzerox[left_lane_points]
        lefty = nonzeroy[left_lane_points]
        rightx = nonzerox[right_lane_points]
        righty = nonzeroy[right_lane_points]

        try:
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)

            ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

            # Color pixels red and blue
            out_img[nonzeroy[left_lane_points], nonzerox[left_lane_points]] = [1, 0, 0]
            out_img[nonzeroy[right_lane_points], nonzerox[right_lane_points]] = [0, 0, 1]

            # Draw lane lines
            right = np.asarray(tuple(zip(right_fitx, ploty)), np.int32)
            left = np.asarray(tuple(zip(left_fitx, ploty)), np.int32)
            cv2.polylines(out_img, [right], False, (1, 0, 1), thickness=5)
            cv2.polylines(out_img, [left], False, (1, 0, 1), thickness=5)
        except:
            pass

        return left_lane_points, right_lane_points, out_img

    def margin_search(self, binary_warped):
        left_line = self.left_line
        right_line = self.right_line

        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 30

        left_lane_points = ((nonzerox > (
                    left_line.current_fit[0] * (nonzeroy ** 2) + left_line.current_fit[1] * nonzeroy +
                    left_line.current_fit[2] - margin)) & (nonzerox < (
                    left_line.current_fit[0] * (nonzeroy ** 2) + left_line.current_fit[1] * nonzeroy +
                    left_line.current_fit[2] + margin)))
        right_lane_points = ((nonzerox > (
                    right_line.current_fit[0] * (nonzeroy ** 2) + right_line.current_fit[1] * nonzeroy +
                    right_line.current_fit[2] - margin)) & (nonzerox < (
                    right_line.current_fit[0] * (nonzeroy ** 2) + right_line.current_fit[1] * nonzeroy +
                    right_line.current_fit[2] + margin)))

        leftx = nonzerox[left_lane_points]
        lefty = nonzeroy[left_lane_points]
        rightx = nonzerox[right_lane_points]
        righty = nonzeroy[right_lane_points]

        try:
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
        except TypeError as te:
            return self.window_search(binary_warped)

        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # Blank image
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        window_img = np.zeros_like(out_img)

        # Gathers points into usable format for fillPoly
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draws the rectangles to highlight the search window area
        cv2.fillPoly(window_img, np.intc([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.intc([right_line_pts]), (0, 255, 0))
        out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_points], nonzerox[left_lane_points]] = [1, 0, 0]
        out_img[nonzeroy[right_lane_points], nonzerox[right_lane_points]] = [0, 0, 1]

        right = np.asarray(tuple(zip(right_fitx, ploty)), np.int32)
        left = np.asarray(tuple(zip(left_fitx, ploty)), np.int32)
        cv2.polylines(out_img, [right], False, (1, 1, 0), thickness=5)
        cv2.polylines(out_img, [left], False, (1, 1, 0), thickness=5)

        return left_lane_points, right_lane_points, out_img

    def validate_lane(self, img, left_lane_points, right_lane_points):

        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_line_allx = nonzerox[left_lane_points]
        left_line_ally = nonzeroy[left_lane_points]
        right_line_allx = nonzerox[right_lane_points]
        right_line_ally = nonzeroy[right_lane_points]

        # Discard lane detections that have very little points,
        # as they tend to have unstable results in most cases
        if len(left_line_allx) <= 400 or len(right_line_allx) <= 400:
            self.left_line.detected = False
            self.right_line.detected = False
            return

        left_x_mean = np.mean(left_line_allx, axis=0)
        right_x_mean = np.mean(right_line_allx, axis=0)
        lane_width = np.subtract(right_x_mean, left_x_mean)

        # Discard the detections if lanes are not in their respective half of their screens
        if left_x_mean > 450 or right_x_mean < 350:
            self.left_line.detected = False
            self.right_line.detected = False
            return

        # Discard the detections if the lane width is too large or too small
        if lane_width < 150 or lane_width > 400:
            self.left_line.detected = False
            self.right_line.detected = False
            return

        # If detection is within the margin the last n lines
        if self.left_line.bestx is None or np.abs(np.subtract(self.left_line.bestx, np.mean(left_line_allx, axis=0))) < 100:
            self.left_line.update_lane(left_line_ally, left_line_allx)
            self.left_line.detected = True
        else:
            self.left_line.detected = False
        if self.right_line.bestx is None or np.abs(np.subtract(self.right_line.bestx, np.mean(right_line_allx, axis=0))) < 100:
            self.right_line.update_lane(right_line_ally, right_line_allx)
            self.right_line.detected = True
        else:
            self.right_line.detected = False

    def draw_lane(self, view, undist, img, src, dst):

        matrix = cv2.getPerspectiveTransform(src, dst)
        ploty = np.linspace(0, undist.shape[0] - 1, undist.shape[0])

        left_fit = self.left_line.best_fit
        right_fit = self.right_line.best_fit

        if left_fit is not None and right_fit is not None:
            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]

            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))

            # Draw the lane
            cv2.fillPoly(undist, np.int_([pts]), (125, 245, 255))

            newwarp = cv2.warpPerspective(undist, matrix, (img.shape[1], img.shape[0]))

            # Combine original image with lane image
            result = cv2.addWeighted(view, 1, newwarp, 0.6, 0)
            self.old_result = result
            return result
        return self.old_result
