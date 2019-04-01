import cv2
import numpy as np
from Common_Tools.window import Window


class DrawLane:

    def __init__(self, right_line, left_line):
        self.right_line = right_line
        self.left_line = left_line
        self.left_line_allx = None
        self.left_line_ally = None
        self.old_result = None
        self.left_test = None
        self.right_test = None

    def find_lanes(self, img):
        # Search
        left_lane_inds, right_lane_points, out_img = self.window_search(img)
        # Update
        self.validate_lane(img, left_lane_inds, right_lane_points)
        return out_img

    def window_search(self, binary_warped):

        histogram = np.sum(binary_warped[int(binary_warped.shape[0] / 2):, :], axis=0)
        # Output image
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        left_window_array = []
        right_window_array = []
        im_shape = binary_warped.shape

        margin = 30

        # Left windows
        if len(left_window_array) == 0:
            for win in range(8):
                if win == 0:
                    left_window_array.append(
                        Window(im_shape, leftx_base, win, margin=margin))
                else:
                    left_window_array.append(
                        Window(im_shape, left_window_array[win-1].get_x(), win, margin=margin))
        # Right windows
        if len(right_window_array) == 0:
            for win in range(8):
                if win == 0:
                    right_window_array.append(
                        Window(im_shape, rightx_base, win, margin=margin))
                else:
                    right_window_array.append(
                        Window(im_shape, right_window_array[win-1].get_x(), win, margin=margin))

        # Non zero pix locations
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_lane_inds = []
        right_lane_points = []

        left_lane_pts = []
        right_lane_pts = []

        for win in left_window_array:
            good_inds = win.update(nonzerox, nonzeroy)
            left_lane_inds.append(good_inds)
            x, y = win.get_win_pos()
            left_lane_pts.append([x, y])

        for win in right_window_array:
            good_inds = win.update(nonzerox, nonzeroy)
            right_lane_points.append(good_inds)
            x, y = win.get_win_pos()
            right_lane_pts.append((x, y))

        leftx = [x for x, y in left_lane_pts]
        lefty = [y for x, y in left_lane_pts]
        rightx = [x for x, y in right_lane_pts]
        righty = [y for x, y in right_lane_pts]

        # Combine sub arrays
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_points = np.concatenate(right_lane_points)

        window_img = np.zeros_like(out_img)

        right_lane_pts = np.array(right_lane_pts)
        left_lane_pts = np.array(left_lane_pts)

        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.intc([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.intc([right_line_pts]), (0, 255, 0))
        out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [1, 0, 0]
        out_img[nonzeroy[right_lane_points], nonzerox[right_lane_points]] = [0, 0, 1]

        # Draw polyline on image
        cv2.polylines(out_img, [right_lane_pts], False, (1, 1, 0), thickness=5)
        cv2.polylines(out_img, [left_lane_pts], False, (1, 1, 0), thickness=5)

        return left_lane_inds, right_lane_points, out_img

    def validate_lane(self, img, left_lane_inds, right_lane_points):

        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Extract left and right line pixel positions
        self.left_line_allx = nonzerox[left_lane_inds]
        self.left_line_ally = nonzeroy[left_lane_inds]
        self.right_line_allx = nonzerox[right_lane_points]
        self.right_line_ally = nonzeroy[right_lane_points]

        # Discard lane detections that have very little points,
        # as they tend to have unstable results in most cases
        if len(self.left_line_allx) <= 400 or len(self.right_line_allx) <= 400:
            self.left_line.detected = False
            self.right_line.detected = False
            return

        left_x_mean = np.mean(self.left_line_allx, axis=0)
        right_x_mean = np.mean(self.right_line_allx, axis=0)
        lane_width = np.subtract(right_x_mean, left_x_mean)

        # Discard the detections if lanes are not in their respective half of their screens
        if left_x_mean > 450 or right_x_mean < 350:
            self.left_line.detected = False
            self.right_line.detected = False
            return

        # Discard the detections if the lane width is too large or too small
        if lane_width < 200 or lane_width > 400:
            self.left_line.detected = False
            self.right_line.detected = False
            return

            # If this is the first detection or
        # the detection is within the margin of the averaged n last lines
        if self.left_line.bestx is None or np.abs(np.subtract(self.left_line.bestx, np.mean(self.left_line_allx, axis=0))) < 100:
            self.left_line.update_lane(self.left_line_ally, self.left_line_allx)
            self.left_line.detected = True
        else:
            self.left_line.detected = False
        if self.right_line.bestx is None or np.abs(np.subtract(self.right_line.bestx, np.mean(self.right_line_allx, axis=0))) < 100:
            self.right_line.update_lane(self.right_line_ally, self.right_line_allx)
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