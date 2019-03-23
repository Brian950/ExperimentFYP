import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt

class DrawLane:

    def __init__(self, right_line, left_line):
        self.right_line = right_line
        self.left_line = left_line
        self.old_result = None

    def find_lanes(self, img):
        if self.left_line.detected and self.right_line.detected:  # Perform margin search if exists prior success.
            # Margin Search
            left_lane_inds, right_lane_inds, out_img = self.margin_search(img)
            # Update the lane detections
            self.validate_lane(img, left_lane_inds, right_lane_inds)
        else:  # Perform a full window search if no prior successful detections.
            # Window Search
            left_lane_inds, right_lane_inds, out_img = self.window_search(img)
            # Update the lane detections
            self.validate_lane(img, left_lane_inds, right_lane_inds)
        return out_img

    def window_search(self, binary_warped):

        histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 8
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)  # 360/8 = 45
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 50
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            #cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (255, 255, 255), 2)
            #cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (255, 255, 255), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                        nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                        nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        try:
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)

            # Generate x and y values for plotting
            ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

            # Generate black image and colour lane lines
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [0, 1, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 1, 0]

            # Draw polyline on image
            right = np.asarray(tuple(zip(right_fitx, ploty)), np.int32)
            left = np.asarray(tuple(zip(left_fitx, ploty)), np.int32)
            cv2.polylines(out_img, [right], False, (1, 0, 1), thickness=5)
            cv2.polylines(out_img, [left], False, (1, 0, 1), thickness=5)
        except:
            pass

        return left_lane_inds, right_lane_inds, out_img

    def margin_search(self, binary_warped):
        left_line = self.left_line
        right_line = self.right_line
        # Performs window search on subsequent frame, given previous frame.
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 30

        left_lane_inds = ((nonzerox > (
                    left_line.current_fit[0] * (nonzeroy ** 2) + left_line.current_fit[1] * nonzeroy +
                    left_line.current_fit[2] - margin)) & (nonzerox < (
                    left_line.current_fit[0] * (nonzeroy ** 2) + left_line.current_fit[1] * nonzeroy +
                    left_line.current_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (
                    right_line.current_fit[0] * (nonzeroy ** 2) + right_line.current_fit[1] * nonzeroy +
                    right_line.current_fit[2] - margin)) & (nonzerox < (
                    right_line.current_fit[0] * (nonzeroy ** 2) + right_line.current_fit[1] * nonzeroy +
                    right_line.current_fit[2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        try:
            # Fit a second order polynomial to each
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
        except TypeError as te:
            return self.window_search(binary_warped)

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # Generate a blank image to draw on
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

        # Create an image to draw on and an image to show the selection window
        window_img = np.zeros_like(out_img)

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
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
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 1]

        # Draw polyline on image
        right = np.asarray(tuple(zip(right_fitx, ploty)), np.int32)
        left = np.asarray(tuple(zip(left_fitx, ploty)), np.int32)
        cv2.polylines(out_img, [right], False, (1, 1, 0), thickness=5)
        cv2.polylines(out_img, [left], False, (1, 1, 0), thickness=5)

        return left_lane_inds, right_lane_inds, out_img

    def validate_lane(self, img, left_lane_inds, right_lane_inds):

        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Extract left and right line pixel positions
        self.left_line_allx = nonzerox[left_lane_inds]
        self.left_line_ally = nonzeroy[left_lane_inds]
        self.right_line_allx = nonzerox[right_lane_inds]
        self.right_line_ally = nonzeroy[right_lane_inds]

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
        # Generate x and y values for plotting (numeric sequence)
        ploty = np.linspace(0, undist.shape[0] - 1, undist.shape[0])

        left_fit = self.left_line.best_fit
        right_fit = self.right_line.best_fit

        if left_fit is not None and right_fit is not None:
            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]

            # Recast the x and y points into usable format for cv2.fillPoly()
            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))

            # Draw the lane
            cv2.fillPoly(undist, np.int_([pts]), (125, 245, 255))

            # Warp the blank back to original image space using inverse perspective matrix (Minv)
            newwarp = cv2.warpPerspective(undist, matrix, (img.shape[1], img.shape[0]))

            # Combine the result with the original image
            result = cv2.addWeighted(view, 1, newwarp, 0.6, 0)
            self.old_result = result
            return result
        return self.old_result

    def assemble_img(self, warped, polynomial_img, lane_img, combo_image):
        # Define output image
        img_out = np.zeros((740, 1290, 3), dtype=np.uint8)
        if lane_img is not None:
            img_out[0:360, 0:854, :] = lane_img

        combo_image = cv2.cvtColor(combo_image, cv2.COLOR_GRAY2RGB)
        # Text format
        fontScale = 1
        thickness = 1
        fontFace = cv2.FONT_HERSHEY_SIMPLEX

        # Perspective transform
        img_out[0:240, 865:1285, :] = cv2.resize(warped, (420, 240))
        boxsize, _ = cv2.getTextSize("Transformed", fontFace, fontScale, thickness)
        cv2.putText(img_out, "Transformed", (int(1090 - boxsize[0] / 2), 40), fontFace, fontScale, (255, 255, 255),
                    thickness, lineType=cv2.LINE_AA)

        #
        img_out[250:490, 865:1285, :] = cv2.resize(combo_image, (420, 240))
        boxsize, _ = cv2.getTextSize("Threshold", fontFace, fontScale, thickness)
        cv2.putText(img_out, "Threshold", (int(1080 - boxsize[0] / 2), 280), fontFace, fontScale, (255, 255, 255),
                    thickness, lineType=cv2.LINE_AA)

        # Polynomial lines
        img_out[500: 740, 865:1285, :] = cv2.resize(polynomial_img * 255, (420, 240))
        boxsize, _ = cv2.getTextSize("Detected Lane", fontFace, fontScale, thickness)
        cv2.putText(img_out, "Detected Lane", (int(1090 - boxsize[0] / 2), 520), fontFace, fontScale, (255, 255, 255),
                    thickness, lineType=cv2.LINE_AA)

        return img_out