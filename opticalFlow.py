# -*- coding: utf-8 -*-
"""
    Description:
        Own implemented optical flow algorithms with image pyramids.
    Author:
        Jingjin Li (20127165)
    Date:
        2022.05.04
"""
import cv2
import numpy as np


# Image Pyramids algorithm, adaptive for two-frame images, low index position corresponds large image.
def image_pyramids(img_old, img_new, levels):
    # Init necessary variables:
    old_pyramid, new_pyramid = [], []
    old_pyramid.append(img_old)
    new_pyramid.append(img_new)
    origin_old = img_old.copy()
    origin_new = img_new.copy()

    # Using gaussian blur to down-sample images in levels
    for level in range(levels - 1):
        # Invoke gaussian filter:
        image_blur_old = cv2.GaussianBlur(origin_old, (5, 5), 1)
        image_blur_new = cv2.GaussianBlur(origin_new, (5, 5), 1)

        # Resize the down-sampled images
        image_small_old = cv2.resize(
            image_blur_old, (image_blur_old.shape[1] // 2, image_blur_old.shape[0] // 2), interpolation=cv2.INTER_CUBIC)
        image_small_new = cv2.resize(
            image_blur_new, (image_blur_new.shape[1] // 2, image_blur_new.shape[0] // 2), interpolation=cv2.INTER_CUBIC)

        # Track the pyramid images
        old_pyramid.append(image_small_old)
        new_pyramid.append(image_small_new)

        # Update two-frame images
        origin_old = image_small_old
        origin_new = image_small_new

    return old_pyramid, new_pyramid


# Lucas-Kanade method to calculate optical flow, return the displaced point.
def lucas_kanade(img_old, img_new, points, window_size=5, pyramid_level=5, iters=10, residual=0.03):
    # Generate pyramid image lists of two-frame images
    old_pyramid, new_pyramid = image_pyramids(img_old, img_new, pyramid_level)

    # Init optical flow results, the same shape as the feature points to track.
    flow_results = np.zeros(points.shape, dtype=np.float64)

    # Iteratively calculate the optical flow according to the image pyramid.
    for level in range(pyramid_level - 1, -1, -1):
        # Obtain the corresponding image from the pyramid list (from small image to large one).
        img_old = old_pyramid[level].astype(np.float64)
        img_new = new_pyramid[level].astype(np.float64)

        # Init necessary lucas related matrices.
        M_left = np.zeros((2, 2))
        M_right = np.zeros((2,))

        # Transform the feature points corresponding to the level:
        current_points = np.round((points / (2 ** level))).astype(np.int16)

        # Calculate gradient x, gradient y using sobel filter:
        I_x = cv2.Sobel(img_old, cv2.CV_64F, 1, 0, ksize=3)
        I_y = cv2.Sobel(img_old, cv2.CV_64F, 0, 1, ksize=3)

        # Calculate optical flow for each points in current pyramid level:
        for i, point in enumerate(current_points):
            x, y = point[0], point[1]

            # Calculate the left matrix, A:
            offset = int(window_size / 2)
            M_left[0, 0] = np.sum((I_x[y - offset:y + offset + 1, x - offset:x + offset + 1]) ** 2)
            M_left[0, 1] = np.sum((I_x[y - offset:y + offset + 1, x - offset:x + offset + 1]) * (
                I_y[y - offset:y + offset + 1, x - offset:x + offset + 1]))
            M_left[1, 0] = np.sum((I_x[y - offset:y + offset + 1, x - offset:x + offset + 1]) * (
                I_y[y - offset:y + offset + 1, x - offset:x + offset + 1]))
            M_left[1, 1] = np.sum((I_y[y - offset:y + offset + 1, x - offset:x + offset + 1]) ** 2)

            # Iteratively calculate the optical flow in one pyramid level.
            flow_iter = np.zeros(points.shape, dtype=np.float64)
            for k in range(iters):
                # Calculate gradient t using transformation matrix:
                mat_translation = np.array([[1, 0, -flow_results[i][0] - flow_iter[i][0]],
                                            [0, 1, -flow_results[i][1] - flow_iter[i][1]]])
                img_translate = cv2.warpAffine(img_new, mat_translation, (img_new.shape[1], img_new.shape[0]),
                                               flags=cv2.INTER_LINEAR)
                It = img_old - img_translate

                # Calculate the right matrix
                M_right[0] = np.sum((I_x[y - offset:y + offset + 1, x - offset:x + offset + 1]) * (
                    It[y - offset:y + offset + 1, x - offset:x + offset + 1]))
                M_right[1] = np.sum((I_y[y - offset:y + offset + 1, x - offset:x + offset + 1]) * (
                    It[y - offset:y + offset + 1, x - offset:x + offset + 1]))

                # Calculate the inverse matrix
                M_left_inverse = np.linalg.pinv(M_left)

                # Calculate the optical flow of the current point within current iteration:
                d = np.matmul(M_right, M_left_inverse)
                flow_iter[i] = flow_iter[i] + d

                # if residual smaller than the threshold, break the iterative process to speed up
                if np.linalg.norm(d) < residual:
                    break

            # Obtain the current flow result of the current point:
            if level > 0:
                # Transform flow corresponding to the level:
                flow_results[i] = 2 * (flow_results[i] + flow_iter[i])
            else:
                # If it is original image (not down-sampled), just add the flow:
                flow_results[i] = flow_results[i] + flow_iter[i]

    # Update the selected points using u,v.
    points = points + flow_results

    return points
