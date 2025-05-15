import numpy as np
import cv2
import random

def get_vessel_region(image, vessel_threshold=30):
    _, vessel_mask = cv2.threshold(image, vessel_threshold, 255, cv2.THRESH_BINARY)
    return vessel_mask

def get_negative_region(vessel_mask, dilation_size=8):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
    dilated_vessel_mask = cv2.dilate(vessel_mask, kernel, iterations=1)
    negative_region = cv2.bitwise_not(dilated_vessel_mask)
    return negative_region

def get_random_points_from_region(region_mask, num_points=10):
    valid_points = [(x, y) for (x, y), val in np.ndenumerate(region_mask) if val]

    if len(valid_points) <= num_points:
        return valid_points

    selected_points = random.sample(valid_points, num_points)
    return selected_points

def draw_points_on_image(image, points, color=(0, 255, 0), radius=3, copy=True):
    if copy:
        marked_img = image.copy()
    else:
        marked_img = image

    if len(marked_img.shape) == 2:
        marked_img = cv2.cvtColor(marked_img, cv2.COLOR_GRAY2BGR)

    for (x, y) in points:
        x, y = int(round(x)), int(round(y))
        cv2.circle(marked_img, (x, y), radius=radius, color=color, thickness=-1)

    return marked_img


def get_points(image, vessel_threshold=30, dilation_size=1, num_positive=16, num_negative=16):
    vessel_region = get_vessel_region(image, vessel_threshold)
    negative_region = get_negative_region(vessel_region, dilation_size)

    positive_points = get_random_points_from_region(vessel_region, num_positive)
    negative_points = get_random_points_from_region(negative_region, num_negative)

    return positive_points, negative_points

