import cv2
import numpy as np
from keyboard_in_VR.finger_detection_color import detect_finger_by_hsv
from keyboard_in_VR.finger_detect_contours import detect_finger_by_contours


def four_orange_corners(frame, hsv_l, hsv_u):
    filt = detect_finger_by_hsv(frame, hsv_l, hsv_u)
    x_l, y_l, w_l, h_l = detect_finger_by_contours(filt)    # add contour area control
    if len(x_l) == 4:
        position_list = [(x, y) for x, y in zip(x_l, y_l)]

    pass
