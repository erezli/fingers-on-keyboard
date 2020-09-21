import cv2
import numpy as np


def HandFiltering(frame):
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(frameGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 899, -20)
    mask = np.zeros(frame.shape, dtype=frame.dtype)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        if cv2.contourArea(c) > 1000:
            cv2.drawContours(mask, [c], -1, (255, 255, 255), -1)
    # mask = cv2.bitwise_not(mask)
    maskF = cv2.bitwise_or(frame, mask)

    return maskF, mask
