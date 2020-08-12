import cv2
import numpy as np

# fgbg = cv2.createBackgroundSubtractorMOG2(history=, varThreshold=, detectShadows= )


def hands_on_keyboard(frame, first_frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    difference = cv2.absdiff(first_gray, gray_frame)
    # difference = cv2.absdiff(first_frame, frame)
    _, difference = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)
    return difference


def background_subtraction(frame, fgbg):
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    fgmask = fgbg.apply(frame)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    return fgmask


def add_translucent_hands(frame, fgmask, first_frame):
    fgmask = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
    hands = cv2.bitwise_and(frame, fgmask)
    res = cv2.addWeighted(first_frame, 0.7, hands, 0.3, 0)
    return res
