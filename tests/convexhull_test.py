# test the usage of convex hull in hand detection

import numpy as np
import cv2
from tests import hsv_ycbcr


def handDetection(img):     # the hand detection method still needs to be improved to be more accurate - remove
    # background noise: most likely to be keyboard
    hsvB = [(0, 0, 160), (206, 28, 255)]
    ycrcbB = [(186, 106, 113), (255, 141, 139)]
    handMask, hsvMask, ycrcbMask = hsv_ycbcr.SkinDetect(img, hsvB, ycrcbB)
    handMask = cv2.erode(handMask, np.ones((3, 3), np.uint8), iterations=3)
    handsImg = cv2.bitwise_not(handMask)
    return handsImg


cap = cv2.VideoCapture(0)

while cap.isOpened():
    _, frame = cap.read()
    hands_included = handDetection(frame)
    contours, _ = cv2.findContours(hands_included, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (255, 34, 100), 5)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
