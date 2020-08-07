import cv2
import numpy as np


def draw_circle(event, x, y, flags, param):
    """
    x, y, flags, param are feed from OpenCV automatically
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x, y), 65, (255, 0, 0), -1)
    elif event == cv2.EVENT_RBUTTONDOWN:
        cv2.circle(img, (x, y), 10, (23, 209, 23), 3)



cv2.namedWindow(winname='drawing')

cv2.setMouseCallback('drawing', draw_circle)

img = np.zeros((512, 512, 3), np.int8)

while True:

    cv2.imshow('drawing', img)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cv2.destroyAllWindows()

