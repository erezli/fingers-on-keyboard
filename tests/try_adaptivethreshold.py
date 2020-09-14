import cv2
import numpy as np


def nothing(x):
    pass


cap = cv2.VideoCapture(0)

cv2.namedWindow('trackbar')

cv2.createTrackbar('C', 'trackbar', 0, 30, nothing)

while cap.isOpened():
    _, frame = cap.read()
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    C = cv2.getTrackbarPos('C', 'trackbar')

    # _, th1 = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
    # th2 = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 359, C)
    th3 = cv2.adaptiveThreshold(frameGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 799, -20)
    # th4 = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 39, 5)
    mask = np.zeros(frame.shape, dtype='uint8')
    contour, hierarchy = cv2.findContours(th3, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for c in contour:
        if cv2.contourArea(c) > 1500:
            cv2.drawContours(mask, [c], -1, (255, 255, 255), -1)
    # cv2.drawContours(frame, contour, -1, (255, 255, 255), thickness=-1)
    # cv2.drawContours(mask, contour, -1, (255, 255, 255), -1)

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    # cv2.imshow('th1', th1)
    # cv2.imshow('th2', th2)
    cv2.imshow('th3', th3)
    # cv2.imshow('th4', th4)

    if cv2.waitKey(2) == 27:
        break

cap.release()
cv2.destroyAllWindows()
