import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    cv2.imshow('frame', frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thre = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    filt = cv2.erode(thre, None)
    corners = cv2.goodFeaturesToTrack(filt, 4, 0.9, 50)  # max number of corners(from strongest); quality level of
    # corner; min distance
    if corners is not None:
        corners = np.int0(corners)

        for i in iter(corners):
            x, y, = i.ravel()
            cv2.circle(frame, (x, y), 9, 255, -1)

    cv2.imshow('dst', frame)
    cv2.imshow('thre', thre)
    cv2.imshow('filter', filt)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
cap.release()
