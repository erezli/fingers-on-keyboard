import cv2
import numpy as np


cap = cv2.VideoCapture(0)

while cap.isOpened():

    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    cv2.imshow('edges', edges)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=10, maxLineGap=5)

    if lines is not None:
        print('captured something')
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow('image', frame)

    else:
        print('captured nothing')
        pass

    if cv2.waitKey(1) == ord('k'):
        break

cap.release()
cv2.destroyAllWindows()
