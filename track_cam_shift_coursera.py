import cv2
import numpy as np

cap = cv2.VideoCapture(0)

ret, frame_1 = cap.read()

face_casc = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
face_rects = face_casc.detectMultiScale(frame_1)

x, y, w, h = tuple(face_rects[0])
track_window = (x, y, w, h)

roi = frame_1[y: y+h, x: x+w]

hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])

cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while 1:
    ret, frame = cap.read()

    if ret == True:

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dest = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        # camshift
        ret, track_window = cv2.CamShift(dest, track_window, term_crit)

        # draw rectangle
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        img2 = cv2.polylines(frame, [pts], True, (0, 255, 0), 5)

        cv2.imshow('Cam Shift', img2)

        if cv2.waitKey(2) & 0xFF == 27:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
