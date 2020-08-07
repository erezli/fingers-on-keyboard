import cv2
import numpy as np

cap = cv2.VideoCapture(0)

ret, frame = cap.read()

face_casc = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
face_rects = face_casc.detectMultiScale(frame)
print(face_rects)

x, y, w, h = tuple(face_rects[0])
track_window = (x, y, w, h)

roi = frame[y:y+h, x:x+w]

hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
print(roi_hist)
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while 1 :

    ret, frame = cap.read()

    if ret:

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        dest_roi = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        ret, track_window = cv2.meanShift(dest_roi, track_window, term_crit)

        w, y, w, h = track_window

        img2 = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 3)

        cv2.imshow('FaceTracker', img2)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()

