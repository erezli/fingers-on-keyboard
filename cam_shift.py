import numpy as np
import cv2

cap = cv2.VideoCapture(0)

# take first frame of the video
ret, frame = cap.read()
# setup initial location of window
x, y, w, h = 300, 200, 100, 50
track_window = (x, y, w, h)
# set up the ROI for tracking
roi = frame[y:y+h, x:x+w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
print(roi_hist.shape)
# setup the termination criteria, either 10 iteration or move by at least 1 pt
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
cv2.imshow('roi', roi)

while 1:
    ret, frame = cap.read()
    if ret:

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)  # ret has value x, y, w, h, rot
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        final_img = cv2.polylines(frame, [pts], True, (255, 255, 9), 2)
        cv2.imshow('dst', dst)
        cv2.imshow('final', final_img)
        cv2.imshow('frame', frame)
        k = cv2.waitKey(1)
        if k == 27:
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()
