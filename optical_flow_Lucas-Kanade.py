import cv2
import numpy as np


cap = cv2.VideoCapture(0)
_, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_RGB2GRAY)

lk_params = dict(winSize=(10, 10), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


def select_point(event, x, y, flags, params):
    global point, point_selected, old_point
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        point_selected = True
        old_point = np.array([[x, y]], dtype=np.float32)


cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', select_point)

point_selected = False
point = ()
old_point = np.array(([[]]))

while 1:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    if point_selected is True:
        cv2.circle(frame, point, 5, (0, 0, 255), 2)

        new_point, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray,
                                                             old_point, None, **lk_params)
        old_gray = gray.copy()
        old_point = new_point

        x, y = new_point.ravel()
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    cv2.imshow('Frame', frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
