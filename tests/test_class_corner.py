from keyboard_in_VR.Corners import Corners
import cv2
import numpy as np


cap = cv2.VideoCapture(0)

orange_corners = Corners([0, 128, 105], [15, 255, 255])

while cap.isOpened():
    _, frame = cap.read()
    if len(orange_corners.positions) == 0:
        orange_corners.update_property(frame)
    else:
        print('found')
        print(orange_corners.positions)
        print(orange_corners.track_win)
        orange_corners.update_roi_hists(frame)
        break

# dont use optical flow
# this is cam shift tracking
while cap.isOpened():
    _, frame = cap.read()

    frame_2 = orange_corners.cam_shift_track(frame)
    cv2.imshow('frame', frame_2)
    print(orange_corners.positions)
    print(orange_corners.track_win)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
