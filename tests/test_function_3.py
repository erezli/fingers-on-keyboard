from keyboard_in_VR import shift_tracking
import cv2
from keyboard_in_VR import keyboard_detect_ROI

cap = cv2.VideoCapture(0)

init = True
ovr = True

while 1:
    _, frame = cap.read()
    if init:
        x, y, w, h, approx, init = keyboard_detect_ROI.locate_keyboard(frame)
        cropped = keyboard_detect_ROI.crop_to_roi(x, y, w, h, approx, frame)
        track_window = (x, y, w, h)
    elif ovr:
        roi_hist = shift_tracking.calc_roi_hist(frame, track_window, [70, 38, 47], [120, 160, 160])
        ovr = False
    else:
        frame, track_window = shift_tracking.cam_shift(frame, track_window, roi_hist)
        (x, y, w, h) = track_window
        cv2.putText(frame, 'Keyboard', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        print('tracking')
        # cropped = keyboard_detect_ROI.crop_to_roi(x, y, w, h, approx, frame)

    cv2.imshow('crop', cropped)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
