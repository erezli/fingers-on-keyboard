import cv2
import numpy as np
from keyboard_in_VR.translucent_hands import hands_on_keyboard, background_subtraction, add_translucent_hands
from keyboard_in_VR.finger_detection_color import detect_finger_by_hsv

cap = cv2.VideoCapture(0)
_, first_frame = cap.read()
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

while cap.isOpened():
    _, frame = cap.read()

    # diff = hands_on_keyboard(frame, first_frame) # takes the first frame
    # - only when the hand is not in the frame initially
    # diff = background_subtraction(frame, fgbg)   # only when the hand is moving
    # diff = np.array(diff, np.uint8)
    # frame = np.array(frame, np.uint8)
    # cv2.imshow('diff', diff)
    # print(type(diff))
    # print(type(frame))
    # print(diff.shape)
    # print(frame.shape)

    filt, mask = detect_finger_by_hsv(frame, [0, 0, 149], [39, 121, 255])
    cv2.imshow('mask', mask)
    cv2.imshow('filt', filt)
    res = add_translucent_hands(frame, filt, first_frame, transparency=3)

    cv2.imshow("result", res)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
