import cv2
import numpy as np
from keyboard_in_VR.translucent_hands import hands_on_keyboard, background_subtraction, add_translucent_hands


cap = cv2.VideoCapture(0)
_, first_frame = cap.read()
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows= False)

while cap.isOpened():
    _, frame = cap.read()

    diff = hands_on_keyboard(frame, first_frame)
    # diff = np.array(diff, np.uint8)
    # frame = np.array(frame, np.uint8)
    cv2.imshow('diff', diff)
    print(type(diff))
    print(type(frame))
    print(diff.shape)
    print(frame.shape)
    res = add_translucent_hands(frame, diff, first_frame)

    cv2.imshow("result", res)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
