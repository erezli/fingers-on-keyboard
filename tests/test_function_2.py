import cv2
from keyboard_in_VR import keyboard_detect_corner

cap = cv2.VideoCapture(0)

while 1:
    _, frame = cap.read()

    frame, x, y = keyboard_detect_corner.detect_keyboard_corner(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
