import cv2
from keyboard_in_VR import finger_detection_color

cap = cv2.VideoCapture(0)

while 1:
    _, frame = cap.read()
    filtered, mask = finger_detection_color.detect_finger_by_hsv(frame, [80, 50, 11], [255, 255, 217])
    # grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(filtered, 5, 50, 100)
    # _, thresh = cv2.threshold(filtered, 150, 255, cv2.THRESH_TOZERO_INV)
    # erosion = cv2.erode(thresh, None, iterations=3)
    #
    #trans = cv2.morphologyEx(blur, cv2.MORPH_OPEN, None, iterations=3)
    #dilation = cv2.dilate(trans, None, iterations=2)

    cv2.imshow('res', blur)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
