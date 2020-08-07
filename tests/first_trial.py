import cv2
from keyboard_in_VR import finger_detection_color
from keyboard_in_VR import finger_track_contours
from keyboard_in_VR import finger_detect_contours


cap = cv2.VideoCapture(0)

while cap.isOpened():

    ret, frame = cap.read()
    ######
    # find fingers
    ######
    res, mask = finger_detection_color.detect_finger_by_hsv(frame, [88, 130, 200], [255, 255, 255])

    x_f, y_f, w_f, h_f = finger_detect_contours.detect_finger_by_contours(res)
    finger_positions = (x_f, y_f, w_f, h_f)
    # print(x2, y2, w2, h2)

    ######
    # find keyboard
    ######

    if cv2.waitKey(9) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
