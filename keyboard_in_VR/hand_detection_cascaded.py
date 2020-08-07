"""
When the functions are being called from other file, modify the return value to the positions of the
detected objects, (x, y, w, h).

"""

import cv2


hand_detection = cv2.CascadeClassifier('../haarcascades/Hand.Cascade.1.xml')
hand_detection_2 = cv2.CascadeClassifier('../haarcascades/hand.xml')
fist_detection = cv2.CascadeClassifier('../haarcascades/fist.xml')
palm_detection = cv2.CascadeClassifier('../haarcascades/fist.xml')

# functions for 4 cascade classifiers


def detect_hand(img):

    hand_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hand_rectangle = hand_detection.detectMultiScale(hand_img, 1.3, 5)

    for (x, y, w, h) in hand_rectangle:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 10)
        return x, y, w, h


def detect_hand_2(img):

    hand_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hand_rectangle = hand_detection.detectMultiScale(hand_img, 1.3, 5)

    for (x, y, w, h) in hand_rectangle:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 10)
        return x, y, w, h


def detect_fist(img):

    fist_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    fist_rectangle = fist_detection.detectMultiScale(fist_img, 1.3, 5)

    for (x, y, w, h) in fist_rectangle:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 10)
        return x, y, w, h


def detect_palm(img):

    palm_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    palm_rectangle = palm_detection.detectMultiScale(palm_img, 1.3, 5)

    for (x, y, w, h) in palm_rectangle:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 10)
        return x, y, w, h


"""
# open camera to detect hands, fist and palm
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    detect_hand(frame)
    detect_hand_2(frame)
    detect_fist(frame)
    detect_palm(frame)

    cv2.imshow('Face Detection Video', frame)

    if cv2.waitKey(3) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()
"""