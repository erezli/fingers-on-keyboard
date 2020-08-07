import cv2
import numpy as np

hand_detection = cv2.CascadeClassifier('../haarcascades/Hand.Cascade.1.xml')
# hand_detection_2 = cv2.CascadeClassifier('../haarcascades/hand.xml')
# fist_detection = cv2.CascadeClassifier('../haarcascades/fist.xml')
# palm_detection = cv2.CascadeClassifier('../haarcascades/fist.xml')


def adaptive_thresholding(img):
    mask = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 39, 2)
    cv2.imshow('thresholding', mask)
    return mask


def morph_transformation(mask, type_num):
    kernel = np.ones((2, 2), np.uint8)
    dilation = cv2.dilate(mask, kernel, iterations=1)
    erosion = cv2.erode(mask, kernel, iterations=1)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mg = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
    th = cv2.morphologyEx(mask, cv2.MORPH_TOPHAT, kernel)
    type_list = [dilation, erosion, opening, closing, mg, th]
    cv2.imshow('morphological transformation', type_list[type_num])
    return type_list[type_num]


def edge_detect(img, type_num):
    lap = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
    sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobelX = np.uint8(np.absolute(sobelX))
    sobelY = np.uint8(np.absolute(sobelY))
    sobel_combined = cv2.bitwise_or(sobelX, sobelY)
    canny = cv2.Canny(img, 100, 200)
    type_list = [lap, sobelX, sobelY, sobel_combined, canny]
    cv2.imshow('Detect edge', type_list[type_num])
    return type_list[type_num]


def detect_hand(img):
    hand_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mask = adaptive_thresholding(hand_img)
    mor_trans = morph_transformation(mask, 1)
    edge = edge_detect(hand_img, 4)

    hand_rectangle = hand_detection.detectMultiScale(edge, 1.3, 5)

    for (x, y, w, h) in hand_rectangle:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 10)

    return img


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    detect_hand(frame)
#    detect_hand_2(frame)
#    detect_fist(frame)
#    detect_palm(frame)

    cv2.imshow('Face Detection Video', frame)

    if cv2.waitKey(3) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()



