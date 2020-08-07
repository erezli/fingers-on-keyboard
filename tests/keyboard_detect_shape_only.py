import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    imgrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # _, thresh = cv2.threshold(imgrey, 150, 255, cv2.THRESH_BINARY)
    thresh = cv2.adaptiveThreshold(imgrey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 2309, 1)
    trans = cv2.erode(thresh, None, iterations=1)
    blur = cv2.GaussianBlur(trans, (1, 1), 0)
    cv2.imshow('thre', thresh)
    contours, _ = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.imshow('dilate', trans)
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour, True), True)
        x = approx.ravel()[0]
        y = approx.ravel()[1]
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            if cv2.contourArea(contour) > 10000:
                cv2.drawContours(img, [approx], 0, (0, 0, 0), 5)

    cv2.imshow('shapes', img)
    if cv2.waitKey(3) == 27:
        break

cap.release()
cv2.destroyAllWindows()
