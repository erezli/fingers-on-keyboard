import cv2
import numpy as np


img = cv2.imread('data/pic1.png')
imgrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(imgrey, 100, 255, cv2.THRESH_BINARY)

trans = cv2.dilate(thresh, None, iterations=1)
blur = cv2.GaussianBlur(trans, (1, 1), 0)
cv2.imshow('thre', thresh)
contours, _ = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cv2.imshow('dilate', trans)
for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour, True), True)
    cv2.drawContours(img, [approx], 0, (0, 0, 0), 5)
    x = approx.ravel()[0]
    y = approx.ravel()[1]
    if len(approx) == 3:
        cv2.putText(img, 'triangle', (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 166, 0))
    elif len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspectRatio = float(w)/h
        print(aspectRatio)
        if 0.95 <= aspectRatio <= 1.05:
            cv2.putText(img, 'square', (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 190, 0))
        else:
            cv2.putText(img, 'rectangle', (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 210, 0))
    elif len(approx) == 5:
        cv2.putText(img, 'pentagon', (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 135, 0))
    elif len(approx) == 10:
        cv2.putText(img, 'star', (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 246, 0))
    else:
        cv2.putText(img, 'circle', (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0))


cv2.imshow('shapes', img)
cv2.waitKey()
cv2.destroyAllWindows()
