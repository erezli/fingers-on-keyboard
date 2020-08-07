import numpy as np
import cv2

img = cv2.imread('data/mess.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# _, thre = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
thre = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 0)
corners = cv2.goodFeaturesToTrack(thre, 100, 0.1, 1)  # max number of corners(from strongest); quality level of
# corner; min distance

corners = np.int0(corners)

for i in corners:
    x, y, = i.ravel()
    cv2.circle(img, (x, y), 3, 255, -1)

cv2.imshow('dst', img)
cv2.imshow('thre', thre)

if cv2.waitKey(0) & 0xFF == 27:
    cv2.destroyAllWindows()
