import numpy as np
import cv2

img_2 = cv2.imread('data/chessboard.png')
img = img_2.copy()
img = cv2.pyrDown(img_2)
img = cv2.pyrDown(img_2)
cv2.imshow('img', img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray, 2, 3, 0.04)    # img, blockSize, ksize, Harris detector free parameter

det = cv2.dilate(dst, None)
img[dst > 0.01 * dst.max()] = [0, 0, 255]

cv2.imshow('dst', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
