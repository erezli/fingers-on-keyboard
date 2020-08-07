import cv2
import numpy as np

img1 = np.zeros((250, 500, 3), np.uint8)
img1 = cv2.rectangle(img1, (200, 0), (300, 100), (255, 255, 255), -1)
img2 = np.zeros((250, 500, 3), np.uint8)
img2 = cv2.rectangle(img2, (0, 0), (250, 250), (255, 255, 255), -1)
img_xor = cv2.bitwise_xor(img1, img2)


cv2.imshow('bitXor', img_xor)
cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.waitKey()
cv2.destroyAllWindows()
