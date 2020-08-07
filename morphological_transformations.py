import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('data/smarties.png', cv2.IMREAD_GRAYSCALE)
print(img.shape)
# img = cv2.resize(img, (1000, 420))
# print(img.shape)
mask = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 39, 5)

kernel = np.ones((2, 2), np.uint8)

dilation = cv2.dilate(mask, kernel, iterations=1)
erosion = cv2.erode(mask, kernel, iterations=1)
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mg = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
th = cv2.morphologyEx(mask, cv2.MORPH_TOPHAT, kernel)


titles = ['image', 'mask', 'dilation', 'erosion', 'opening', 'closing', 'gradient', 'tophat']
images = [img, mask, dilation, erosion, opening, closing, mg, th]

for i in range(len(images)):

    plt.subplot(3, 3, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])


plt.show()

