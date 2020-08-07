import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('data/mess.jpg', cv2.IMREAD_GRAYSCALE)

lap = cv2.Laplacian(img, cv2.CV_64F, ksize=3)

lap = np.uint8(np.absolute(lap))

sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY))

sobel_combined = cv2.bitwise_or(sobelX, sobelY)

titles = ['image', 'Laplacian', 'sobelX', 'sobelY', 'XY combined']
images = [img, lap, sobelX, sobelY, sobel_combined]

for i in range(len(images)):
    plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])


plt.show()
