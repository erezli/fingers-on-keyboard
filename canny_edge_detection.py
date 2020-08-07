import cv2
# import numpy as np
# import matplotlib.pyplot as plt


def nothing(x):
    print(x)


cap = cv2.VideoCapture(0)
# img = cv2.imread('data/sudoku.png', 0)
# print(img.shape)
cv2.namedWindow('image')
cv2.createTrackbar('th1', 'image', 0, 255, nothing)
cv2.createTrackbar('th2', 'image', 0, 255, nothing)

while 1:
    _, img = cap.read()
    th1 = cv2.getTrackbarPos('th1', 'image')
    th2 = cv2.getTrackbarPos('th2', 'image')
    canny = cv2.Canny(img, th1, th2)
    # canny = cv2.resize(canny, (1000, 420))
    canny = cv2.pyrDown(canny)
    cv2.imshow('image', canny)
    if cv2.waitKey(2) & 0xFF == 27:
        break

cv2.destroyAllWindows()

'''
titles = ['image', 'canny']
images = [img, canny]
for i in range(len(images)):
    plt.subplot(1, 2, 1+i), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
'''
