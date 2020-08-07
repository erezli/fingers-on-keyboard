import cv2
import numpy as np


img = cv2.imread("data/Hands-Front-Back.jpg")
img = cv2.resize(img, (int(img.shape[1]*0.5), int(img.shape[0]*0.5)))
grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
template = cv2.imread("data/hand1.jfif", 0)
template = cv2.resize(template, (int(img.shape[1]*0.65), int(img.shape[0]*0.65)))
template = cv2.rotate(template, cv2.ROTATE_90_CLOCKWISE)
w, h = template.shape[::-1]

res = cv2.matchTemplate(grey_img, template, cv2.TM_CCOEFF_NORMED)

threshold = 0.5
loc = np.where(res >= threshold)
print(loc)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 3)

cv2.imshow("img", img)
cv2.imshow("tem", template)
cv2.waitKey(0)
cv2.destroyAllWindows()
