import numpy as np
import matplotlib.pyplot as plt
import cv2

black_img = np.zeros(shape=(512, 512, 3), dtype=np.int16)
print(black_img.shape)
plt.imshow(black_img)
w, h, c = black_img.shape
print(w, h)
print(black_img.size)

cv2.circle(img=black_img, center=(400, 100), radius=50, color=(255, 0, 0), thickness=8)
plt.imshow(black_img)

# cv2.waitKey()
# cv2.destroyAllWindows()
