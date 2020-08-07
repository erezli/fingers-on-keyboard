import cv2


lena = cv2.imread('data/lena.jpg', -1)
# print(lena.shape)
# print(lena)

cv2.imshow("lena\"s", lena)
k = cv2.waitKey(0) & 0xFF # for 64-bit

if k == 27: # this is the value of esc key
    cv2.destroyAllWindows()

elif k == ord('s'):
    cv2.imwrite('lena_copy.png', lena)
    cv2.destroyAllWindows()

