import cv2


img = cv2.imread('data/apple.jpg')
img2 = cv2.imread('data/opencv-logo.png')

print(img.shape)
print(img2.shape)
print(img.size)
print(img.dtype)
b, g, r = cv2.split(img)
# print(b)
img = cv2.merge((b, g, r))
img2 = cv2.resize(img2, (512, 512))

'''
def click_event(event, x, y,  flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ', ', y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        strXY = str(x) + ', ' + str(y)
        cv2.putText(img, strXY, (x, y), font, .5, (255, 255, 0), 1)
        cv2.imshow('image', img)
'''

bar = img[4:60, 132:232]    # region of interest - ROI
img[4:60, 232:332] = cv2.flip(bar, 1)
dst = cv2.addWeighted(img, .5, img2, .5, 0)

cv2.imshow('added', dst)
cv2.imshow("image", img)
cv2.imshow('cv', img2)
# cv2.setMouseCallback('image', click_event)

cv2.waitKey()
cv2.destroyAllWindows()


