import cv2
from tests import hsv_ycbcr

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
hsvB = [(0, 0, 160), (206, 28, 255)]
ycrcbB = [(186, 106, 113), (255, 141, 139)]


def movementFilter(frame):
    fg = fgbg.apply(frame)
    msk = cv2.bitwise_and(frame, frame, mask=fg)
    sub = cv2.absdiff(frame, msk)
    return sub, msk


def colourFilter(frame):
    nohand, hsv, ycrcb = hsv_ycbcr.SkinDetect(frame, hsvB, ycrcbB)
    nohand = cv2.bitwise_and(frame, frame, mask=nohand)
    cv2.imshow("hsvOnly", hsv)
    cv2.imshow("ycrcbOnly", ycrcb)
    cv2.imshow("hands", nohand)
    return nohand, hsv, ycrcb


cap = cv2.VideoCapture(0)
count = 0

while cap.isOpened():
    global added
    count += 1
    _, frame = cap.read()
    bg, hsvbg, ycrcbbg = colourFilter(frame)
    if count == 1:
        added = bg
    else:
        added = cv2.addWeighted(bg, 1/count, added, 1-(1/count), 0)

    show = cv2.addWeighted(added, .5, frame, .5, 0)
    cv2.imshow("Frame", show)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()