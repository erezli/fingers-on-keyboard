import cv2

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()


def backgroundOnly(frame):
    fg = fgbg.apply(frame)
    msk = cv2.bitwise_and(frame, frame, mask=fg)
    sub = cv2.absdiff(frame, msk)
    return sub, msk


cap = cv2.VideoCapture(0)

while cap.isOpened():
    _, frame1 = cap.read()
    subtract1, mask1 = backgroundOnly(frame1)
    cv2.waitKey(10)
    _, frame2 = cap.read()
    subtract2, mask2 = backgroundOnly(frame2)
    cv2.waitKey(10)
    _, frame3 = cap.read()
    subtract3, mask3 = backgroundOnly(frame3)
    cv2.waitKey(10)
    _, frame4 = cap.read()
    subtract4, mask4 = backgroundOnly(frame4)
    cv2.waitKey(10)
    _, frame5 = cap.read()
    subtract5, mask5 = backgroundOnly(frame5)
    _, frame6 = cap.read()
    subtract6, mask6 = backgroundOnly(frame6)
    _, frame7 = cap.read()
    subtract7, mask7 = backgroundOnly(frame7)

    cv2.imshow("Frame", frame1)

    #cv2.imshow("s1", subtract1)
    #cv2.imshow("s2", subtract2)
    #cv2.imshow("s3", subtract3)
    #cv2.imshow("s4", subtract4)
    #cv2.imshow("s5", subtract5)
    #cv2.imshow("s6", subtract6)
    #cv2.imshow("s7", subtract7)

    added = cv2.addWeighted(subtract1, .5, mask1, .5, 0)
    added = cv2.addWeighted(subtract2, .334, added, .666, 0)
    added = cv2.addWeighted(subtract3, .25, added, .75, 0)
    added = cv2.addWeighted(subtract4, .2, added, .8, 0)
    added = cv2.addWeighted(subtract5, .1667, added, .8223, 0)
    added = cv2.addWeighted(subtract6, .143, added, .857, 0)
    added = cv2.addWeighted(subtract7, .125, added, .875, 0)
    cv2.imshow("added", added)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()