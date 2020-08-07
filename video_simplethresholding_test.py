import cv2


cap = cv2.VideoCapture(0)

while 1:

    ret, frame = cap.read()

    _, th1 = cv2.threshold(frame, 100, 255, cv2.THRESH_BINARY)

    cv2.imshow('video', th1)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
