import cv2

cap = cv2.VideoCapture(0)
_, first_frame = cap.read()

while cap.isOpened():
    _, frame = cap.read()
    res = cv2.addWeighted(first_frame, 0.5, frame, 0.5, 0)

    cv2.imshow('res', res)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()