import cv2


def drawBox(img, bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 1)
    cv2.putText(img, 'lost', (75, 75), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2)


cap = cv2.VideoCapture(0)

tracker = cv2.TrackerMOSSE_create()
# tracker = cv2.TrackerCSRT_create()  # better tracking but lower fps
success, img = cap.read()
bbox = cv2.selectROI('Tracking', img, False)
tracker.init(img, bbox)

while 1:
    timer = cv2.getTickCount()
    ret, img = cap.read()

    _, bbox = tracker.update(img)   # bbox has 4 values in the tuple

    if _:
        drawBox(img, bbox)
    else:
        cv2.putText(img, 'lost', (75, 75), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)
    fps = cv2.getTickFrequency()/(cv2.getTickCount()-timer)
    cv2.putText(img, str(int(fps)), (75, 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2)
    cv2.imshow('Tracking', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
