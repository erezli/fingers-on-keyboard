import cv2


face_detection = cv2.CascadeClassifier( cv2.data.haarcascades +'haarcascade_frontalface_default.xml')


def detect_face(img):

    face_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_rectangle = face_detection.detectMultiScale(face_img)

    for (x, y, w, h) in face_rectangle:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 10)

    return img


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    detect_face(frame)

    cv2.imshow('Face Detection Video', frame)

    if cv2.waitKey(3) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()



