"""
dictionary = {1: 'a', 2: 'b', 3: 'c'}
print(dictionary[1])
for i in dictionary.keys():
    print(i)

pos = [(1, 2), (3, 4)]
for (x, y) in pos:
    print(x)
    print(y)

"""

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
