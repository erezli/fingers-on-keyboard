import cv2


def ask_for_tracker():
    print('Hi! What tracker API would you like to use?')
    print('Enter 0 for BOOSTING')
    print('Enter 1 for MIL: ')
    print('Enter 2 for KCF: ')
    print('Enter 3 for TLD: ')
    print('Enter 4 for MEDIANFLOW: ')
    print('Enter 5 for GOTURN: ')
    print('Enter 6 for MOSSE: ')
    print('Enter 7 for CSRT: ')

    choice = input('Please select your tracker: ')

    if choice == '0':
        tracker0 = cv2.TrackerBoosting_create()
    if choice == '1':
        tracker0 = cv2.TrackerMIL_create()
    if choice == '2':
        tracker0 = cv2.TrackerKCF_create()
    if choice == '3':
        tracker0 = cv2.TrackerTLD_create()
    if choice == '4':
        tracker0 = cv2.TrackerMedianFlow_create()
    if choice == '5':
        tracker0 = cv2.TrackerGOTURN_create()
    if choice == '6':
        tracker0 = cv2.TrackerMOSSE_create()
    if choice == '7':
        tracker0 = cv2.TrackerCSRT_create()

    return tracker0


tracker = ask_for_tracker()

tracker_name = str(tracker).split()[0][1:]

cap = cv2.VideoCapture(0)

ret, frame = cap.read()

roi = cv2.selectROI('roi', frame)

ret = tracker.init(frame, roi)

while 1:
    ret, frame = cap.read()
    success, roi = tracker.update(frame)
    (x, y, w, h) = tuple(map(int, roi))

    if success:
        pts1 = (x, y)
        pts2 = (x+w, y+h)
        cv2.rectangle(frame, pts1, pts2, (255, 255, 0), 3)
    else:
        cv2.putText(frame, 'Fail to track the object', (100, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (24, 25, 255), 3)

    cv2.putText(frame, tracker_name, (20, 400),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 3, 35), 3)

    cv2.imshow(tracker_name, frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
