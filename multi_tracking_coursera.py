import cv2
import sys
from random import randint


tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD',
                 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']


def tracker_name(tracker_type):
    if tracker_type == tracker_types[0]:
        tracker = cv2.TrackerBoosting_create()
    elif tracker_type == tracker_types[1]:
        tracker = cv2.TrackerMIL_create()
    elif tracker_type == tracker_types[2]:
        tracker = cv2.TrackerKCF_create()
    elif tracker_type == tracker_types[3]:
        tracker = cv2.TrackerTLD_create()
    elif tracker_type == tracker_types[4]:
        tracker = cv2.TrackerMedianFlow_create()
    elif tracker_type == tracker_types[5]:
        tracker = cv2.TrackerGOTURN_create()
    elif tracker_type == tracker_types[6]:
        tracker = cv2.TrackerMOSSE_create()
    elif tracker_type == tracker_types[7]:
        tracker = cv2.TrackerCSRT_create()

    else:
        tracker = None
        print('No tracker found')
        print('Choose from these trackers: ')
        for tr in tracker_types:
            print(tr)

    return tracker


if __name__ == '__main__':
    print('Default tracking algorithm MOSSE \nAvailable algorithms are: \n')
    for ta in tracker_types:
        print(ta)

    trackerType = 'MOSSE'

    cap = cv2.VideoCapture(0)
    success, frame = cap.read()

    if not success:
        print('Cannot read the video')

    rects = []
    colors = []

    while True:

        rect_box = cv2.selectROI('MultiTracker', frame)
        rects.append(rect_box)
        colors.append((randint(64, 255), randint(64, 255), randint(64, 255)))
        print('Press q to stop selecting boxes and start multi-tracking')
        print('Press any key to select another box')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print('Selected boxes {rects}')
    multitracker = cv2.MultiTracker_create()

    for rect_box in rects:
        multitracker.add(tracker_name(trackerType), frame, rect_box)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        success, boxes = multitracker.update(frame)

        for i, newbox in enumerate(boxes):
            pts1 = (int(newbox[0]), int(newbox[1]))
            pts2 = (int(newbox[0]) + int(newbox[2]),
                    int(newbox[1]) + int(newbox[3]))
            cv2.rectangle(frame, pts1, pts2, colors[i], 2, 1)

        cv2.imshow('Multitracker', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
