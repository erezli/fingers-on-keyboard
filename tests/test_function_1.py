import cv2
from keyboard_in_VR import keyboard_detect_ROI

cap = cv2.VideoCapture(0)

while 1:
    _, frame = cap.read()
    '''
    filtered, mask = finger_detection_color.detect_finger_by_hsv(frame, [80, 25, 11], [255, 255, 217])
    blur = cv2.bilateralFilter(filtered, 9, 50, 100)
    trans = cv2.morphologyEx(blur, cv2.MORPH_OPEN, None, iterations=2)
    dilation = cv2.dilate(trans, None, iterations=2)
    grey = cv2.cvtColor(dilation, cv2.COLOR_BGR2GRAY)
    cv2.imshow('处理后', dilation)
    # detect keyboard
    contours, _ = cv2.findContours(grey, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        count = 0
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

        # x = approx.ravel()[0]
        # y = approx.ravel()[1]
        if len(approx) == 4:
            if cv2.contourArea(contour) >= 50000:
                count += 1
                if count >= 2:
                    print('seems like two keyboards are being detected...')
                    cv2.putText(frame, 'seems like two keyboards are being detected...', (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                else:
                    cv2.putText(frame, 'found keyboard', (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                x, y, w, h = cv2.boundingRect(approx)
                cv2.drawContours(frame, [approx], 0, (255, 76, 186), 5)
                # return x, y, w, h
            elif 50000 > cv2.contourArea(contour) > 1500:
                # print('maybe bring the camera closer')
                cv2.putText(frame, 'maybe bring the camera closer', (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                # print('no keyboard detected, try harder')
                cv2.putText(frame, 'no keyboard detected, try harder', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    '''
    x, y, w, h, approx, init = keyboard_detect_ROI.locate_keyboard(frame)
    print(approx)
    # cropped = keyboard_detect_ROI.crop_to_roi(x, y, w, h, approx, frame)
    # cv2.imshow('crop', cropped)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
