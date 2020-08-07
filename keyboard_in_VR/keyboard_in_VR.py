import cv2
import numpy as np
from keyboard_in_VR.detected_object import ObjectFrame
from keyboard_in_VR.Keyboard import Keyboard
from keyboard_in_VR.Finger import Fingers


cap = cv2.VideoCapture(0)

keyboard = Keyboard([70, 38, 47], [120, 255, 255])
while cap.isOpened():
    _, frame = cap.read()
    keyboard.get_position_contour(frame)
    if keyboard.track_window != [0, 0, 0, 0]:
        break

_, frame = cap.read()
# get position of vertices
# keyboard.get_position_4_corners(frame)
keyboard.vertices = keyboard.approx
keyboard_frame = keyboard.perspective_transformation(frame)
w, h, c = keyboard_frame.shape
keyboard_layout = {(range(0, w/2), range(0, h/2)): 'A',
                   (range(w/2, w), range(0, h/2)): 'B',
                   (range(0, w/2), range(h/2, h)): 'C',
                   (range(w/2, w), range(h/2, h)): 'D'}
keyboard.layout = keyboard_layout

fingers = Fingers([88, 130, 200], [255, 255, 255])
