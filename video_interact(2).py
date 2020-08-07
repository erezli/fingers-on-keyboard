# Import Library
import cv2

# Callback Function for the Mouse, Circle
def draw_circle(event, x, y, flags, param):
    global pt, mouse_clicked
    if event == cv2.EVENT_LBUTTONDOWN():
        pt = (x, y)
        mouse_clicked = False
    if event == cv2.EVENT_RBUTTONDOWN():


        '''
        if mouse_clicked:
            pt = (0, 0)
            mouse_clicked = False

        if not mouse_clicked:
            pt = (x, y)
        '''
# Get the Mouse Click Down & Up
# and Track the Center


# Zero Drawing of the Circle
mouse_clicked = False
pt = (0, 0)

# Take a video
cap = cv2.VideoCapture(0)

# Create a Named Window for the Connections
cv2.namedWindow('test')


# Bind our Function with the Mouse Clicks
cv2.setMouseCallback('test', draw_circle)


# Time for Magic
while 1:
    ret, frame = cap.read()

# Capture the frame


# Check if Clicked is True


# Draw a Circle on the Frame


# Display the Frame


# 27 == esc button. You can use any other button like ord('q')
# If you try to close the window pressing the x you'll get in trouble (^_^)


# Never forget first to release
# And then to destroy
