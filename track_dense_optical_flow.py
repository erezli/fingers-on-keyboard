# Import Libraries
import cv2
import numpy as np

# Video Capture
cap = cv2.VideoCapture(0)

# Read the capture and get the first frame
ret, first_frame = cap.read()
# print(first_frame.shape)
# Convert frame to Grayscale
p_grey = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
# print(p_grey.shape)
# Create# Create an image with the same dimensions as the frame for later drawing purposes
mask = np.zeros_like(first_frame)
# print(mask)
# Saturation to maximum
mask[..., 1] = 255
# print(mask)

# While loop
while cap.isOpened():

    # Read the capture and get the first frame
    ret, frame = cap.read()

    # Open new window and display the input frame
    cv2.imshow('input', frame)

    # Convert all frame to Grayscale (previously we did only the first frame)
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate dense optical flow by Farneback
    flow = cv2.calcOpticalFlowFarneback(p_grey, grey, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    '''
    Gunnar Farneback's algorithm.
calcOpticalFlowFarneback(prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags) -> flow

1. prev first 8-bit single-channel input image.
2. next second input image of the same size and the same type as prev.
3. flow computed flow image that has the same size as prev and type CV_32FC2.
4. pyr_scale parameter, specifying the image scale (<1) to build pyramids for each image
   4a.   pyr_scale=0.5 means a classical pyramid, where each next layer is twice smaller than the previous one.
5. levels number of pyramid layers including the initial image; 
levels=1 means that no extra layers are created and only the original images are used.
6. winsize averaging window size
   6a.  larger values increase the algorithm robustness to image
7. noise and give more chances for fast motion detection, but yield more blurred motion field.
8. iterations number of iterations the algorithm does at each pyramid level.
9. poly_n size of the pixel neighborhood used to find polynomial expansion in each pixel
   9a.   larger values mean that the image will be approximated with smoother surfaces, 
10. yielding more robust algorithm and more blurred motion field, typically poly_n =5 or 7.
11. poly_sigma standard deviation of the Gaussian that is used to smooth derivatives used as a basis for the polynomial 
expansion; for poly_n=5, you can set poly_sigma=1.1, for poly_n=7, a good value would be poly_sigma=1.5.
'''

    # Compute Magnitude and Angle
    magn, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Set image hue depanding on the optical flow direction
    # print(mask[..., 0])
    # print(angle)
    # print(flow)
    mask[..., 0] = angle*180/np.pi/2

    # Normalize the magnitude
    mask[..., 2] = cv2.normalize(magn, None, 0, 255, cv2.NORM_MINMAX)

    # Convert HSV to RGB
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2RGB)

    # Open new window and display the output
    cv2.imshow('Dense Optical Flow', rgb)

    # Update previous frame
    p_grey = grey

    # Close the frame
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release and Destroy
cap.release()
cv2.destroyAllWindows()
