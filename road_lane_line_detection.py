import matplotlib.pylab as plt
import cv2
import numpy as np


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    # channel_count = img.shape[2]
    # print(channel_count)
    match_mask_color = 255      # (255,) * channel_count   # should give (255, 255, 255)
    # print(match_mask_color)
    cv2.fillPoly(mask, vertices, match_mask_color)
    # cv2.imshow('mask', mask)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img


def draw_the_lines(img, lines):
    copy_img = np.copy(img)
    blank_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(copy_img, (x1, y1), (x2, y2), (0, 255, 0), thickness=3)
    img = cv2.addWeighted(copy_img, 0.8, blank_img, 1, 0.0)
    return img


def process(image):
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # print(image.shape)
    height = image.shape[0]
    width = image.shape[1]

    # define our ROI
    region_of_interest_vertices = [
        (0, height),
        (width/2, height/2),
        (width, height)
    ]

    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny_img = cv2.Canny(gray_img, 100, 200)
    cropped_image = region_of_interest(canny_img, np.array([region_of_interest_vertices], np.int32))

    lines = cv2.HoughLinesP(cropped_image, rho=6, theta=np.pi/60, threshold=160,
                            lines=np.array([]), minLineLength=40, maxLineGap=25)
    image_with_lines = draw_the_lines(image, lines)
    return image_with_lines


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    frame = process(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(2) == 27:
        break

cap.release()
cv2.destroyAllWindows()

